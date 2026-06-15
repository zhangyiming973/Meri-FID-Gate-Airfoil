"""从生成 SDF 提取翼型轮廓，进行几何校核与条件一致性评估。

用法:
    python scripts/airfoil/evaluate_generated_airfoils.py \
        --ae-run-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_110639/autoencoder \
        --diff-run-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion \
        --out-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# 将项目根加入 path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.airfoil_geometry import (
    ResampledAirfoil,
    compute_airfoil_conditions,
    cosine_x_grid,
    normalize_airfoil,
    resample_surfaces,
    split_surfaces,
    validate_airfoil,
)
from schemes.pca_unet.data.dataset import ConditionStats
from schemes.pca_unet.models.diffusion import DiffusionSchedule
from schemes.pca_unet.models.latent_pca import load_latent_pca
from schemes.pca_unet.records.body_latent import load_records
from schemes.pca_unet.train.diffusion_codec import DiffusionLatentCodec
from schemes.pca_unet.train.train_diffusion import (
    build_pca_unet_for_conditions,
    load_autoencoder_from_checkpoint,
)

# ── SDF → 轮廓提取 ──────────────────────────────────────────────

import matplotlib.pyplot as plt


def _extract_largest_contour(
    sdf: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray | None:
    """用 matplotlib contour 从 SDF 提取零水平集，返回最大闭合轮廓 (N, 2)。"""
    if sdf.min() >= 0 or sdf.max() <= 0:
        return None

    fig = plt.figure()
    cs = plt.contour(grid_x, grid_y, sdf, levels=[0.0])
    paths = cs.get_paths()
    plt.close(fig)

    if not paths:
        return None

    # 选点数最多的路径
    best = None
    best_len = 0
    for path in paths:
        verts = path.vertices
        if len(verts) > best_len:
            best_len = len(verts)
            best = verts.copy()

    if best is None or len(best) < 8:
        return None

    # 确保闭合
    if not np.allclose(best[0], best[-1]):
        best = np.vstack([best, best[:1]])

    return best.astype(np.float64)


def _contour_to_airfoil(
    contour: np.ndarray,
    n_points: int = 257,
) -> ResampledAirfoil | None:
    """将轮廓转换为归一化翼型上下表面表示。"""
    try:
        # 归一化到弦长坐标
        normed = normalize_airfoil(contour)
        # 拆分上下表面
        surfaces = split_surfaces(normed)
        x_grid = cosine_x_grid(n_points)
        resampled = resample_surfaces(surfaces, x_grid)
        return resampled
    except Exception:
        return None


def _interp_sdf_column(sdf: np.ndarray, grid_x: np.ndarray, x_value: float) -> np.ndarray:
    """Interpolate an SDF vertical column at a physical x coordinate."""
    x_axis = grid_x[0]
    if x_value < x_axis[0] or x_value > x_axis[-1]:
        return np.full(sdf.shape[0], np.nan, dtype=np.float64)
    cols = np.arange(sdf.shape[1], dtype=np.float64)
    col = float(np.interp(x_value, x_axis, cols))
    left = int(np.floor(col))
    right = min(left + 1, sdf.shape[1] - 1)
    weight = col - left
    return (1.0 - weight) * sdf[:, left] + weight * sdf[:, right]


def _zero_crossings(y_axis: np.ndarray, values: np.ndarray) -> list[float]:
    """Return y positions where an SDF column crosses zero."""
    crossings: list[float] = []
    finite = np.isfinite(values)
    for i in range(len(values) - 1):
        if not finite[i] or not finite[i + 1]:
            continue
        v0 = float(values[i])
        v1 = float(values[i + 1])
        if v0 == 0.0:
            crossings.append(float(y_axis[i]))
        if v0 * v1 < 0.0:
            denom = abs(v0) + abs(v1)
            t = abs(v0) / denom if denom > 0.0 else 0.5
            crossings.append(float((1.0 - t) * y_axis[i] + t * y_axis[i + 1]))
    if finite[-1] and float(values[-1]) == 0.0:
        crossings.append(float(y_axis[-1]))
    return sorted(crossings)


def _surface_pair_from_sdf_column(y_axis: np.ndarray, values: np.ndarray) -> tuple[float, float] | None:
    """Pick the main airfoil lower/upper zero crossings from one SDF column."""
    crossings = _zero_crossings(y_axis, values)
    if len(crossings) >= 2:
        pairs = list(zip(crossings[0::2], crossings[1::2]))
        if len(crossings) % 2 == 1:
            pairs.append((crossings[0], crossings[-1]))
        lower, upper = max(pairs, key=lambda p: p[1] - p[0])
        if upper > lower:
            return lower, upper

    inside = np.where(np.isfinite(values) & (values <= 0.0))[0]
    if len(inside) >= 2:
        return float(y_axis[inside[0]]), float(y_axis[inside[-1]])
    return None


def _fill_nan_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    finite = np.isfinite(y)
    if finite.sum() < 2:
        raise ValueError("not enough finite samples")
    return np.interp(x, x[finite], y[finite]).astype(np.float64)


def _despike_line(y: np.ndarray, passes: int = 3, threshold: float = 6.0) -> np.ndarray:
    """Replace isolated second-difference spikes with local linear estimates."""
    arr = np.asarray(y, dtype=np.float64).copy()
    for _ in range(max(0, passes)):
        if len(arr) < 5:
            break
        d2 = arr[:-2] - 2.0 * arr[1:-1] + arr[2:]
        med = float(np.median(d2))
        mad = float(np.median(np.abs(d2 - med)))
        scale = 1.4826 * mad
        if scale < 1e-8:
            scale = float(np.percentile(np.abs(d2 - med), 90))
        if scale < 1e-8:
            break
        spike_idx = np.where(np.abs(d2 - med) > threshold * scale)[0] + 1
        if len(spike_idx) == 0:
            break
        for idx in spike_idx:
            if 0 < idx < len(arr) - 1:
                arr[idx] = 0.5 * (arr[idx - 1] + arr[idx + 1])
    return arr


def _sample_sdf_airfoil(
    sdf: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    contour: np.ndarray,
    n_points: int = 257,
) -> ResampledAirfoil:
    """Sample upper/lower surfaces directly from SDF zero crossings.

    The contour is used only to estimate the chordwise extent.  Surface points
    are then taken from independent vertical SDF cuts, avoiding contour vertex
    ordering jumps.
    """
    finite_contour = contour[np.isfinite(contour).all(axis=1)]
    if len(finite_contour) < 8:
        raise ValueError("contour has too few finite points")
    x_min = float(np.percentile(finite_contour[:, 0], 0.5))
    x_max = float(np.percentile(finite_contour[:, 0], 99.5))
    chord = x_max - x_min
    if chord <= 1e-6:
        raise ValueError("sampled chord must be positive")

    x_norm = cosine_x_grid(n_points)
    x_phys = x_min + chord * x_norm
    y_axis = grid_y[:, 0]
    y_lower = np.full(n_points, np.nan, dtype=np.float64)
    y_upper = np.full(n_points, np.nan, dtype=np.float64)
    for i, xp in enumerate(x_phys):
        pair = _surface_pair_from_sdf_column(y_axis, _interp_sdf_column(sdf, grid_x, float(xp)))
        if pair is None:
            continue
        y_lower[i], y_upper[i] = pair

    y_lower = _fill_nan_line(x_norm, y_lower)
    y_upper = _fill_nan_line(x_norm, y_upper)
    camber = 0.5 * (y_upper + y_lower)
    thickness = y_upper - y_lower
    camber = _despike_line(camber)
    thickness = _despike_line(thickness)
    thickness = np.maximum(thickness, 0.0)

    leading_camber = float(camber[0])
    trailing_camber = float(camber[-1])
    chordline = leading_camber + (trailing_camber - leading_camber) * x_norm
    camber = camber - chordline
    y_upper_norm = camber / chord + 0.5 * thickness / chord
    y_lower_norm = camber / chord - 0.5 * thickness / chord
    y_upper_norm[0] = y_lower_norm[0] = 0.0
    return ResampledAirfoil(x=x_norm, y_upper=y_upper_norm, y_lower=y_lower_norm)


def _check_airfoil_validity(resampled: ResampledAirfoil) -> tuple[bool, list[str]]:
    """检查生成翼型的几何合法性。"""
    issues: list[str] = []

    # 厚度检查
    thickness = resampled.y_upper - resampled.y_lower
    neg_frac = float(np.mean(thickness < -1e-6))
    if neg_frac > 0.01:
        issues.append(f"negative_thickness_fraction={neg_frac:.4f}")

    # 基本条件检查
    cond = compute_airfoil_conditions(resampled)
    if not (0.005 <= cond["t_max"] <= 0.40):
        issues.append(f"t_max_out_of_range={cond['t_max']:.4f}")
    if cond["te_gap"] < -0.005:
        issues.append(f"te_gap_negative={cond['te_gap']:.6f}")
    if abs(cond["camber_max"]) > 0.25:
        issues.append(f"camber_max_excessive={cond['camber_max']:.4f}")
    if cond["x_tmax"] < 0.0 or cond["x_tmax"] > 1.0:
        issues.append(f"x_tmax_out_of_range={cond['x_tmax']:.4f}")

    # 前缘/尾缘合理性
    le_thickness = float(np.interp(0.02, resampled.x, thickness))
    if le_thickness <= 0:
        issues.append("leading_edge_collapsed")

    is_valid = len(issues) == 0
    return is_valid, issues


def _condition_score(measured: dict[str, float], target: dict[str, float], is_valid: bool = True) -> float:
    """按归一化几何误差给候选翼型打分，越小越好。"""
    tolerances = {
        "t_max": 0.02,
        "camber_max": 0.02,
        "x_tmax": 0.08,
        "x_camber_max": 0.10,
        "te_gap": 0.01,
    }
    weights = {
        "t_max": 1.0,
        "camber_max": 1.0,
        "x_tmax": 0.7,
        "x_camber_max": 0.7,
        "te_gap": 0.5,
    }
    score = 0.0
    for key, tol in tolerances.items():
        if key not in measured or key not in target:
            score += 100.0
            continue
        score += weights[key] * abs(float(measured[key]) - float(target[key])) / tol
    if not is_valid:
        score += 1000.0
    return float(score)


def _dat_coordinates(resampled: ResampledAirfoil) -> np.ndarray:
    """转换为 UIUC/XFOIL 常用坐标顺序：upper TE→LE，再 lower LE→TE。"""
    upper = np.column_stack([resampled.x[::-1], resampled.y_upper[::-1]])
    lower = np.column_stack([resampled.x[1:], resampled.y_lower[1:]])
    return np.vstack([upper, lower]).astype(np.float64)


def _write_airfoil_dat(path: Path, name: str, resampled: ResampledAirfoil) -> None:
    """导出清晰化翼型坐标为 `.dat`。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = _dat_coordinates(resampled)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"{x:.8f} {y:.8f}\n")


def _candidate_from_sdf(
    sdf: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    target_cond: dict[str, float],
    n_points: int,
) -> dict:
    contour = _extract_largest_contour(sdf, grid_x, grid_y)
    airfoil = None
    is_valid = False
    issues: list[str] = ["no_contour"]
    measured_cond: dict[str, float] = {}
    if contour is not None and len(contour) >= 10:
        try:
            airfoil = _sample_sdf_airfoil(sdf, grid_x, grid_y, contour, n_points)
        except Exception:
            airfoil = _contour_to_airfoil(contour, n_points)
        if airfoil is not None:
            is_valid, issues = _check_airfoil_validity(airfoil)
            measured_cond = compute_airfoil_conditions(airfoil)
        else:
            issues = ["contour_to_airfoil_failed"]
    errors = {k: float(measured_cond[k] - target_cond[k]) for k in target_cond if k in measured_cond}
    score = _condition_score(measured_cond, target_cond, is_valid)
    return {
        "airfoil": airfoil,
        "contour": contour,
        "is_valid": is_valid,
        "issues": issues,
        "has_contour": contour is not None and len(contour) >= 10,
        "has_airfoil": airfoil is not None,
        "measured_conditions": measured_cond,
        "condition_errors": errors,
        "condition_score": score,
    }


# ── 主评估逻辑 ──────────────────────────────────────────────────


def evaluate(
    ae_run_dir: Path,
    diff_run_dir: Path,
    out_dir: Path,
    n_points: int = 257,
    cfg_scale: float = 1.5,
    sample_steps: int = 50,
    candidates_per_condition: int = 1,
    export_top_k: int = 0,
    limit_samples: int | None = None,
) -> dict:
    """完整几何评估管线。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 加载模型 ──
    ae_cfg_path = ae_run_dir / "condition_stats.json"
    if not ae_cfg_path.exists():
        raise FileNotFoundError(f"AE config not found: {ae_cfg_path}")
    ae_full_cfg = json.loads((Path(_project_root) / "schemes" / "pca_unet" / "config" / "airfoil_uiuc_sdf.json").read_text())
    ae = load_autoencoder_from_checkpoint(ae_run_dir / "best_autoencoder.pt", ae_full_cfg["autoencoder"], device)

    pca = load_latent_pca(diff_run_dir / "latent_pca.json")
    unet_cfg = ae_full_cfg.get("unet", {})
    cond_stats = ConditionStats.from_dict(json.loads((ae_run_dir / "condition_stats.json").read_text()))
    unet, _, = build_pca_unet_for_conditions(pca.dim, cond_stats, base_ch=unet_cfg.get("unet_base_ch", 64))
    ckpt = torch.load(diff_run_dir / "best_unet_diffusion.pt", map_location=device, weights_only=False)
    unet.load_state_dict(ckpt["model"])
    unet = unet.to(device)
    unet.eval()
    ae.eval()

    # ── 加载测试数据 ──
    test_recs = load_records(ae_run_dir / "body_latent_test.json")
    if limit_samples is not None:
        test_recs = test_recs[: max(0, limit_samples)]
    if not test_recs:
        raise ValueError("No test records selected for geometry evaluation")
    z_test_np = np.stack([r.z_m for r in test_recs])
    codec = DiffusionLatentCodec.from_mode("pca_unet", pca, z_test_np.shape)
    schedule = DiffusionSchedule(timesteps=unet_cfg.get("timesteps", 200), device=device)

    # Grid 信息（用于 SDF 坐标反算）
    sdf_h, sdf_w = ae_full_cfg["autoencoder"]["input_size"]
    x_bounds = (-0.05, 1.05)
    y_bounds = (-0.25, 0.25)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_bounds[0], x_bounds[1], sdf_w, dtype=np.float64),
        np.linspace(y_bounds[0], y_bounds[1], sdf_h, dtype=np.float64),
    )

    sample_shape = codec.sample_shape(1)
    results: list[dict] = []
    top_dir = out_dir / "top_dat"
    candidates_dir = out_dir / "candidate_dat"

    print(f"Evaluating {len(test_recs)} test samples with {candidates_per_condition} candidate(s) each...")
    for idx, r in enumerate(test_recs):
        cond_np = r.condition_norm.astype(np.float32)
        cond = torch.from_numpy(cond_np).unsqueeze(0).to(device)

        target_cond = dict(zip(cond_stats.columns, r.condition_raw.tolist()))
        candidates: list[dict] = []
        with torch.no_grad():
            z_gt = torch.from_numpy(r.z_m).unsqueeze(0).to(device)
            sdf_gt = ae.decode(z_gt)[0, 0].cpu().numpy()
            for cand_idx in range(candidates_per_condition):
                z_gen_enc = schedule.sample(
                    unet, sample_shape, cond,
                    cfg_scale=cfg_scale,
                    steps=sample_steps,
                    use_ddim=True,
                )
                z_gen = codec.decode_to_raw(z_gen_enc)
                sdf_gen = ae.decode(z_gen)[0, 0].cpu().numpy()
                cand = _candidate_from_sdf(sdf_gen, grid_x, grid_y, target_cond, n_points)
                cand["candidate_index"] = cand_idx
                cand["gen_l1_vs_gt"] = float(np.mean(np.abs(sdf_gen - sdf_gt)))
                candidates.append(cand)

        candidates.sort(key=lambda c: c["condition_score"])
        best = candidates[0]
        airfoil = best["airfoil"]
        if export_top_k > 0:
            for rank, cand in enumerate(candidates[:export_top_k], start=1):
                if cand["airfoil"] is None:
                    continue
                _write_airfoil_dat(
                    candidates_dir / r.sample_id / f"rank{rank:02d}_cand{cand['candidate_index']:03d}.dat",
                    f"{r.sample_id}_rank{rank:02d}_score{cand['condition_score']:.3f}",
                    cand["airfoil"],
                )
            if airfoil is not None:
                _write_airfoil_dat(
                    top_dir / f"{r.sample_id}.dat",
                    f"{r.sample_id}_best_score{best['condition_score']:.3f}",
                    airfoil,
                )

        results.append({
            "sample_id": r.sample_id,
            "is_valid": best["is_valid"],
            "issues": best["issues"],
            "has_contour": best["has_contour"],
            "has_airfoil": best["has_airfoil"],
            "target_conditions": target_cond,
            "measured_conditions": best["measured_conditions"],
            "condition_errors": best["condition_errors"],
            "condition_score": best["condition_score"],
            "best_candidate_index": best["candidate_index"],
            "num_candidates": candidates_per_condition,
            "gen_l1_vs_gt": best["gen_l1_vs_gt"],
        })

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(test_recs)}")

    # ── 汇总统计 ──
    valid_results = [r for r in results if r["is_valid"]]
    has_contour_count = sum(1 for r in results if r["has_contour"])
    has_airfoil_count = sum(1 for r in results if r["has_airfoil"])

    # 条件误差统计
    cond_mae: dict[str, float] = {}
    for k in cond_stats.columns:
        errors = [r["condition_errors"].get(k, 0.0) for r in valid_results]
        if errors:
            cond_mae[k] = float(np.mean(np.abs(errors)))

    # 条件成功率（严格/宽松）
    strict_pass = 0
    loose_pass = 0
    for r in valid_results:
        err = r["condition_errors"]
        if all(abs(err.get(k, 999)) <= {"t_max": 0.01, "x_tmax": 0.08, "camber_max": 0.01, "x_camber_max": 0.10, "te_gap": 0.01}.get(k, 0.02) for k in cond_stats.columns):
            strict_pass += 1
        if all(abs(err.get(k, 999)) <= {"t_max": 0.03, "x_tmax": 0.15, "camber_max": 0.03, "x_camber_max": 0.15, "te_gap": 0.03}.get(k, 0.05) for k in cond_stats.columns):
            loose_pass += 1

    n_valid = len(valid_results)
    report = {
        "num_test_samples": len(test_recs),
        "num_valid_airfoils": n_valid,
        "num_with_contour": has_contour_count,
        "num_with_airfoil": has_airfoil_count,
        "validity_rate": n_valid / len(test_recs) if test_recs else 0.0,
        "condition_mae": cond_mae,
        "condition_success_strict": strict_pass,
        "condition_success_strict_rate": strict_pass / n_valid if n_valid > 0 else 0.0,
        "condition_success_loose": loose_pass,
        "condition_success_loose_rate": loose_pass / n_valid if n_valid > 0 else 0.0,
        "mean_gen_l1": float(np.mean([r["gen_l1_vs_gt"] for r in results])),
        "failure_modes": _summarize_issues(results),
        "cfg_scale": cfg_scale,
        "sample_steps": sample_steps,
        "candidates_per_condition": candidates_per_condition,
        "export_top_k": export_top_k,
        "limit_samples": limit_samples,
    }

    # ── 保存 ──
    save_json(report, out_dir / "geometry_eval_report.json")
    # 保存逐样本详情（CSV）
    _save_per_sample_csv(results, cond_stats.columns, out_dir / "per_sample_geometry.csv")
    _save_rerank_csv(results, cond_stats.columns, out_dir / "rerank_summary.csv")
    # 保存汇总
    print(f"\n=== Geometry Evaluation Summary ===")
    print(f"  Valid airfoils:  {n_valid}/{len(test_recs)} ({report['validity_rate']:.1%})")
    print(f"  With contours:   {has_contour_count}/{len(test_recs)}")
    print(f"  Mean gen L1:     {report['mean_gen_l1']:.6f}")
    print(f"  Candidates each: {candidates_per_condition}")
    print(f"  Condition MAE:")
    for k, v in cond_mae.items():
        print(f"    {k:20s}: {v:.6f}")
    print(f"  Strict success:  {strict_pass}/{n_valid} ({report['condition_success_strict_rate']:.1%})" if n_valid > 0 else "  Strict success: N/A")
    print(f"  Loose success:   {loose_pass}/{n_valid} ({report['condition_success_loose_rate']:.1%})" if n_valid > 0 else "  Loose success: N/A")
    print(f"\n  Failure modes:")
    for mode, count in report["failure_modes"].items():
        print(f"    {mode}: {count}")
    print(f"\n  Full report: {out_dir / 'geometry_eval_report.json'}")

    return report


def _summarize_issues(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        for issue in r["issues"]:
            issue_type = issue.split("=")[0] if "=" in issue else issue
            counts[issue_type] = counts.get(issue_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _save_per_sample_csv(results: list[dict], cond_columns: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_id", "is_valid", "has_contour", "has_airfoil", "gen_l1",
                  *[f"target_{c}" for c in cond_columns],
                  *[f"measured_{c}" for c in cond_columns],
                  *[f"error_{c}" for c in cond_columns],
                  "issues"]
        writer.writerow(header)
        for r in results:
            row = [
                r["sample_id"], r["is_valid"], r["has_contour"], r["has_airfoil"],
                r["gen_l1_vs_gt"],
                *[r["target_conditions"].get(c, "") for c in cond_columns],
                *[r["measured_conditions"].get(c, "") for c in cond_columns],
                *[r["condition_errors"].get(c, "") for c in cond_columns],
                "; ".join(r["issues"]),
            ]
            writer.writerow(row)


def _save_rerank_csv(results: list[dict], cond_columns: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "sample_id", "condition_score", "best_candidate_index", "num_candidates", "is_valid",
            *[f"abs_error_{c}" for c in cond_columns],
        ]
        writer.writerow(header)
        for r in sorted(results, key=lambda x: x["condition_score"]):
            writer.writerow([
                r["sample_id"],
                r["condition_score"],
                r["best_candidate_index"],
                r["num_candidates"],
                r["is_valid"],
                *[abs(r["condition_errors"].get(c, float("nan"))) for c in cond_columns],
            ])


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))


# ── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated airfoils geometry")
    parser.add_argument("--ae-run-dir", required=True, help="AE training directory")
    parser.add_argument("--diff-run-dir", required=True, help="Diffusion training directory")
    parser.add_argument("--out-dir", required=True, help="Output directory for evaluation")
    parser.add_argument("--n-points", type=int, default=257, help="Resample points")
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="CFG scale for sampling")
    parser.add_argument("--sample-steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--candidates-per-condition", type=int, default=1, help="Number of candidates sampled per test condition")
    parser.add_argument("--export-top-k", type=int, default=0, help="Export top-k cleaned candidates as .dat files")
    parser.add_argument("--limit-samples", type=int, help="Evaluate only the first N test samples")
    args = parser.parse_args()
    evaluate(
        Path(args.ae_run_dir),
        Path(args.diff_run_dir),
        Path(args.out_dir),
        n_points=args.n_points,
        cfg_scale=args.cfg_scale,
        sample_steps=args.sample_steps,
        candidates_per_condition=args.candidates_per_condition,
        export_top_k=args.export_top_k,
        limit_samples=args.limit_samples,
    )


if __name__ == "__main__":
    main()
