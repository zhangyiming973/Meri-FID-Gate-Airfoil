#!/usr/bin/env python3
"""Batch XFOIL polar evaluation for generated and baseline airfoil `.dat` files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.airfoil_geometry import (  # noqa: E402
    ResampledAirfoil,
    cosine_x_grid,
    normalize_airfoil,
    parse_selig_dat,
    resample_surfaces,
    split_surfaces,
)


@dataclass(frozen=True)
class XfoilCase:
    group: str
    sample_id: str
    dat_path: Path


def _dat_coordinates(resampled: ResampledAirfoil) -> list[tuple[float, float]]:
    upper = list(zip(resampled.x[::-1], resampled.y_upper[::-1], strict=True))
    lower = list(zip(resampled.x[1:], resampled.y_lower[1:], strict=True))
    return [(float(x), float(y)) for x, y in upper + lower]


def _smooth_line(y: object, passes: int) -> object:
    import numpy as np

    arr = np.asarray(y, dtype=np.float64).copy()
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    for _ in range(max(0, passes)):
        padded = np.pad(arr, (2, 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        smoothed[0] = arr[0]
        smoothed[-1] = arr[-1]
        arr = smoothed
    return arr


def _prepare_for_xfoil(
    resampled: ResampledAirfoil,
    smooth_passes: int,
    te_gap: float,
    min_thickness: float,
    te_blend_start: float,
) -> ResampledAirfoil:
    import numpy as np

    x = np.asarray(resampled.x, dtype=np.float64)
    camber = 0.5 * (np.asarray(resampled.y_upper) + np.asarray(resampled.y_lower))
    thickness = np.asarray(resampled.y_upper) - np.asarray(resampled.y_lower)
    camber = _smooth_line(camber, smooth_passes)
    thickness = _smooth_line(thickness, smooth_passes)
    thickness = np.maximum(thickness, min_thickness)
    if te_gap >= 0.0:
        start = min(max(te_blend_start, 0.0), 0.995)
        span = max(1.0 - start, 1e-6)
        s = np.clip((x - start) / span, 0.0, 1.0)
        smoothstep = s * s * (3.0 - 2.0 * s)
        te_camber = float(camber[-1])
        thickness = (1.0 - smoothstep) * thickness + smoothstep * te_gap
        camber = (1.0 - smoothstep) * camber + smoothstep * te_camber
    thickness[0] = max(thickness[0], 0.0)
    y_upper = camber + 0.5 * thickness
    y_lower = camber - 0.5 * thickness
    y_upper[0] = y_lower[0] = 0.0
    if te_gap >= 0.0:
        te_camber = 0.5 * (y_upper[-1] + y_lower[-1])
        y_upper[-1] = te_camber + 0.5 * te_gap
        y_lower[-1] = te_camber - 0.5 * te_gap
    return ResampledAirfoil(x=x, y_upper=y_upper, y_lower=y_lower)


def _write_xfoil_dat(
    source_path: Path,
    target_path: Path,
    sample_id: str,
    n_points: int,
    smooth_passes: int,
    te_gap: float,
    min_thickness: float,
    te_blend_start: float,
) -> None:
    coords = parse_selig_dat(source_path)
    resampled = resample_surfaces(split_surfaces(normalize_airfoil(coords)), cosine_x_grid(n_points))
    resampled = _prepare_for_xfoil(resampled, smooth_passes, te_gap, min_thickness, te_blend_start)
    lines = [sample_id]
    lines.extend(f"{x:.8f} {y:.8f}" for x, y in _dat_coordinates(resampled))
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_name(group: str, sample_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in sample_id)
    return f"{group}__{safe}"


def _read_ranked_sample_ids(path: Path, limit: int | None) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ids = [row["sample_id"] for row in reader if row.get("sample_id")]
    return ids[:limit] if limit else ids


def _collect_cases(
    generated_dir: Path,
    baseline_raw_dir: Path,
    ranked_csv: Path,
    limit: int | None,
) -> list[XfoilCase]:
    sample_ids = _read_ranked_sample_ids(ranked_csv, limit)
    cases: list[XfoilCase] = []
    missing: list[str] = []
    baseline_index = {p.stem: p for p in sorted(baseline_raw_dir.rglob("*.dat"))}
    for sample_id in sample_ids:
        generated_path = generated_dir / f"{sample_id}.dat"
        baseline_path = baseline_index.get(sample_id)
        if not generated_path.exists():
            missing.append(f"generated:{sample_id}")
            continue
        if baseline_path is None:
            missing.append(f"baseline:{sample_id}")
            continue
        cases.append(XfoilCase("generated", sample_id, generated_path))
        cases.append(XfoilCase("baseline", sample_id, baseline_path))
    if missing:
        print(f"Skipped {len(missing)} missing files: {', '.join(missing[:8])}", file=sys.stderr)
    return cases


def _write_xfoil_input(
    case: XfoilCase,
    local_dat_name: str,
    polar_name: str,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    reynolds: float,
    mach: float,
    ncrit: float,
    max_iter: int,
) -> str:
    return "\n".join(
        [
            "PLOP",
            "G",
            "",
            f"LOAD {local_dat_name}",
            "PANE",
            "OPER",
            f"VISC {reynolds:.0f}",
            f"MACH {mach:.4f}",
            "VPAR",
            f"N {ncrit:.3f}",
            "",
            f"ITER {max_iter}",
            "PACC",
            polar_name,
            "",
            f"ASEQ {alpha_start:.6g} {alpha_end:.6g} {alpha_step:.6g}",
            "PACC",
            "",
            "QUIT",
            "",
        ]
    )


def _parse_polar(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            values = [float(x) for x in parts[:7]]
        except ValueError:
            continue
        rows.append(
            {
                "alpha": values[0],
                "cl": values[1],
                "cd": values[2],
                "cdp": values[3],
                "cm": values[4],
                "top_xtr": values[5],
                "bot_xtr": values[6],
            }
        )
    return rows


def _summarize(case: XfoilCase, rows: list[dict[str, float]], expected_rows: int) -> dict[str, float | str | int | bool]:
    summary: dict[str, float | str | int | bool] = {
        "group": case.group,
        "sample_id": case.sample_id,
        "dat_path": str(case.dat_path),
        "num_points": len(rows),
        "expected_points": expected_rows,
        "converged_all": len(rows) == expected_rows,
        "converged_any": bool(rows),
    }
    if not rows:
        return summary
    valid_ld = [r | {"ld": r["cl"] / r["cd"]} for r in rows if r["cd"] > 0 and math.isfinite(r["cd"])]
    best_ld = max(valid_ld, key=lambda r: r["ld"]) if valid_ld else None
    max_cl = max(rows, key=lambda r: r["cl"])
    min_cd = min(rows, key=lambda r: r["cd"])
    summary.update(
        {
            "best_ld": float(best_ld["ld"]) if best_ld else math.nan,
            "alpha_at_best_ld": float(best_ld["alpha"]) if best_ld else math.nan,
            "cl_at_best_ld": float(best_ld["cl"]) if best_ld else math.nan,
            "cd_at_best_ld": float(best_ld["cd"]) if best_ld else math.nan,
            "max_cl": float(max_cl["cl"]),
            "alpha_at_max_cl": float(max_cl["alpha"]),
            "min_cd": float(min_cd["cd"]),
            "alpha_at_min_cd": float(min_cd["alpha"]),
        }
    )
    return summary


def _run_case(
    xfoil_bin: Path,
    case: XfoilCase,
    out_dir: Path,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    reynolds: float,
    mach: float,
    ncrit: float,
    max_iter: int,
    timeout: int,
    xfoil_points: int,
    smooth_passes: int,
    te_gap: float,
    min_thickness: float,
    te_blend_start: float,
) -> dict[str, float | str | int | bool]:
    case_dir = out_dir / "cases" / _case_name(case.group, case.sample_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    local_dat = case_dir / "airfoil.dat"
    polar_path = case_dir / "polar.dat"
    stdin_path = case_dir / "xfoil.in"
    stdout_path = case_dir / "xfoil.log"
    polar_path.unlink(missing_ok=True)
    _write_xfoil_dat(
        case.dat_path,
        local_dat,
        case.sample_id,
        xfoil_points,
        smooth_passes,
        te_gap,
        min_thickness,
        te_blend_start,
    )
    xfoil_input = _write_xfoil_input(
        case,
        local_dat.name,
        polar_path.name,
        alpha_start,
        alpha_end,
        alpha_step,
        reynolds,
        mach,
        ncrit,
        max_iter,
    )
    stdin_path.write_text(xfoil_input, encoding="utf-8")
    start = time.time()
    try:
        result = subprocess.run(
            [str(xfoil_bin)],
            input=xfoil_input,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd=case_dir,
        )
        stdout_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8")
        return_code = result.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text((exc.stdout or "") + "\n--- TIMEOUT ---\n" + (exc.stderr or ""), encoding="utf-8")
        return_code = 124
        timed_out = True
    rows = _parse_polar(polar_path)
    expected_rows = int(round((alpha_end - alpha_start) / alpha_step)) + 1
    summary = _summarize(case, rows, expected_rows)
    summary.update(
        {
            "return_code": return_code,
            "timed_out": timed_out,
            "runtime_sec": round(time.time() - start, 3),
            "polar_path": str(polar_path),
            "log_path": str(stdout_path),
        }
    )
    with open(case_dir / "polar.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _paired_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_id.setdefault(str(row["sample_id"]), {})[str(row["group"])] = row
    paired: list[dict[str, object]] = []
    metrics = ["best_ld", "max_cl", "min_cd", "alpha_at_best_ld", "cl_at_best_ld", "cd_at_best_ld"]
    for sample_id, groups in sorted(by_id.items()):
        gen = groups.get("generated")
        base = groups.get("baseline")
        if not gen or not base:
            continue
        out: dict[str, object] = {"sample_id": sample_id}
        for metric in metrics:
            gv = gen.get(metric, math.nan)
            bv = base.get(metric, math.nan)
            out[f"generated_{metric}"] = gv
            out[f"baseline_{metric}"] = bv
            try:
                out[f"delta_{metric}"] = float(gv) - float(bv)
            except (TypeError, ValueError):
                out[f"delta_{metric}"] = math.nan
        out["generated_points"] = gen.get("num_points", 0)
        out["baseline_points"] = base.get("num_points", 0)
        paired.append(out)
    return paired


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else math.nan


def _write_report(out_dir: Path, rows: list[dict[str, object]], paired: list[dict[str, object]], metadata: dict[str, object]) -> None:
    def group_rows(group: str) -> list[dict[str, object]]:
        return [r for r in rows if r["group"] == group and r.get("converged_any")]

    generated = group_rows("generated")
    baseline = group_rows("baseline")
    report = {
        "metadata": metadata,
        "counts": {
            "generated_cases": len([r for r in rows if r["group"] == "generated"]),
            "baseline_cases": len([r for r in rows if r["group"] == "baseline"]),
            "generated_converged_any": len(generated),
            "baseline_converged_any": len(baseline),
            "paired_cases": len(paired),
        },
        "means": {
            "generated_best_ld": _mean([float(r.get("best_ld", math.nan)) for r in generated]),
            "baseline_best_ld": _mean([float(r.get("best_ld", math.nan)) for r in baseline]),
            "generated_max_cl": _mean([float(r.get("max_cl", math.nan)) for r in generated]),
            "baseline_max_cl": _mean([float(r.get("max_cl", math.nan)) for r in baseline]),
            "delta_best_ld": _mean([float(r.get("delta_best_ld", math.nan)) for r in paired]),
            "delta_max_cl": _mean([float(r.get("delta_max_cl", math.nan)) for r in paired]),
        },
        "top_generated_by_ld": sorted(generated, key=lambda r: float(r.get("best_ld", -1e9)), reverse=True)[:10],
        "top_pair_deltas_by_ld": sorted(paired, key=lambda r: float(r.get("delta_best_ld", -1e9)), reverse=True)[:10],
    }
    (out_dir / "xfoil_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# XFOIL Airfoil Evaluation",
        "",
        f"- XFOIL: `{metadata['xfoil_bin']}`",
        f"- Re: `{metadata['reynolds']}`",
        f"- Mach: `{metadata['mach']}`",
        f"- Alpha sweep: `{metadata['alpha_start']}:{metadata['alpha_step']}:{metadata['alpha_end']}`",
        f"- Ncrit: `{metadata['ncrit']}`",
        f"- Max iter: `{metadata['max_iter']}`",
        "",
        "## Summary",
        "",
        f"- Generated converged: {report['counts']['generated_converged_any']}/{report['counts']['generated_cases']}",
        f"- Baseline converged: {report['counts']['baseline_converged_any']}/{report['counts']['baseline_cases']}",
        f"- Mean generated best L/D: {report['means']['generated_best_ld']:.3f}",
        f"- Mean baseline best L/D: {report['means']['baseline_best_ld']:.3f}",
        f"- Mean delta best L/D: {report['means']['delta_best_ld']:.3f}",
        f"- Mean generated max CL: {report['means']['generated_max_cl']:.3f}",
        f"- Mean baseline max CL: {report['means']['baseline_max_cl']:.3f}",
        f"- Mean delta max CL: {report['means']['delta_max_cl']:.3f}",
        "",
        "## Top Generated By L/D",
        "",
        "| sample_id | best L/D | alpha | CL | CD |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["top_generated_by_ld"]:
        lines.append(
            f"| {row['sample_id']} | {float(row.get('best_ld', math.nan)):.3f} | "
            f"{float(row.get('alpha_at_best_ld', math.nan)):.2f} | "
            f"{float(row.get('cl_at_best_ld', math.nan)):.4f} | "
            f"{float(row.get('cd_at_best_ld', math.nan)):.5f} |"
        )
    lines.extend(["", "## Best Generated Minus Baseline L/D Deltas", "", "| sample_id | delta L/D | generated L/D | baseline L/D |", "|---|---:|---:|---:|"])
    for row in report["top_pair_deltas_by_ld"]:
        lines.append(
            f"| {row['sample_id']} | {float(row.get('delta_best_ld', math.nan)):.3f} | "
            f"{float(row.get('generated_best_ld', math.nan)):.3f} | "
            f"{float(row.get('baseline_best_ld', math.nan)):.3f} |"
        )
    (out_dir / "xfoil_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XFOIL polar sweeps for generated and baseline airfoils.")
    parser.add_argument("--xfoil-bin", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--baseline-raw-dir", type=Path, required=True)
    parser.add_argument("--ranked-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--re", type=float, default=1_000_000.0, dest="reynolds")
    parser.add_argument("--mach", type=float, default=0.10)
    parser.add_argument("--alpha-start", type=float, default=-2.0)
    parser.add_argument("--alpha-end", type=float, default=8.0)
    parser.add_argument("--alpha-step", type=float, default=1.0)
    parser.add_argument("--ncrit", type=float, default=9.0)
    parser.add_argument("--iter", type=int, default=120, dest="max_iter")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--xfoil-points", type=int, default=161)
    parser.add_argument("--smooth-passes", type=int, default=2)
    parser.add_argument("--te-gap", type=float, default=0.002)
    parser.add_argument("--te-blend-start", type=float, default=0.95)
    parser.add_argument("--min-thickness", type=float, default=1e-5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.xfoil_bin.exists():
        raise FileNotFoundError(f"XFOIL binary not found: {args.xfoil_bin}")
    if args.out_dir.exists() and args.force:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = _collect_cases(args.generated_dir, args.baseline_raw_dir, args.ranked_csv, args.limit)
    rows: list[dict[str, object]] = []
    workers = max(1, min(args.workers, len(cases) or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_case,
                args.xfoil_bin,
                case,
                args.out_dir,
                args.alpha_start,
                args.alpha_end,
                args.alpha_step,
                args.reynolds,
                args.mach,
                args.ncrit,
                args.max_iter,
                args.timeout,
                args.xfoil_points,
                args.smooth_passes,
                args.te_gap,
                args.min_thickness,
                args.te_blend_start,
            ): case
            for case in cases
        }
        for i, future in enumerate(as_completed(futures), 1):
            case = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve batch progress across bad cases
                row = {
                    "group": case.group,
                    "sample_id": case.sample_id,
                    "dat_path": str(case.dat_path),
                    "num_points": 0,
                    "expected_points": int(round((args.alpha_end - args.alpha_start) / args.alpha_step)) + 1,
                    "converged_all": False,
                    "converged_any": False,
                    "return_code": -1,
                    "timed_out": False,
                    "runtime_sec": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            status = "ok" if row.get("converged_any") else "no-polar"
            if row.get("timed_out"):
                status = "timeout"
            print(f"[{i}/{len(cases)}] {case.group} {case.sample_id}: {status}, points={row.get('num_points', 0)}", flush=True)
    rows.sort(key=lambda r: (str(r["sample_id"]), str(r["group"])))
    paired = _paired_rows(rows)
    _write_csv(args.out_dir / "xfoil_summary.csv", rows)
    _write_csv(args.out_dir / "xfoil_paired_comparison.csv", paired)
    metadata = {
        "xfoil_bin": str(args.xfoil_bin),
        "generated_dir": str(args.generated_dir),
        "baseline_raw_dir": str(args.baseline_raw_dir),
        "ranked_csv": str(args.ranked_csv),
        "limit": args.limit,
        "reynolds": args.reynolds,
        "mach": args.mach,
        "alpha_start": args.alpha_start,
        "alpha_end": args.alpha_end,
        "alpha_step": args.alpha_step,
        "ncrit": args.ncrit,
        "max_iter": args.max_iter,
        "workers": workers,
        "xfoil_points": args.xfoil_points,
        "smooth_passes": args.smooth_passes,
        "te_gap": args.te_gap,
        "te_blend_start": args.te_blend_start,
        "min_thickness": args.min_thickness,
    }
    _write_report(args.out_dir, rows, paired, metadata)
    print(f"Wrote XFOIL results to {args.out_dir}")


if __name__ == "__main__":
    main()
