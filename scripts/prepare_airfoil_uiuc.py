#!/usr/bin/env python3
"""将 UIUC/Selig 翼型坐标预处理为二维 SDF 数据集。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.airfoil_geometry import (  # noqa: E402
    compute_airfoil_conditions,
    cosine_x_grid,
    normalize_airfoil,
    parse_selig_dat,
    resample_surfaces,
    split_surfaces,
    validate_airfoil,
)
from scripts.airfoil_sdf import rasterize_airfoil_sdf  # noqa: E402


CONDITION_COLUMNS = ["t_max", "x_tmax", "camber_max", "x_camber_max", "te_gap"]


def _sample_id(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _write_npz(
    path: Path,
    sdf: dict[str, np.ndarray],
    coords_raw: np.ndarray,
    coords_resampled: np.ndarray,
    conditions: dict[str, float],
) -> None:
    condition_raw = np.array([conditions[c] for c in CONDITION_COLUMNS], dtype=np.float32)
    np.savez_compressed(
        path,
        sdf2d_norm=sdf["sdf2d_norm"].astype(np.float32),
        semantic_mask=sdf["semantic_mask"].astype(np.uint8),
        coords_raw=coords_raw.astype(np.float32),
        coords_resampled=coords_resampled.astype(np.float32),
        condition_raw=condition_raw,
        condition_json=json.dumps(conditions, ensure_ascii=False),
    )


def prepare_airfoil_uiuc(
    raw_dir: Path,
    output_dir: Path,
    height: int = 128,
    width: int = 256,
    n_resample: int = 257,
    sdf_scale: float = 0.08,
    force: bool = False,
) -> dict[str, int]:
    """批量预处理 UIUC/Selig `.dat` 文件。

    Args:
        raw_dir: 含 `.dat` 文件的目录，可递归搜索。
        output_dir: 输出 processed 目录。
        height: SDF 网格高度。
        width: SDF 网格宽度。
        n_resample: 上下表面重采样点数。
        sdf_scale: SDF 归一化尺度。
        force: 为 True 时允许覆盖已有样本。

    Returns:
        汇总计数字典。
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    sample_dir = output_dir / "airfoil_samples"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw UIUC directory not found: {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    report_items: list[dict[str, Any]] = []
    dat_files = sorted(raw_dir.rglob("*.dat"))
    x_grid = cosine_x_grid(n_resample)

    for dat_path in dat_files:
        sid = _sample_id(dat_path)
        rel_npz = f"{sid}.npz"
        npz_path = sample_dir / rel_npz
        row: dict[str, Any] = {
            "sample_id": sid,
            "npz_file": rel_npz,
            "source_file": str(dat_path),
            "passed_quality": False,
            "num_raw_points": 0,
            "num_resampled_points": n_resample,
        }
        reasons: list[str] = []
        try:
            coords_raw = parse_selig_dat(dat_path)
            row["num_raw_points"] = int(len(coords_raw))
            coords_norm = normalize_airfoil(coords_raw)
            resampled = resample_surfaces(split_surfaces(coords_norm), x_grid)
            conditions = compute_airfoil_conditions(resampled)
            reasons = validate_airfoil(resampled, conditions)
            row.update(conditions)
            if not reasons:
                if force or not npz_path.exists():
                    coords_resampled = np.vstack([resampled.y_upper, resampled.y_lower])
                    sdf = rasterize_airfoil_sdf(
                        resampled,
                        height=height,
                        width=width,
                        sdf_scale=sdf_scale,
                    )
                    _write_npz(npz_path, sdf, coords_raw, coords_resampled, conditions)
                row["passed_quality"] = True
        except Exception as exc:  # noqa: BLE001 - 需要记录单样本失败原因并继续处理
            reasons = [type(exc).__name__, str(exc)]

        for col in CONDITION_COLUMNS:
            row.setdefault(col, np.nan)
        rows.append(row)
        report_items.append({"sample_id": sid, "source_file": str(dat_path), "reasons": reasons})

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "airfoil_index.csv", index=False)
    summary = {
        "total": len(dat_files),
        "passed": int(df["passed_quality"].astype(bool).sum()) if len(df) else 0,
        "failed": int(len(dat_files) - (df["passed_quality"].astype(bool).sum() if len(df) else 0)),
    }
    with open(output_dir / "filter_report.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": report_items}, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UIUC airfoil SDF dataset")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "airfoil" / "raw" / "uiuc" / "coord_seligFmt",
        help="Directory containing UIUC/Selig .dat files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "airfoil_uiuc_sdf" / "processed",
        help="Processed output directory",
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--n-resample", type=int, default=257)
    parser.add_argument("--sdf-scale", type=float, default=0.08)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = prepare_airfoil_uiuc(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        n_resample=args.n_resample,
        sdf_scale=args.sdf_scale,
        force=args.force,
    )
    print(f"Prepared UIUC airfoils: {summary}")


if __name__ == "__main__":
    main()
