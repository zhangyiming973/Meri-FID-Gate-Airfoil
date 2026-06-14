#!/usr/bin/env python3
"""Meridian SDF 扩散项目统一命令行入口。

本模块将数据集准备、训练与测试/可视化三条工作流收敛到单一 CLI（``run.py``），
通过 ``--scheme`` 在 MLP 去噪器与 PCA-UNet 两套方案之间切换。

子命令：
    prepare-airfoil-uiuc  将 UIUC 坐标预处理为翼型 SDF 数据集
    prepare-splits  生成固定的 train/test 划分 CSV
    train           训练自编码器与扩散模型（可分阶段）
    test            对已完成的 run 目录做评估与可视化

用法示例::

    python run.py prepare-splits --dataset single
    python run.py train --scheme mlp --dataset single --fast
    python run.py test --scheme pca_unet --run-dir outputs/pca_unet/single/...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根目录，确保可从任意工作目录导入 schemes / scripts
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 方案名 -> 对应 pipeline 模块路径（动态 import，避免未使用方案拖慢启动）
SCHEMES = {
    "mlp": "schemes.mlp.pipeline",
    "pca_unet": "schemes.pca_unet.pipeline",
    "dim_guided": "schemes.dim_guided.pipeline",
    "dim_unet": "schemes.dim_unet.pipeline",
    "edm_guided": "schemes.edm_guided.pipeline",
    "pidm_guided": "schemes.pidm_guided.pipeline",
}


def _import_pipeline(scheme: str):
    """按方案名延迟加载 pipeline 模块。

    Args:
        scheme: 方案标识，须为 ``SCHEMES`` 的键（``mlp`` 或 ``pca_unet``）。

    Returns:
        已导入的 pipeline 模块，须实现 ``run_train`` / ``run_test``。

    Raises:
        ValueError: 未知方案名。
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose from: {list(SCHEMES)}")
    import importlib

    return importlib.import_module(SCHEMES[scheme])


def cmd_prepare_splits(args: argparse.Namespace) -> None:
    """执行 prepare-splits 子命令：写入 data/{dataset}/processed/ 下的划分文件。"""
    from scripts.prepare_splits import prepare_splits

    prepare_splits(args.dataset, force=args.force)


def cmd_prepare_airfoil_uiuc(args: argparse.Namespace) -> None:
    """执行 prepare-airfoil-uiuc 子命令：写入 airfoil_uiuc_sdf processed 数据。"""
    from scripts.prepare_airfoil_uiuc import prepare_airfoil_uiuc

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


def cmd_train(args: argparse.Namespace) -> None:
    """执行 train 子命令：调用对应方案的 ``run_train``。"""
    pipeline = _import_pipeline(args.scheme)
    run_dir = pipeline.run_train(
        dataset=args.dataset,
        stage=args.stage,
        fast=args.fast,
        ae_run_dir=args.ae_run_dir,
    )
    print(f"Training finished: {run_dir}")
    timing_path = Path(run_dir) / "timing.json"
    if timing_path.exists():
        print(f"Timing report: {timing_path}")


def cmd_test(args: argparse.Namespace) -> None:
    """执行 test 子命令：对指定 run 目录做推理/评估/可视化。"""
    pipeline = _import_pipeline(args.scheme)
    pipeline.run_test(args.run_dir)
    timing_path = Path(args.run_dir) / "timing.json"
    if timing_path.exists():
        print(f"Timing report: {timing_path}")


def cmd_collect_timing(args: argparse.Namespace) -> None:
    """汇总 outputs/ 下各 run 的 timing.json，导出汇报用 CSV/JSON。"""
    from scripts.timing_utils import (
        collect_timing_reports,
        export_timing_chart,
        export_timing_csv,
        save_timing_report,
    )

    outputs_root = ROOT / "outputs"
    rows = collect_timing_reports(outputs_root, scheme=args.scheme, dataset=args.dataset)
    if not rows:
        print("No timing.json found under outputs/")
        return
    out_dir = ROOT / "outputs"
    summary_json = out_dir / "timing_summary.json"
    summary_csv = out_dir / "timing_summary.csv"
    summary_chart = out_dir / "timing_comparison.png"
    save_timing_report(summary_json, {"runs": rows, "count": len(rows)})
    export_timing_csv(rows, summary_csv)
    export_timing_chart(rows, summary_chart)
    print(f"Collected {len(rows)} run(s)")
    print(f"  JSON:  {summary_json}")
    print(f"  CSV:   {summary_csv}")
    print(f"  Chart: {summary_chart}")


def main() -> None:
    """解析命令行参数并分发到各子命令处理函数。"""
    parser = argparse.ArgumentParser(description="Meridian SDF diffusion runner")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- 数据集划分 ---
    airfoil = sub.add_parser("prepare-airfoil-uiuc", help="Create UIUC airfoil SDF dataset")
    airfoil.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "airfoil" / "raw" / "uiuc" / "coord_seligFmt",
        help="Directory containing UIUC/Selig .dat files",
    )
    airfoil.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "airfoil_uiuc_sdf" / "processed",
        help="Processed output directory",
    )
    airfoil.add_argument("--height", type=int, default=128)
    airfoil.add_argument("--width", type=int, default=256)
    airfoil.add_argument("--n-resample", type=int, default=257)
    airfoil.add_argument("--sdf-scale", type=float, default=0.08)
    airfoil.add_argument("--force", action="store_true", help="Overwrite existing NPZ samples")
    airfoil.set_defaults(func=cmd_prepare_airfoil_uiuc)

    prep = sub.add_parser("prepare-splits", help="Create fixed train/test CSV under data/{dataset}/processed/")
    prep.add_argument("--dataset", required=True, help="Dataset name, e.g. single or F404")
    prep.add_argument("--force", action="store_true", help="Overwrite existing split files")
    prep.set_defaults(func=cmd_prepare_splits)

    # --- 训练 ---
    train = sub.add_parser("train", help="Train autoencoder + diffusion for a scheme")
    train.add_argument("--scheme", required=True, choices=list(SCHEMES), help="mlp, pca_unet, dim_guided, or dim_unet")
    train.add_argument("--dataset", required=True, help="Dataset name, e.g. single or F404")
    train.add_argument(
        "--stage",
        choices=["all", "ae", "diff", "unet"],
        default="all",
        # unet 为 pca_unet 方案下扩散阶段的别名
        help="Training stage (unet alias for pca_unet diffusion)",
    )
    train.add_argument("--fast", action="store_true", help="Fewer epochs for smoke test")
    train.add_argument("--ae-run-dir", default=None, help="Existing AE run when stage=diff/unet")
    train.set_defaults(func=cmd_train)

    # --- 测试 / 可视化 ---
    test = sub.add_parser("test", help="Evaluate / visualize a completed run")
    test.add_argument("--scheme", required=True, choices=list(SCHEMES))
    test.add_argument("--run-dir", required=True, help="Run directory under outputs/{scheme}/{dataset}/{timestamp}/")
    test.set_defaults(func=cmd_test)

    # --- 计时汇总 ---
    timing = sub.add_parser("collect-timing", help="Aggregate timing.json from all runs under outputs/")
    timing.add_argument("--scheme", default=None, help="Filter by scheme, e.g. mlp")
    timing.add_argument("--dataset", default=None, help="Filter by dataset, e.g. single")
    timing.set_defaults(func=cmd_collect_timing)

    args = parser.parse_args()

    # 阶段别名归一化：mlp 无 UNet，unet 等价于 diff
    if args.command == "train" and args.stage == "unet" and args.scheme in ("mlp", "dim_guided", "edm_guided", "pidm_guided"):
        args.stage = "diff"
    if args.command == "train" and args.stage == "diff" and args.scheme in ("pca_unet", "dim_unet"):
        pass  # diff / unet 均为 UNet 扩散阶段

    args.func(args)


if __name__ == "__main__":
    main()
