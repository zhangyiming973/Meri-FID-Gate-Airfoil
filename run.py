#!/usr/bin/env python3
"""Meridian SDF 扩散项目统一命令行入口。

本模块将数据集准备、训练与测试/可视化三条工作流收敛到单一 CLI（``run.py``），
通过 ``--scheme`` 在 MLP 去噪器与 PCA-UNet 两套方案之间切换。

子命令：
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


def cmd_test(args: argparse.Namespace) -> None:
    """执行 test 子命令：对指定 run 目录做推理/评估/可视化。"""
    pipeline = _import_pipeline(args.scheme)
    pipeline.run_test(args.run_dir)


def main() -> None:
    """解析命令行参数并分发到各子命令处理函数。"""
    parser = argparse.ArgumentParser(description="Meridian SDF diffusion runner")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- 数据集划分 ---
    prep = sub.add_parser("prepare-splits", help="Create fixed train/test CSV under data/{dataset}/processed/")
    prep.add_argument("--dataset", required=True, help="Dataset name, e.g. single or F404")
    prep.add_argument("--force", action="store_true", help="Overwrite existing split files")
    prep.set_defaults(func=cmd_prepare_splits)

    # --- 训练 ---
    train = sub.add_parser("train", help="Train autoencoder + diffusion for a scheme")
    train.add_argument("--scheme", required=True, choices=list(SCHEMES), help="mlp, pca_unet, or dim_guided")
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

    args = parser.parse_args()

    # 阶段别名归一化：mlp 无 UNet，unet 等价于 diff
    if args.command == "train" and args.stage == "unet" and args.scheme == "mlp":
        args.stage = "diff"
    if args.command == "train" and args.stage == "diff" and args.scheme == "pca_unet":
        pass  # pca_unet 的 diff 阶段即 UNet 扩散训练，无需改写

    args.func(args)


if __name__ == "__main__":
    main()
