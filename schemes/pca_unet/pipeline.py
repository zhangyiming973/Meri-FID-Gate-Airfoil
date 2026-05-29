"""PCA-UNet 扩散方案训练/测试入口。

本方案与标准 latent diffusion 共用同一自编码器（AE），但去噪器改为在 PCA 降维后的
2D 网格上运行的条件 UNet。训练分两阶段：
  1. AE：学习 SDF 的潜空间表示 z_m
  2. Diffusion：对 PCA 网格上的潜变量做条件扩散建模

命令行用法见 ``__main__`` 块。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from schemes.pca_unet.train.train_ae import train_autoencoder
from schemes.pca_unet.train.train_diffusion import train_diffusion
from schemes.pca_unet.utils.config import load_config
from schemes.pca_unet.utils.paths import make_run_dir, project_root, scheme_name


def resolve_config(dataset: str, scheme_cfg: dict[str, Any]) -> dict[str, Any]:
    """将方案配置与数据集元信息合并为完整训练配置。

    从 ``data/<dataset>/dataset.json`` 读取处理后数据路径、NPZ 目录、
    条件列等元数据，再与 ``schemes/pca_unet/config/<dataset>.json`` 中的
    超参合并。

    Args:
        dataset: 数据集名称（如 ``single``、``F404``）。
        scheme_cfg: 方案级 JSON 配置字典。

    Returns:
        包含数据路径、条件列、AE/UNet 超参的完整配置。
    """
    root = project_root()
    meta_path = root / "data" / dataset / "dataset.json"
    with open(meta_path, encoding="utf-8") as f:
        dmeta = json.load(f)
    processed = root / "data" / dataset / dmeta["processed_dir"]
    merged = {
        "dataset_name": dataset,
        "data_root": str(processed),
        "npz_dir": str(processed / dmeta["npz_dir"]),
        "param_table_path": dmeta.get("param_table_path", ""),
        "condition_columns": dmeta.get("condition_columns"),
    }
    merged.update(scheme_cfg)
    merged["dataset"] = dataset
    return merged


def load_scheme_config(dataset: str) -> dict[str, Any]:
    """加载指定数据集的 PCA-UNet 方案配置。"""
    cfg_path = project_root() / "schemes" / scheme_name() / "config" / f"{dataset}.json"
    scheme_cfg = load_config(cfg_path)
    return resolve_config(dataset, scheme_cfg)


def setup_logging(run_dir: Path) -> None:
    """在运行目录下配置日志：同时输出到文件与标准输出。"""
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def run_train(
    dataset: str,
    stage: str = "all",
    fast: bool = False,
    ae_run_dir: str | None = None,
) -> Path:
    """执行 PCA-UNet 训练流水线。

    Args:
        dataset: 数据集名称。
        stage: 训练阶段——``all``（AE+扩散）、``ae``（仅 AE）、
               ``diff``/``unet``（仅扩散，需提供 AE 检查点）。
        fast: 快速调试模式，缩短 epoch 数。
        ae_run_dir: 已有 AE 运行目录；仅扩散阶段必填。

    Returns:
        本次运行的根目录路径。
    """
    cfg = load_scheme_config(dataset)
    if fast:
        # 快速模式：减少 epoch 以便冒烟测试
        cfg["autoencoder"]["epochs"] = 15
        cfg["unet"]["epochs"] = 20

    run_dir = make_run_dir(dataset)
    setup_logging(run_dir)
    logging.info("Scheme=%s dataset=%s stage=%s run_dir=%s", scheme_name(), dataset, stage, run_dir)

    ae_dir = Path(ae_run_dir) if ae_run_dir else None
    if stage in ("all", "ae"):
        ae_dir = train_autoencoder(cfg, run_dir / "autoencoder")
    if stage in ("all", "diff", "unet"):
        if ae_dir is None:
            raise ValueError("--ae-run-dir required for diffusion-only stage")
        train_diffusion(cfg, ae_dir, run_dir / "diffusion")
    return run_dir


def run_test(run_dir: str) -> None:
    """对已完成训练运行评估与可视化。"""
    from schemes.pca_unet.test import run_eval

    run_eval(Path(run_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCA-UNet diffusion scheme")
    parser.add_argument("task", choices=["train", "test"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", choices=["all", "ae", "diff", "unet"], default="all")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--ae-run-dir", default=None)
    args = parser.parse_args()
    if args.task == "train":
        run_train(args.dataset, args.stage, args.fast, args.ae_run_dir)
    else:
        if not args.run_dir:
            raise ValueError("--run-dir required for test")
        run_test(args.run_dir)
