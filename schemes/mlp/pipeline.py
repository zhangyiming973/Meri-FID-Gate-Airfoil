"""MLP 扩散方案统一入口：训练与测试流水线。

流程概览：
1. 自编码器（AE）将 2D SDF 编码为潜空间 z_m；
2. PCA 将 z_m 压缩为低维向量，供 MLP 去噪器做条件扩散；
3. 采样得到的潜码经 PCA 逆变换与 AE 解码，生成子午线 SDF。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from schemes.mlp.train.train_ae import train_autoencoder
from schemes.mlp.train.train_diffusion import train_diffusion
from schemes.mlp.utils.config import load_config
from schemes.mlp.utils.paths import make_run_dir, project_root, scheme_name


def resolve_config(dataset: str, scheme_cfg: dict[str, Any]) -> dict[str, Any]:
    """将数据集元信息与方案配置合并为完整训练配置。

    从 data/<dataset>/dataset.json 读取处理后数据路径、NPZ 目录、
    条件列名等，再叠加 schemes/mlp/config/<dataset>.json 中的超参。
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
    """加载指定数据集的 MLP 方案配置并解析路径。"""
    cfg_path = project_root() / "schemes" / scheme_name() / "config" / f"{dataset}.json"
    scheme_cfg = load_config(cfg_path)
    return resolve_config(dataset, scheme_cfg)


def setup_logging(run_dir: Path) -> None:
    """配置日志：同时写入 run_dir/logs/train.log 与标准输出。"""
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
    """执行训练流水线。

    Args:
        dataset: 数据集名称（对应 data/<dataset>/）。
        stage: ``"all"`` 先 AE 后扩散；``"ae"`` 仅 AE；``"diff"`` 仅扩散（需 ae_run_dir）。
        fast: 为 True 时缩短 epoch 数，用于快速冒烟测试。
        ae_run_dir: 扩散阶段使用的已训练 AE 目录；stage 为 ``"diff"`` 时必填。

    Returns:
        本次运行的输出根目录（含 autoencoder/ 与 diffusion/ 子目录）。
    """
    cfg = load_scheme_config(dataset)
    if fast:
        cfg["autoencoder"]["epochs"] = 15
        cfg["diffusion"]["epochs"] = 20

    run_dir = make_run_dir(dataset)
    setup_logging(run_dir)
    logging.info("Scheme=%s dataset=%s stage=%s run_dir=%s", scheme_name(), dataset, stage, run_dir)

    ae_dir = Path(ae_run_dir) if ae_run_dir else None
    if stage in ("all", "ae"):
        ae_dir = train_autoencoder(cfg, run_dir / "autoencoder")
    if stage in ("all", "diff"):
        if ae_dir is None:
            raise ValueError("--ae-run-dir required for diffusion-only stage")
        train_diffusion(cfg, ae_dir, run_dir / "diffusion")
    return run_dir


def run_test(run_dir: str) -> None:
    """对已完成训练的运行目录执行评估与可视化。"""
    from schemes.mlp.test import run_eval

    run_eval(Path(run_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLP diffusion scheme")
    parser.add_argument("task", choices=["train", "test"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", choices=["all", "ae", "diff"], default="all")
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
