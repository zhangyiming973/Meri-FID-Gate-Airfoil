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
from scripts.timing_utils import StageTimer, finalize_test_timing, finalize_train_timing, print_timing_summary


def _as_hw(value: int | list[int] | tuple[int, ...], name: str) -> tuple[int, int]:
    """将整数或二维列表配置解析为 (H, W)。"""
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be an int or a two-item list, got {value!r}")


def validate_autoencoder_geometry(ae_cfg: dict[str, Any]) -> None:
    """校验 AE 输入尺寸与 latent 空间尺寸匹配当前 4 次下采样结构。"""
    input_size = _as_hw(ae_cfg.get("input_size", [256, 256]), "input_size")
    latent_spatial = _as_hw(ae_cfg.get("latent_spatial", 16), "latent_spatial")
    if input_size[0] % latent_spatial[0] != 0 or input_size[1] % latent_spatial[1] != 0:
        raise ValueError(f"input_size {input_size} must be divisible by latent_spatial {latent_spatial}")
    ratio = (input_size[0] // latent_spatial[0], input_size[1] // latent_spatial[1])
    if ratio != (16, 16):
        raise ValueError(
            f"input_size / latent_spatial must be 16 in each dimension for the current AE, got {ratio}"
        )


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
    if not merged.get("condition_columns"):
        raise ValueError(f"Dataset {dataset} must define non-empty condition_columns")
    validate_autoencoder_geometry(merged["autoencoder"])
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

    timer = StageTimer()
    timer.start()
    ae_dir = Path(ae_run_dir) if ae_run_dir else None
    if stage in ("all", "ae"):
        with timer.stage("autoencoder"):
            ae_dir = train_autoencoder(cfg, run_dir / "autoencoder")
    if stage in ("all", "diff", "unet"):
        if ae_dir is None:
            raise ValueError("--ae-run-dir required for diffusion-only stage")
        with timer.stage("diffusion"):
            train_diffusion(cfg, ae_dir, run_dir / "diffusion")

    timing_report = finalize_train_timing(run_dir, scheme_name(), dataset, stage, fast, timer)
    print_timing_summary(timing_report, title="PCA-UNet Training Timing")
    logging.info(
        "Training timing: total=%s | ae=%s | diffusion=%s",
        timing_report["train"]["total_human"],
        timing_report["train"]["stages"].get("autoencoder", {}).get("human", "N/A"),
        timing_report["train"]["stages"].get("diffusion", {}).get("human", "N/A"),
    )
    return run_dir


def run_test(run_dir: str) -> None:
    """对已完成训练运行评估与可视化。"""
    from schemes.pca_unet.test import run_eval
    from schemes.pca_unet.utils.visualization import save_json

    run_path = Path(run_dir)
    timer = StageTimer()
    timer.start()
    with timer.stage("test_eval"):
        run_eval(run_path)
    timing_report = finalize_test_timing(run_path, scheme_name(), timer)
    vis_report = run_path / "visualizations" / "summary_report.json"
    if vis_report.exists():
        import json

        with open(vis_report, encoding="utf-8") as f:
            report = json.load(f)
        report["timing"] = timing_report
        save_json(report, vis_report)
        save_json(timing_report, run_path / "visualizations" / "timing.json")
    print_timing_summary(timing_report, title="PCA-UNet Test Timing")


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
