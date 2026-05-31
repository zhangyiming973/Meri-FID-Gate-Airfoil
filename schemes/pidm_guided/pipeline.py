"""PIDM 风格物理信息扩散方案统一入口：训练与测试流水线。

基于 MLP 潜空间扩散，在训练损失中引入 PIDM (ICLR 2025) 虚拟似然：
  L = c_data * L_DDPM + c_residual * (-log p(r=0 | x0_pred, var_t))

几何残差（壁厚、Eikonal）在 AE 解码 SDF 上计算，目标趋近 0。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from schemes.pidm_guided.train.train_ae import train_autoencoder
from schemes.pidm_guided.train.train_diffusion import train_diffusion
from schemes.pidm_guided.utils.config import load_config
from schemes.pidm_guided.utils.paths import make_run_dir, project_root, scheme_name
from scripts.timing_utils import StageTimer, finalize_test_timing, finalize_train_timing, print_timing_summary


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

    timer = StageTimer()
    timer.start()
    ae_dir = Path(ae_run_dir) if ae_run_dir else None
    if stage in ("all", "ae"):
        with timer.stage("autoencoder"):
            ae_dir = train_autoencoder(cfg, run_dir / "autoencoder")
    if stage in ("all", "diff"):
        if ae_dir is None:
            raise ValueError("--ae-run-dir required for diffusion-only stage")
        with timer.stage("diffusion"):
            train_diffusion(cfg, ae_dir, run_dir / "diffusion")

    timing_report = finalize_train_timing(run_dir, scheme_name(), dataset, stage, fast, timer)
    print_timing_summary(timing_report, title="PIDM-Guided Training Timing")
    logging.info(
        "Training timing: total=%s | ae=%s | diffusion=%s",
        timing_report["train"]["total_human"],
        timing_report["train"]["stages"].get("autoencoder", {}).get("human", "N/A"),
        timing_report["train"]["stages"].get("diffusion", {}).get("human", "N/A"),
    )
    return run_dir


def run_test(run_dir: str) -> None:
    """对已完成训练的运行目录执行评估与可视化。"""
    from schemes.pidm_guided.test import run_eval
    from schemes.pidm_guided.utils.visualization import save_json

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
    print_timing_summary(timing_report, title="PIDM-Guided Test Timing")


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
