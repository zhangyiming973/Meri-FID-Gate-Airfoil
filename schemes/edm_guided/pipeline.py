"""edm_guided 方案统一入口：EDM 预条件化潜空间扩散。

流程概览：
1. 自编码器（AE）将 2D SDF 编码为潜空间 z_m；
2. PCA 将 z_m 压缩为低维向量；
3. EDM 预条件化 MLP + 对数正态噪声训练 + Heun 采样（含 CFG）；
4. 采样潜码经 PCA 逆变换与 AE 解码，生成子午线 SDF。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from schemes.edm_guided.train.train_ae import train_autoencoder
from schemes.edm_guided.train.train_diffusion import train_diffusion
from schemes.edm_guided.utils.config import load_config
from schemes.edm_guided.utils.paths import make_run_dir, project_root, scheme_name
from scripts.timing_utils import StageTimer, finalize_test_timing, finalize_train_timing, print_timing_summary


def resolve_config(dataset: str, scheme_cfg: dict[str, Any]) -> dict[str, Any]:
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
    cfg_path = project_root() / "schemes" / scheme_name() / "config" / f"{dataset}.json"
    scheme_cfg = load_config(cfg_path)
    return resolve_config(dataset, scheme_cfg)


def setup_logging(run_dir: Path) -> None:
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
    cfg = load_scheme_config(dataset)
    if fast:
        cfg["autoencoder"]["epochs"] = 15
        cfg["diffusion"]["epochs"] = 20

    run_dir = make_run_dir(dataset)
    setup_logging(run_dir)
    logging.info("Scheme=%s dataset=%s stage=%s run_dir=%s", scheme_name(), dataset, stage, run_dir)
    logging.info(
        "EDM config: sigma_data=%s p_mean=%s p_std=%s rho=%s sample_steps=%s",
        cfg.get("edm", {}).get("sigma_data", "auto"),
        cfg.get("edm", {}).get("p_mean", -1.2),
        cfg.get("edm", {}).get("p_std", 1.2),
        cfg.get("edm", {}).get("rho", 7.0),
        cfg.get("diffusion", {}).get("sample_steps", 35),
    )

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
    print_timing_summary(timing_report, title="edm_guided Training Timing")
    logging.info(
        "Training timing: total=%s | ae=%s | diffusion=%s",
        timing_report["train"]["total_human"],
        timing_report["train"]["stages"].get("autoencoder", {}).get("human", "N/A"),
        timing_report["train"]["stages"].get("diffusion", {}).get("human", "N/A"),
    )
    return run_dir


def run_test(run_dir: str) -> None:
    from schemes.edm_guided.test import run_eval
    from schemes.edm_guided.utils.visualization import save_json

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
    print_timing_summary(timing_report, title="edm_guided Test Timing")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="edm_guided EDM diffusion scheme")
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
