"""dim_guided 方案统一入口：尺寸 ConditionVector 校验 + 条件潜空间扩散。

流程：
1. 校验 5 维尺寸参数在数据集中的适用性（Excel/NPZ/CSV 来源）
2. AE 编码 SDF → z_m
3. 以 ConditionVector 为条件，在 PCA 潜空间训练 MLP 扩散去噪器
4. 采样解码为子午线 SDF
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from schemes.dim_guided.data.condition_vector import evaluate_applicability
from schemes.dim_guided.data.splits import load_splits
from schemes.dim_guided.train.train_ae import train_autoencoder
from schemes.dim_guided.train.train_diffusion import train_diffusion
from schemes.dim_guided.utils.config import load_config
from schemes.dim_guided.utils.paths import make_run_dir, project_root, scheme_name
from schemes.dim_guided.utils.visualization import save_json
from scripts.timing_utils import StageTimer, finalize_test_timing, finalize_train_timing, print_timing_summary


def resolve_config(dataset: str, scheme_cfg: dict[str, Any]) -> dict[str, Any]:
    root = project_root()
    meta_path = root / "data" / dataset / "dataset.json"
    with open(meta_path, encoding="utf-8") as f:
        dmeta = json.load(f)
    processed = root / "data" / dataset / dmeta["processed_dir"]
    param_rel = dmeta.get("param_table_path", "")
    param_path = root / param_rel if param_rel else ""
    merged = {
        "dataset_name": dataset,
        "data_root": str(processed),
        "npz_dir": str(processed / dmeta["npz_dir"]),
        "param_table_path": str(param_path) if param_path else "",
        "condition_columns": dmeta.get("condition_columns"),
        "condition_excel_path": dmeta.get("condition_excel_path", ""),
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


def validate_condition_applicability(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """在训练前评估 ConditionVector 五维参数是否适用于条件扩散。"""
    processed_dir = Path(cfg["data_root"])
    train_df, test_df = load_splits(processed_dir)
    columns = cfg.get("condition_columns")
    train_report = evaluate_applicability(train_df, columns)
    test_report = evaluate_applicability(test_df, columns)
    report = {
        "train": train_report.to_dict(),
        "test": test_report.to_dict(),
        "overall_applicable": train_report.applicable,
    }
    save_json(report, run_dir / "condition_applicability.json")
    logging.info("Condition applicability: train=%s test=%s", train_report.applicable, test_report.applicable)
    for note in train_report.notes:
        logging.warning("  [train] %s", note)
    if not train_report.applicable:
        logging.warning(
            "训练集条件向量适用性不足（见 condition_applicability.json），仍将继续训练；"
            "建议检查低方差列或补全 Excel/NPZ 条件来源。"
        )
    return report


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

    timer = StageTimer()
    timer.start()
    with timer.stage("condition_validation"):
        validate_condition_applicability(cfg, run_dir)

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
    print_timing_summary(timing_report, title="dim_guided Training Timing")
    logging.info(
        "Training timing: total=%s | ae=%s | diffusion=%s",
        timing_report["train"]["total_human"],
        timing_report["train"]["stages"].get("autoencoder", {}).get("human", "N/A"),
        timing_report["train"]["stages"].get("diffusion", {}).get("human", "N/A"),
    )
    return run_dir


def run_test(run_dir: str) -> None:
    from schemes.dim_guided.test import run_eval

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
    print_timing_summary(timing_report, title="dim_guided Test Timing")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="dim_guided diffusion scheme")
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
