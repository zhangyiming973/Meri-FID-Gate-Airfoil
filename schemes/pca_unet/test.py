#!/usr/bin/env python3
"""PCA-UNet 方案：对已完成的训练运行进行评估与可视化。

读取 ``diffusion/generation_metrics.json``，生成 L1 柱状图等测试图表，
输出到 ``<run_dir>/visualizations/``。
"""
from __future__ import annotations

import json
from pathlib import Path

from schemes.pca_unet.utils.visualization import plot_per_sample_metrics, save_json


def run_eval(run_dir: Path) -> None:
    """对指定运行目录执行后处理评估。

    Args:
        run_dir: 训练输出根目录，需包含 ``diffusion/generation_metrics.json``。
    """
    diff_dir = run_dir / "diffusion"
    metrics_path = diff_dir / "generation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    # 绘制各样本生成 SDF 与原始 SDF 的 L1 误差
    plot_per_sample_metrics(
        [{"sample_id": m["sample_id"], "l1": m["l1_vs_original"]} for m in metrics["generations"]],
        vis_dir / "test_generation_l1.png",
        "PCA-UNet Generation L1 vs Original",
    )
    save_json(metrics, vis_dir / "generation_metrics.json")
    print(f"PCA-UNet test visualization saved -> {vis_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    run_eval(Path(parser.parse_args().run_dir))
