#!/usr/bin/env python3
"""MLP 方案测试入口：读取扩散生成指标并绘制可视化。

依赖 diffusion/generation_metrics.json（由 train_diffusion 在训练结束时写出）。
"""
from __future__ import annotations

import json
from pathlib import Path

from schemes.mlp.utils.visualization import plot_per_sample_metrics, save_json


def run_eval(run_dir: Path) -> None:
    """对一次完整训练运行做后处理评估。

    读取各测试样本的 L1 误差，绘制柱状图，并将指标副本保存到
    visualizations/ 目录。
    """
    diff_dir = run_dir / "diffusion"
    metrics_path = diff_dir / "generation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    # 逐样本生成 L1 vs 原始 SDF 柱状图
    plot_per_sample_metrics(
        [{"sample_id": m["sample_id"], "l1": m["l1_vs_original"]} for m in metrics["generations"]],
        vis_dir / "test_generation_l1.png",
        "MLP Generation L1 vs Original",
    )
    save_json(metrics, vis_dir / "generation_metrics.json")
    print(f"MLP test visualization saved -> {vis_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    run_eval(Path(parser.parse_args().run_dir))
