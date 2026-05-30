#!/usr/bin/env python3
"""dim_unet 方案测试入口：ConditionVector + UNet 扩散评估与可视化汇总。"""

替代旧版 ``visualize.py``：从 ``autoencoder/`` 与 ``diffusion/`` 子目录读取
训练阶段已保存的指标与图像，汇总到 ``<run_dir>/visualizations/``。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from schemes.dim_unet.utils.visualization import plot_per_sample_metrics, save_json

SCHEME_LABEL = "DimUNet"

_AE_ARTIFACTS = [
    ("training_dashboard.png", "ae_training_dashboard.png"),
    ("training_curves.png", "ae_training_curves.png"),
    ("latent_gate_report.png", "ae_latent_gate_report.png"),
    ("recon_l1_train_vs_test.png", "ae_recon_l1_train_vs_test.png"),
    ("latent_pca_test.png", "ae_latent_pca_test.png"),
    ("visualizations/grid_test.png", "ae_grid_test.png"),
    ("visualizations/metrics_test.png", "ae_metrics_test.png"),
]

_DIFF_ARTIFACTS = [
    ("training_dashboard.png", "diff_training_dashboard.png"),
    ("training_curves.png", "diff_training_curves.png"),
    ("grid_test.png", "diff_grid_test.png"),
    ("metrics_test.png", "diff_metrics_test.png"),
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _copy_artifacts(src_dir: Path, dst_dir: Path, mapping: list[tuple[str, str]]) -> list[str]:
    copied: list[str] = []
    for src_rel, dst_name in mapping:
        src = src_dir / src_rel
        if src.exists():
            shutil.copy2(src, dst_dir / dst_name)
            copied.append(dst_name)
    return copied


def _plot_gen_vs_ae(metrics: list[dict[str, Any]], save_path: Path, title: str) -> None:
    """并排柱状图：扩散生成 vs AE 重建，相对原始 SDF 的 L1。"""
    if not metrics:
        return
    ids = [m["sample_id"] for m in metrics]
    gen = [m["l1_vs_original"] for m in metrics]
    ae = [m["l1_ae_recon"] for m in metrics]
    x = np.arange(len(ids))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.7), 4))
    ax.bar(x - w / 2, gen, w, label="Generation", color="#e67e22")
    ax.bar(x + w / 2, ae, w, label="AE recon", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylabel("L1 vs original SDF")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_eval(run_dir: Path) -> None:
    """对一次完整训练运行做后处理评估与可视化汇总。"""
    run_dir = Path(run_dir)
    ae_dir = run_dir / "autoencoder"
    diff_dir = run_dir / "diffusion"
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "scheme": "dim_unet",
        "artifacts": {},
    }

    if ae_dir.is_dir():
        ae_test_metrics = _load_json(ae_dir / "visualizations" / "metrics_test.json")
        report["autoencoder"] = {
            "summary": _load_json(ae_dir / "summary.json"),
            "gate": _load_json(ae_dir / "latent_gate_report.json"),
            "test_recon_mean_l1": ae_test_metrics.get("mean_l1") if ae_test_metrics else None,
        }
        report["artifacts"]["autoencoder"] = _copy_artifacts(ae_dir, vis_dir, _AE_ARTIFACTS)

    metrics_path = diff_dir / "generation_metrics.json"
    if metrics_path.exists():
        metrics = _load_json(metrics_path) or {}
        gens = metrics.get("generations", [])
        report["diffusion"] = {
            "mean_l1_vs_original": metrics.get("mean_l1_vs_original"),
            "mean_ae_recon_l1": metrics.get("mean_ae_recon_l1"),
            "summary": _load_json(diff_dir / "summary.json"),
            "num_samples": len(gens),
        }
        report["artifacts"]["diffusion"] = _copy_artifacts(diff_dir, vis_dir, _DIFF_ARTIFACTS)

        plot_per_sample_metrics(
            [{"sample_id": m["sample_id"], "l1": m["l1_vs_original"]} for m in gens],
            vis_dir / "test_generation_l1.png",
            f"{SCHEME_LABEL} Generation L1 vs Original",
        )
        plot_per_sample_metrics(
            [{"sample_id": m["sample_id"], "l1": m["l1_ae_recon"]} for m in gens],
            vis_dir / "test_ae_recon_l1.png",
            f"{SCHEME_LABEL} AE Recon L1 vs Original",
        )
        _plot_gen_vs_ae(gens, vis_dir / "test_gen_vs_ae_recon.png", f"{SCHEME_LABEL}: Generation vs AE Recon")
        save_json(metrics, vis_dir / "generation_metrics.json")

        gen_src = diff_dir / "generations"
        if gen_src.is_dir():
            gen_dst = vis_dir / "generations"
            gen_dst.mkdir(exist_ok=True)
            copied_gens = []
            for png in sorted(gen_src.glob("*.png")):
                shutil.copy2(png, gen_dst / png.name)
                copied_gens.append(png.name)
            report["artifacts"]["generations"] = copied_gens
    elif not ae_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory has neither autoencoder/ nor diffusion/generation_metrics.json: {run_dir}"
        )

    from scripts.timing_utils import load_timing_report

    timing = load_timing_report(run_dir / "timing.json")
    if timing:
        report["timing"] = timing
        save_json(timing, vis_dir / "timing.json")

    save_json(report, vis_dir / "summary_report.json")
    print(f"{SCHEME_LABEL} test report -> {vis_dir / 'summary_report.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    run_eval(Path(parser.parse_args().run_dir))
