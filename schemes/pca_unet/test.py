#!/usr/bin/env python3
"""PCA-UNet 方案：对已完成的训练运行进行评估与可视化汇总。

替代旧版 ``visualize_unet.py``：聚合 ``autoencoder/`` 与 ``diffusion/`` 产物，
输出到 ``<run_dir>/visualizations/``。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from schemes.pca_unet.data.dataset import ConditionStats
from schemes.pca_unet.models.diffusion import DiffusionSchedule
from schemes.pca_unet.models.latent_pca import load_latent_pca
from schemes.pca_unet.records.body_latent import load_records
from schemes.pca_unet.train.diffusion_codec import DiffusionLatentCodec
from schemes.pca_unet.train.train_diffusion import build_pca_unet_for_conditions, load_autoencoder_from_checkpoint
from schemes.pca_unet.utils.config import load_config
from schemes.pca_unet.utils.paths import project_root
from schemes.pca_unet.utils.visualization import plot_per_sample_metrics, save_json

SCHEME_LABEL = "PCA-UNet"

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


def _find_ae_run_dir(run_dir: Path, diff_dir: Path) -> Path:
    """Resolve the AE directory used by a diffusion run."""
    summary = _load_json(diff_dir / "summary.json") or {}
    if summary.get("ae_run_dir"):
        ae_dir = Path(summary["ae_run_dir"])
        if not ae_dir.is_absolute():
            ae_dir = project_root() / ae_dir
        return ae_dir
    return run_dir / "autoencoder"


def _load_dataset_config(dataset: str) -> dict[str, Any]:
    """Load enough dataset/scheme config to decode saved diffusion checkpoints."""
    root = project_root()
    meta = _load_json(root / "data" / dataset / "dataset.json")
    if meta is None:
        raise FileNotFoundError(f"Dataset metadata not found: data/{dataset}/dataset.json")
    scheme_cfg = load_config(root / "schemes" / "pca_unet" / "config" / f"{dataset}.json")
    processed = root / "data" / dataset / meta["processed_dir"]
    cfg: dict[str, Any] = {
        "dataset_name": dataset,
        "data_root": str(processed),
        "npz_dir": str(processed / meta["npz_dir"]),
        "condition_columns": meta.get("condition_columns"),
    }
    cfg.update(scheme_cfg)
    return cfg


@torch.no_grad()
def _export_generated_sdf_npz(run_dir: Path, max_existing: bool = True) -> dict[str, Any]:
    """Re-sample the trained diffusion model and persist generated SDF arrays as NPZ files."""
    diff_dir = run_dir / "diffusion"
    ckpt_path = diff_dir / "best_unet_diffusion.pt"
    pca_path = diff_dir / "latent_pca.json"
    if not ckpt_path.exists() or not pca_path.exists():
        return {
            "enabled": False,
            "reason": f"missing diffusion checkpoint or PCA: {ckpt_path}, {pca_path}",
        }

    dataset = run_dir.parent.name
    cfg = _load_dataset_config(dataset)
    ae_dir = _find_ae_run_dir(run_dir, diff_dir)
    if not ae_dir.exists():
        return {"enabled": False, "reason": f"AE run directory not found: {ae_dir}"}

    out_dir = diff_dir / "generated_sdf_npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_existing:
        for old in out_dir.glob("*.npz"):
            old.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    ae_cfg = cfg["autoencoder"]
    unet_cfg = cfg.get("unet", cfg.get("diffusion", {}))
    cond_stats = ConditionStats.from_dict(json.loads((ae_dir / "condition_stats.json").read_text()))
    test_recs = load_records(ae_dir / "body_latent_test.json")
    pca = load_latent_pca(pca_path)
    z_shape = (len(test_recs), *test_recs[0].z_m.shape)
    codec = DiffusionLatentCodec.from_mode("pca_unet", pca, z_shape)

    ae = load_autoencoder_from_checkpoint(ae_dir / "best_autoencoder.pt", ae_cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet, grid_spec = build_pca_unet_for_conditions(
        int(ckpt.get("pca_dim", pca.dim)),
        cond_stats,
        base_ch=unet_cfg.get("unet_base_ch", 64),
    )
    unet.load_state_dict(ckpt["model"])
    unet = unet.to(device).eval()
    schedule = DiffusionSchedule(unet_cfg["timesteps"], device)
    sample_shape = codec.sample_shape(1)

    metrics: list[dict[str, Any]] = []
    root = project_root()
    for rec in test_recs:
        cond = torch.from_numpy(rec.condition_norm).unsqueeze(0).to(device)
        z_gen_enc = schedule.sample(
            unet,
            sample_shape,
            cond,
            cfg_scale=unet_cfg.get("cfg_scale", 1.5),
            steps=unet_cfg.get("sample_steps", 50),
            use_ddim=unet_cfg.get("use_ddim", True),
        )
        z_gen = codec.decode_to_raw(z_gen_enc)
        z_gt = torch.from_numpy(rec.z_m).unsqueeze(0).to(device)
        sdf_gen = ae.decode(z_gen)[0, 0].detach().cpu().numpy().astype(np.float32)
        ae_recon = ae.decode(z_gt)[0, 0].detach().cpu().numpy().astype(np.float32)

        npz_path = Path(rec.npz_path) if rec.npz_path else root / cfg["npz_dir"] / f"{rec.sample_id}.npz"
        if not npz_path.is_absolute():
            npz_path = root / npz_path
        target_sdf = np.load(npz_path)["sdf2d_norm"].astype(np.float32)
        l1 = float(np.mean(np.abs(sdf_gen - target_sdf)))
        l1_recon = float(np.mean(np.abs(ae_recon - target_sdf)))
        cond_dict = {c: float(rec.condition_raw[i]) for i, c in enumerate(cond_stats.columns)}

        out_path = out_dir / f"{rec.sample_id}.npz"
        np.savez_compressed(
            out_path,
            generated_sdf=sdf_gen,
            target_sdf=target_sdf,
            ae_recon_sdf=ae_recon,
            condition_raw=rec.condition_raw.astype(np.float32),
            condition_norm=rec.condition_norm.astype(np.float32),
            condition_columns=np.array(cond_stats.columns),
            condition_json=json.dumps(cond_dict, ensure_ascii=False),
            sample_id=np.array(rec.sample_id),
            source_npz_path=np.array(str(npz_path)),
            l1_vs_original=np.array(l1, dtype=np.float32),
            l1_ae_recon=np.array(l1_recon, dtype=np.float32),
        )
        metrics.append(
            {
                "sample_id": rec.sample_id,
                "npz_file": out_path.name,
                "l1_vs_original": l1,
                "l1_ae_recon": l1_recon,
                "condition": cond_dict,
            }
        )

    manifest = {
        "enabled": True,
        "output_dir": str(out_dir),
        "num_samples": len(metrics),
        "dataset": dataset,
        "ae_run_dir": str(ae_dir),
        "diffusion_checkpoint": str(ckpt_path),
        "pca_path": str(pca_path),
        "device": str(device),
        "cfg_scale": unet_cfg.get("cfg_scale", 1.5),
        "sample_steps": unet_cfg.get("sample_steps", 50),
        "use_ddim": unet_cfg.get("use_ddim", True),
        "grid": {
            "channels": grid_spec.channels,
            "height": grid_spec.height,
            "width": grid_spec.width,
        },
        "files": metrics,
    }
    save_json(manifest, out_dir / "manifest.json")
    return manifest


def run_eval(run_dir: Path) -> None:
    """对指定运行目录执行后处理评估与可视化汇总。"""
    run_dir = Path(run_dir)
    ae_dir = run_dir / "autoencoder"
    diff_dir = run_dir / "diffusion"
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "scheme": "pca_unet",
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
        generated_sdf_export = _export_generated_sdf_npz(run_dir)
        report["diffusion"] = {
            "mean_l1_vs_original": metrics.get("mean_l1_vs_original"),
            "mean_ae_recon_l1": metrics.get("mean_ae_recon_l1"),
            "summary": _load_json(diff_dir / "summary.json"),
            "num_samples": len(gens),
            "generated_sdf_npz": generated_sdf_export,
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
