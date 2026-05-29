"""Visualize PCA-UNet pipeline results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.dataset import ConditionStats
from src.models.autoencoder import MeridianAutoEncoder
from src.models.diffusion import DiffusionSchedule
from src.models.latent_pca import load_latent_pca
from src.models.pca_unet import build_pca_unet
from src.records.body_latent import load_records
from src.train.diffusion_codec import DiffusionLatentCodec
from src.utils.config import load_config
from src.utils.paths import project_root
from src.utils.visualization import (
    plot_per_sample_metrics,
    plot_sample_grid,
    plot_sdf_panel,
    plot_training_dashboard,
    save_json,
    to_numpy,
)
from src.visualize.results import visualize_ae_run


@torch.no_grad()
def visualize_unet_run(unet_run_dir: Path, ae_run_dir: Path, cfg: dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    vis_root = unet_run_dir / "visualizations"
    vis_root.mkdir(exist_ok=True)

    history = json.loads((unet_run_dir / "training_history.json").read_text())
    plot_training_dashboard(history, vis_root / "training_dashboard.png", "PCA-UNet Training Dashboard")

    ae_cfg = cfg["autoencoder"]
    unet_cfg = cfg.get("unet", cfg.get("diffusion", {}))
    cond_stats = ConditionStats.from_dict(json.loads((ae_run_dir / "condition_stats.json").read_text()))

    ckpt_ae = torch.load(ae_run_dir / "best_autoencoder.pt", map_location=device, weights_only=False)
    ae = MeridianAutoEncoder(ckpt_ae["in_channels"], ae_cfg["latent_channels"], ae_cfg["latent_spatial"]).to(device)
    ae.load_state_dict(ckpt_ae["model"])
    ae.eval()

    ckpt_path = unet_run_dir / "best_unet_diffusion.pt"
    if not ckpt_path.exists():
        ckpt_path = unet_run_dir / "best_diffusion.pt"
    ckpt_unet = torch.load(ckpt_path, map_location=device, weights_only=False)

    pca = load_latent_pca(unet_run_dir / "latent_pca.json")
    z_shape = (1, ae_cfg["latent_channels"], ae_cfg["latent_spatial"], ae_cfg["latent_spatial"])
    codec = DiffusionLatentCodec.from_mode("pca_unet", pca, z_shape)

    unet, _ = build_pca_unet(pca.dim, len(cond_stats.columns), base_ch=unet_cfg.get("unet_base_ch", 64))
    unet.load_state_dict(ckpt_unet["model"])
    unet = unet.to(device).eval()

    schedule = DiffusionSchedule(unet_cfg["timesteps"], device)
    test_records = load_records(ae_run_dir / "body_latent_test.json")
    npz_dir = project_root() / cfg["npz_dir"]
    out_dir = vis_root / "generations_test"
    out_dir.mkdir(exist_ok=True)

    grid_items: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    sample_shape = codec.sample_shape(1)

    for r in test_records:
        cond = torch.from_numpy(r.condition_norm).unsqueeze(0).to(device)
        z_gen_enc = schedule.sample(
            unet,
            sample_shape,
            cond,
            unet_cfg.get("cfg_scale", 1.5),
            unet_cfg.get("sample_steps", 50),
            use_ddim=unet_cfg.get("use_ddim", True),
        )
        z_gen = codec.decode_to_raw(z_gen_enc)
        sdf_orig = np.load(npz_dir / f"{r.sample_id}.npz")["sdf2d_norm"].astype(np.float32)
        sdf_gen = ae.decode(z_gen)[0, 0]
        gt_np, gen_np = to_numpy(torch.from_numpy(sdf_orig)), to_numpy(sdf_gen)
        cond_dict = {c: float(r.condition_raw[i]) for i, c in enumerate(cond_stats.columns)}
        l1 = float(np.abs(gt_np - gen_np).mean())
        plot_sdf_panel(gt_np, gen_np, f"UNet Gen [test] {r.sample_id}", out_dir / f"{r.sample_id}.png", cond_dict)
        grid_items.append({"sample_id": r.sample_id, "gt": gt_np, "pred": gen_np, "l1": l1})
        metrics.append({"sample_id": r.sample_id, "l1": l1, "condition": cond_dict})

    plot_sample_grid(grid_items, vis_root / "grid_test.png", f"PCA-UNet Generations (n={len(grid_items)})", max_samples=10)
    plot_per_sample_metrics(metrics, vis_root / "metrics_test.png", "Per-sample UNet Generation L1")
    save_json({"generations": metrics, "mean_l1": float(np.mean([m["l1"] for m in metrics]))}, vis_root / "generation_metrics.json")
    print(f"UNet visualizations -> {vis_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PCA-UNet training results")
    parser.add_argument("--config", default="configs/train_unet.json")
    parser.add_argument("--ae-run-dir", required=True)
    parser.add_argument("--unet-run-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(project_root() / args.config)
    ae_dir = Path(args.ae_run_dir)
    visualize_ae_run(ae_dir, cfg)
    visualize_unet_run(Path(args.unet_run_dir), ae_dir, cfg)


if __name__ == "__main__":
    main()
