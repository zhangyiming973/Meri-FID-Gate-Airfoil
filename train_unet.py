#!/usr/bin/env python3
"""Train PCA-grid Conditional UNet diffusion pipeline for sdf2d meridian bodies."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.train.train_autoencoder import train_autoencoder
from src.train.train_unet_diffusion import train_unet_diffusion
from src.utils.config import load_config
from src.utils.paths import make_run_dir, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AE + PCA-UNet conditional diffusion")
    parser.add_argument("--config", default="configs/train_unet.json")
    parser.add_argument("--stage", choices=["all", "ae", "unet"], default="all")
    parser.add_argument("--ae-run-dir", default=None, help="Existing AE run for UNet-only stage")
    parser.add_argument("--fast", action="store_true", help="Fewer epochs for smoke test")
    args = parser.parse_args()

    cfg = load_config(project_root() / args.config)
    if args.fast:
        cfg["autoencoder"]["epochs"] = 15
        unet_key = "unet" if "unet" in cfg else "diffusion"
        cfg[unet_key]["epochs"] = 20

    pipeline_dir = make_run_dir(cfg["dataset_name"], "unet_pipeline")
    print(f"UNet pipeline output: {pipeline_dir}")

    ae_dir = Path(args.ae_run_dir) if args.ae_run_dir else None
    if args.stage in ("all", "ae"):
        ae_dir = train_autoencoder(cfg, pipeline_dir / "autoencoder")
    if args.stage in ("all", "unet"):
        if ae_dir is None:
            raise ValueError("--ae-run-dir required for unet-only stage")
        train_unet_diffusion(cfg, ae_dir, pipeline_dir / "diffusion_unet")


if __name__ == "__main__":
    main()
