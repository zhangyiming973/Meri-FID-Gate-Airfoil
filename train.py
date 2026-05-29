#!/usr/bin/env python3
"""Train size-guided latent diffusion pipeline for sdf2d meridian bodies."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.train.train_autoencoder import train_autoencoder
from src.train.train_diffusion import train_diffusion
from src.utils.config import load_config
from src.utils.paths import make_run_dir, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--stage", choices=["all", "ae", "diff"], default="all")
    parser.add_argument("--ae-run-dir", default=None, help="Existing AE run for diffusion-only")
    parser.add_argument("--fast", action="store_true", help="Fewer epochs for smoke test")
    args = parser.parse_args()

    cfg_path = project_root() / args.config
    cfg = load_config(cfg_path)
    if args.fast:
        cfg["autoencoder"]["epochs"] = 15
        cfg["diffusion"]["epochs"] = 20

    pipeline_dir = make_run_dir(cfg["dataset_name"], "pipeline")
    print(f"Pipeline output: {pipeline_dir}")

    ae_dir = Path(args.ae_run_dir) if args.ae_run_dir else None
    if args.stage in ("all", "ae"):
        ae_dir = train_autoencoder(cfg, pipeline_dir / "autoencoder")
    if args.stage in ("all", "diff"):
        if ae_dir is None:
            raise ValueError("--ae-run-dir required for diffusion-only stage")
        train_diffusion(cfg, ae_dir, pipeline_dir / "diffusion")


if __name__ == "__main__":
    main()
