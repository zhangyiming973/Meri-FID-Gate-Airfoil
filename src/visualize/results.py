"""从已保存的训练 run 重新生成综合可视化。

支持自编码器 run 与 MLP/UNet 扩散 run 的离线可视化，
无需重新训练即可产出训练曲线、重建/生成对比图等。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ConditionStats, MeridianSDFDataset
from src.models.autoencoder import MeridianAutoEncoder
from src.models.diffusion import DiffusionSchedule, LatentDiffusionUNet, load_latent_stats
from src.models.latent_pca import load_latent_pca
from src.models.mlp_denoiser import MLPDenoiser
from src.models.pca_unet import build_pca_unet
from src.train.diffusion_codec import DiffusionLatentCodec
from src.records.body_latent import load_records
from src.utils.config import load_config
from src.utils.paths import project_root
from src.utils.visualization import (
    plot_gate_report,
    plot_latent_pca,
    plot_per_sample_metrics,
    plot_sample_grid,
    plot_sdf_panel,
    plot_train_test_metric_compare,
    plot_training_dashboard,
    save_json,
    to_numpy,
)


@torch.no_grad()
def visualize_ae_run(ae_run_dir: Path, cfg: dict[str, Any]) -> None:
    """为自编码器训练 run 生成完整可视化套件。

    包括：训练仪表盘、门控报告、潜变量 PCA、train/test 重建对比。
    """
    root = project_root()
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    vis_root = ae_run_dir / "visualizations"
    vis_root.mkdir(exist_ok=True)

    history = json.loads((ae_run_dir / "training_history.json").read_text())
    plot_training_dashboard(history, vis_root / "training_dashboard.png", "AutoEncoder Training Dashboard")

    gate_report = json.loads((ae_run_dir / "latent_gate_report.json").read_text())
    plot_gate_report(gate_report, vis_root / "gate_report.png")

    train_records = load_records(ae_run_dir / "body_latent_train.json")
    test_records = load_records(ae_run_dir / "body_latent_test.json")
    plot_latent_pca(
        np.stack([r.z_m.reshape(-1) for r in train_records]),
        vis_root / "latent_pca_train.png",
        [r.sample_id for r in train_records],
    )
    plot_latent_pca(
        np.stack([r.z_m.reshape(-1) for r in test_records]),
        vis_root / "latent_pca_test.png",
        [r.sample_id for r in test_records],
    )
    plot_train_test_metric_compare(
        [r.recon_l1 for r in train_records],
        [r.recon_l1 for r in test_records],
        vis_root / "recon_l1_train_vs_test.png",
    )

    cond_stats = ConditionStats.from_dict(json.loads((ae_run_dir / "condition_stats.json").read_text()))
    param_path = root / cfg.get("param_table_path", "")
    param_table = pd.read_csv(param_path) if param_path.exists() else None
    npz_dir = root / cfg["npz_dir"]
    ae_cfg = cfg["autoencoder"]

    ckpt = torch.load(ae_run_dir / "best_autoencoder.pt", map_location=device, weights_only=False)
    model = MeridianAutoEncoder(ckpt["in_channels"], ae_cfg["latent_channels"], ae_cfg["latent_spatial"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for split_name, split_csv in [("train", "train_split.csv"), ("test", "test_split.csv")]:
        df = pd.read_csv(ae_run_dir / "splits" / split_csv)
        ds = MeridianSDFDataset(df, npz_dir, param_table, cond_stats, ae_cfg["use_semantic"])
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        out_dir = vis_root / f"reconstructions_{split_name}"
        out_dir.mkdir(exist_ok=True)
        grid_items: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []

        for batch in loader:
            x, sdf = batch["x"].to(device), batch["sdf"].to(device)
            _, recon = model(x)
            for i in range(x.shape[0]):
                sid = batch["sample_id"][i]
                gt_np = to_numpy(sdf[i, 0])
                pred_np = to_numpy(recon[i, 0])
                cond = {c: float(batch["condition_raw"][i][j]) for j, c in enumerate(cond_stats.columns)}
                l1 = float(np.abs(gt_np - pred_np).mean())
                plot_sdf_panel(gt_np, pred_np, f"AE Recon [{split_name}] {sid}", out_dir / f"{sid}.png", cond)
                grid_items.append({"sample_id": sid, "gt": gt_np, "pred": pred_np, "l1": l1})
                metrics.append({"sample_id": sid, "l1": l1, "split": split_name})

        plot_sample_grid(
            grid_items,
            vis_root / f"grid_{split_name}.png",
            f"AutoEncoder Reconstructions ({split_name}, n={len(grid_items)})",
            max_samples=10,
        )
        plot_per_sample_metrics(metrics, vis_root / f"metrics_{split_name}.png",
                                f"Per-sample Recon L1 ({split_name})")
        save_json({"metrics": metrics, "mean_l1": float(np.mean([m["l1"] for m in metrics]))},
                  vis_root / f"metrics_{split_name}.json")

    print(f"AE visualizations -> {vis_root}")


@torch.no_grad()
def visualize_diff_run(diff_run_dir: Path, ae_run_dir: Path, cfg: dict[str, Any]) -> None:
    """为扩散训练 run 生成测试集生成可视化。

    自动识别 denoiser 模式（mlp / pca_unet / 遗留 raw UNet）并加载对应模型。
    """
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    vis_root = diff_run_dir / "visualizations"
    vis_root.mkdir(exist_ok=True)

    history = json.loads((diff_run_dir / "training_history.json").read_text())
    plot_training_dashboard(history, vis_root / "training_dashboard.png", "Diffusion Training Dashboard")

    ae_cfg, diff_cfg = cfg["autoencoder"], cfg["diffusion"]
    cond_stats = ConditionStats.from_dict(json.loads((ae_run_dir / "condition_stats.json").read_text()))

    ckpt_ae = torch.load(ae_run_dir / "best_autoencoder.pt", map_location=device, weights_only=False)
    ae = MeridianAutoEncoder(ckpt_ae["in_channels"], ae_cfg["latent_channels"], ae_cfg["latent_spatial"]).to(device)
    ae.load_state_dict(ckpt_ae["model"])
    ae.eval()

    ckpt_diff = torch.load(diff_run_dir / "best_diffusion.pt", map_location=device, weights_only=False)
    mode = ckpt_diff.get("mode", diff_cfg.get("denoiser", "unet"))
    pca_path = diff_run_dir / "latent_pca.json"
    pca = load_latent_pca(pca_path) if pca_path.exists() else None
    codec = None
    z_shape = (1, ae_cfg["latent_channels"], ae_cfg["latent_spatial"], ae_cfg["latent_spatial"])

    # 按 checkpoint 中记录的 mode 构建去噪器与编解码器
    if mode == "pca_unet" and pca is not None:
        codec = DiffusionLatentCodec.from_mode("pca_unet", pca, z_shape)
        denoiser, _ = build_pca_unet(pca.dim, len(cond_stats.columns), base_ch=diff_cfg.get("unet_base_ch", 64))
        denoiser.load_state_dict(ckpt_diff["model"])
        sample_shape = codec.sample_shape(1)
    elif mode == "mlp" and pca is not None:
        denoiser = MLPDenoiser(pca.dim, len(cond_stats.columns), hidden=diff_cfg.get("mlp_hidden", 512)).to(device)
        denoiser.load_state_dict(ckpt_diff["model"])
        codec = DiffusionLatentCodec.from_mode("mlp", pca, z_shape)
        sample_shape = codec.sample_shape(1)
    else:
        # 遗留路径：直接在 AE 潜特征图上扩散
        denoiser = LatentDiffusionUNet(ae_cfg["latent_channels"], len(cond_stats.columns)).to(device)
        denoiser.load_state_dict(ckpt_diff["model"])
        sample_shape = (1, ae_cfg["latent_channels"], ae_cfg["latent_spatial"], ae_cfg["latent_spatial"])

    denoiser = denoiser.to(device)
    denoiser.eval()
    schedule = DiffusionSchedule(diff_cfg["timesteps"], device)
    latent_stats_path = diff_run_dir / "latent_stats.json"
    latent_stats = load_latent_stats(latent_stats_path) if latent_stats_path.exists() else None

    test_records = load_records(ae_run_dir / "body_latent_test.json")
    npz_dir = project_root() / cfg["npz_dir"]
    out_dir = vis_root / "generations_test"
    out_dir.mkdir(exist_ok=True)
    grid_items: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    shape = sample_shape

    for r in test_records:
        cond = torch.from_numpy(r.condition_norm).unsqueeze(0).to(device)
        z_gen_enc = schedule.sample(
            denoiser, shape, cond,
            diff_cfg.get("cfg_scale", 1.5),
            diff_cfg.get("sample_steps", 50),
            use_ddim=diff_cfg.get("use_ddim", True),
        )
        if codec is not None:
            z_gen = codec.decode_to_raw(z_gen_enc)
        elif latent_stats is not None:
            z_gen = latent_stats.denormalize(z_gen_enc)
        else:
            z_gen = z_gen_enc
        npz_path = npz_dir / f"{r.sample_id}.npz"
        sdf_orig = np.load(npz_path)["sdf2d_norm"].astype(np.float32)
        sdf_gen = ae.decode(z_gen)[0, 0]
        gt_np, gen_np = to_numpy(torch.from_numpy(sdf_orig)), to_numpy(sdf_gen)
        cond_dict = {c: float(r.condition_raw[i]) for i, c in enumerate(cond_stats.columns)}
        l1 = float(np.abs(gt_np - gen_np).mean())
        plot_sdf_panel(gt_np, gen_np, f"Diffusion Gen [test] {r.sample_id}", out_dir / f"{r.sample_id}.png", cond_dict)
        grid_items.append({"sample_id": r.sample_id, "gt": gt_np, "pred": gen_np, "l1": l1})
        metrics.append({"sample_id": r.sample_id, "l1": l1, "condition": cond_dict})

    plot_sample_grid(grid_items, vis_root / "grid_test.png",
                     f"Diffusion Generations (test, n={len(grid_items)})", max_samples=10)
    plot_per_sample_metrics(metrics, vis_root / "metrics_test.png", "Per-sample Generation L1 (test)")
    save_json({"generations": metrics, "mean_l1": float(np.mean([m["l1"] for m in metrics]))},
              vis_root / "generation_metrics.json")

    print(f"Diffusion visualizations -> {vis_root}")


def main() -> None:
    """CLI 入口：可视化 AE run，可选同时可视化扩散 run。"""
    parser = argparse.ArgumentParser(description="Visualize training & test results")
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--ae-run-dir", required=True)
    parser.add_argument("--diff-run-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(project_root() / args.config)
    ae_dir = Path(args.ae_run_dir)
    visualize_ae_run(ae_dir, cfg)
    if args.diff_run_dir:
        visualize_diff_run(Path(args.diff_run_dir), ae_dir, cfg)


if __name__ == "__main__":
    main()
