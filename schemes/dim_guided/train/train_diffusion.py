"""条件扩散训练：以 ConditionVector 为条件，在 PCA 潜空间训练 MLP 去噪器。

依赖已训练 AE 导出的 body_latent_*.json；训练完成后对测试集条件采样并解码为 SDF，
与原始 NPZ 对比 L1 指标。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from schemes.dim_guided.data.dataset import ConditionStats
from schemes.dim_guided.losses.ae_losses import physics_risk_proxy
from schemes.dim_guided.models.autoencoder import MeridianAutoEncoder
from schemes.dim_guided.models.diffusion import DiffusionSchedule
from schemes.dim_guided.models.latent_pca import LatentPCA, save_latent_pca
from schemes.dim_guided.models.mlp_denoiser import MLPDenoiser
from schemes.dim_guided.records.body_latent import load_records
from schemes.dim_guided.train.diffusion_codec import DiffusionLatentCodec
from schemes.dim_guided.utils.paths import project_root
from schemes.dim_guided.utils.visualization import (
    plot_per_sample_metrics,
    plot_sample_grid,
    plot_sdf_panel,
    plot_training_curves,
    plot_training_dashboard,
    save_json,
    to_numpy,
)


def _build_denoiser(diff_cfg: dict, codec: DiffusionLatentCodec, cond_dim: int, device: torch.device):
    """根据配置构建去噪网络；MLP 方案默认 denoiser=mlp。"""
    mode = diff_cfg.get("denoiser", "mlp")
    if mode == "unet":
        raise ValueError("Raw UNet denoiser is not supported in dim_guided scheme; use pca_unet scheme instead")
    return MLPDenoiser(codec.pca.dim, cond_dim, hidden=diff_cfg.get("mlp_hidden", 512)).to(device), "mlp"


@torch.no_grad()
def eval_generation_l1(
    denoiser,
    ae,
    schedule,
    codec: DiffusionLatentCodec,
    test_recs,
    device,
    diff_cfg,
    max_samples: int = 5,
) -> float:
    """快速生成评估：条件采样 z -> 解码 SDF，与 AE 重建真值比 L1（子集）。"""
    denoiser.eval()
    ae.eval()
    losses = []
    sample_shape = codec.sample_shape(1)
    for r in test_recs[:max_samples]:
        cond = torch.from_numpy(r.condition_norm).unsqueeze(0).to(device)
        z_gen_enc = schedule.sample(
            denoiser,
            sample_shape,
            cond,
            cfg_scale=diff_cfg.get("cfg_scale", 1.5),
            steps=diff_cfg.get("sample_steps", 50),
            use_ddim=diff_cfg.get("use_ddim", True),
        )
        z_gen = codec.decode_to_raw(z_gen_enc)
        z_gt = torch.from_numpy(r.z_m).unsqueeze(0).to(device)
        sdf_gen = ae.decode(z_gen)[0, 0]
        sdf_gt = ae.decode(z_gt)[0, 0]
        losses.append(float(torch.nn.functional.l1_loss(sdf_gen, sdf_gt)))
    return float(np.mean(losses)) if losses else float("inf")


def train_diffusion(cfg: dict[str, Any], ae_run_dir: Path, run_dir: Path) -> Path:
    """在 AE 潜空间 PCA 系数上训练条件扩散模型。

    Args:
        cfg: 合并后的方案配置。
        ae_run_dir: AE 训练输出目录（含 best_autoencoder.pt、body_latent_*.json）。
        run_dir: 本次扩散训练输出目录。

    Returns:
        run_dir，内含 best_diffusion.pt、latent_pca.json、generation_metrics.json 等。
    """
    root = project_root()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")

    ae_cfg, diff_cfg = cfg["autoencoder"], cfg["diffusion"]
    cond_stats = ConditionStats.from_dict(json.loads((ae_run_dir / "condition_stats.json").read_text()))
    train_recs = load_records(ae_run_dir / "body_latent_train.json")
    test_recs = load_records(ae_run_dir / "body_latent_test.json")

    # 在训练集 z_m 上拟合 PCA，降低 MLP 输入维度
    z_train_np = np.stack([r.z_m for r in train_recs])
    z_shape = z_train_np.shape
    pca_dim = diff_cfg.get("pca_dim", 128)
    pca = LatentPCA.fit(z_train_np, pca_dim)
    save_latent_pca(pca, run_dir / "latent_pca.json")
    print(f"PCA: {z_shape[1:]} -> {pca.dim} dims, explained std={pca.std:.4f}")

    codec = DiffusionLatentCodec.from_mode("mlp", pca, z_shape)
    z_enc = codec.encode_numpy(z_train_np)
    c_train = torch.from_numpy(np.stack([r.condition_norm for r in train_recs])).float()
    train_loader = DataLoader(TensorDataset(z_enc, c_train), batch_size=diff_cfg["batch_size"], shuffle=True)

    # 冻结 AE，仅用于解码与物理引导
    ckpt = torch.load(ae_run_dir / "best_autoencoder.pt", map_location=device, weights_only=False)
    ae = MeridianAutoEncoder(ckpt["in_channels"], ae_cfg["latent_channels"], ae_cfg["latent_spatial"]).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()

    denoiser, mode = _build_denoiser(diff_cfg, codec, len(cond_stats.columns), device)
    schedule = DiffusionSchedule(diff_cfg["timesteps"], device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=diff_cfg["lr"], weight_decay=1e-4)
    pg_cfg = cfg.get("physics_guidance", {})
    pg_on, pg_w = pg_cfg.get("enabled", False), pg_cfg.get("weight", 0.02)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_gen_l1": []}
    best_gen = float("inf")
    sample_shape = codec.sample_shape(1)

    def eval_noise_loss() -> float:
        """验证集上的噪声预测 MSE。"""
        denoiser.eval()
        losses = []
        with torch.no_grad():
            for r in test_recs:
                z0 = codec.encode_raw(torch.from_numpy(r.z_m).unsqueeze(0).to(device))
                cond = torch.from_numpy(r.condition_norm).unsqueeze(0).to(device)
                t = torch.randint(0, schedule.timesteps, (1,), device=device)
                xt, noise = schedule.q_sample(z0, t)
                pred = denoiser(xt, t, cond)
                losses.append(float(torch.nn.functional.mse_loss(pred, noise)))
        return float(np.mean(losses))

    for epoch in range(1, diff_cfg["epochs"] + 1):
        denoiser.train()
        ep_loss, nb = 0.0, 0
        for z0, cond in tqdm(train_loader, desc=f"Diff {epoch}/{diff_cfg['epochs']}", leave=False):
            z0, cond = z0.to(device), cond.to(device)
            t = torch.randint(0, schedule.timesteps, (z0.shape[0],), device=device)
            xt, noise = schedule.q_sample(z0, t)
            # CFG 训练：随机 dropout 条件，使模型学会无条件分支
            drop = torch.rand(z0.shape[0], device=device) < diff_cfg["cfg_dropout"]
            pred = denoiser(xt, t, cond, cond_mask=(~drop).float())

            loss = torch.nn.functional.mse_loss(pred, noise)
            # 可选物理引导：后半程对齐预测 x0 与真 x0 的“壁厚风险”代理
            if pg_on and epoch > diff_cfg["epochs"] // 3:
                x0_hat_enc = schedule.predict_x0(xt, t, pred)
                z_raw_hat = codec.decode_to_raw(x0_hat_enc)
                risk_pred = physics_risk_proxy(ae.decode(z_raw_hat))
                with torch.no_grad():
                    z0_raw = codec.decode_to_raw(z0)
                    risk_gt = physics_risk_proxy(ae.decode(z0_raw))
                loss = loss + pg_w * torch.nn.functional.mse_loss(risk_pred, risk_gt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1

        val_loss = eval_noise_loss()
        gen_l1 = eval_generation_l1(denoiser, ae, schedule, codec, test_recs, device, diff_cfg)
        history["train_loss"].append(ep_loss / max(nb, 1))
        history["val_loss"].append(val_loss)
        history["val_gen_l1"].append(gen_l1)

        # 以生成 L1 为早停/保存依据（比纯噪声 MSE 更贴近下游）
        if gen_l1 < best_gen:
            best_gen = gen_l1
            torch.save(
                {"model": denoiser.state_dict(), "epoch": epoch, "mode": mode, "val_gen_l1": gen_l1, "pca_dim": pca.dim},
                run_dir / "best_diffusion.pt",
            )

        if epoch == 1 or epoch % 20 == 0 or epoch == diff_cfg["epochs"]:
            print(f"[Diff] ep{epoch}: train={history['train_loss'][-1]:.5f} val={val_loss:.5f} gen_l1={gen_l1:.4f}")

    plot_training_curves(history, run_dir / "training_curves.png", "Diffusion Training")
    plot_training_dashboard(history, run_dir / "training_dashboard.png", "Diffusion Training Dashboard")
    save_json(history, run_dir / "training_history.json")

    denoiser.load_state_dict(torch.load(run_dir / "best_diffusion.pt", map_location=device, weights_only=False)["model"])
    denoiser.eval()

    # 全测试集：条件生成 -> AE 解码 -> 与原始 SDF 对比
    vis = run_dir / "generations"
    vis.mkdir(exist_ok=True)
    metrics, grid_items = [], []

    with torch.no_grad():
        for r in test_recs:
            cond = torch.from_numpy(r.condition_norm).unsqueeze(0).to(device)
            z_gen_enc = schedule.sample(
                denoiser,
                sample_shape,
                cond,
                diff_cfg.get("cfg_scale", 1.5),
                diff_cfg.get("sample_steps", 50),
                use_ddim=diff_cfg.get("use_ddim", True),
            )
            z_gen = codec.decode_to_raw(z_gen_enc)
            z_gt = torch.from_numpy(r.z_m).unsqueeze(0).to(device)

            npz_path = root / cfg["npz_dir"] / f"{r.sample_id}.npz"
            sdf_orig = torch.from_numpy(np.load(npz_path)["sdf2d_norm"].astype(np.float32)).to(device)
            sdf_gen = ae.decode(z_gen)[0, 0]
            cond_dict = {c: float(r.condition_raw[i]) for i, c in enumerate(cond_stats.columns)}
            gt_np, gen_np = to_numpy(sdf_orig), to_numpy(sdf_gen)
            plot_sdf_panel(gt_np, gen_np, f"Gen {r.sample_id}", vis / f"{r.sample_id}.png", cond_dict)
            l1 = float(torch.nn.functional.l1_loss(sdf_gen, sdf_orig))
            l1_recon = float(torch.nn.functional.l1_loss(ae.decode(z_gt)[0, 0], sdf_orig))
            metrics.append({"sample_id": r.sample_id, "l1_vs_original": l1, "l1_ae_recon": l1_recon, "condition": cond_dict})
            grid_items.append({"sample_id": r.sample_id, "gt": gt_np, "pred": gen_np})

    plot_sample_grid(grid_items, run_dir / "grid_test.png", "Diffusion Generations vs Original SDF", max_samples=10)
    plot_per_sample_metrics(
        [{"sample_id": m["sample_id"], "l1": m["l1_vs_original"]} for m in metrics],
        run_dir / "metrics_test.png",
        "Per-sample Generation L1 vs Original",
    )
    mean_l1 = float(np.mean([m["l1_vs_original"] for m in metrics]))
    mean_recon = float(np.mean([m["l1_ae_recon"] for m in metrics]))
    save_json(
        {"generations": metrics, "mean_l1_vs_original": mean_l1, "mean_ae_recon_l1": mean_recon},
        run_dir / "generation_metrics.json",
    )
    save_json(
        {
            "ae_run_dir": str(ae_run_dir),
            "mean_gen_l1": mean_l1,
            "mean_ae_recon_l1": mean_recon,
            "best_gen_l1": best_gen,
            "pca_dim": pca.dim,
            "denoiser": mode,
        },
        run_dir / "summary.json",
    )
    print(f"Diffusion done -> {run_dir} | mean_gen_l1={mean_l1:.4f} (AE recon={mean_recon:.4f})")
    return run_dir
