"""PCA-UNet 第二阶段：在 PCA 网格上训练条件 UNet 扩散模型。

流程：加载 AE 潜记录 → PCA 拟合 → 网格编解码 → UNet 噪声预测训练
（含可选物理引导）→ DDIM 采样生成 → 与原始 SDF 对比评估。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from schemes.pca_unet.data.dataset import ConditionStats
from schemes.pca_unet.losses.ae_losses import physics_risk_proxy
from schemes.pca_unet.models.autoencoder import MeridianAutoEncoder
from schemes.pca_unet.models.diffusion import DiffusionSchedule
from schemes.pca_unet.models.latent_pca import LatentPCA, save_latent_pca
from schemes.pca_unet.models.pca_unet import build_pca_unet
from schemes.pca_unet.records.body_latent import load_records
from schemes.pca_unet.train.diffusion_codec import DiffusionLatentCodec
from schemes.pca_unet.utils.paths import project_root
from schemes.pca_unet.utils.visualization import (
    plot_per_sample_metrics,
    plot_sample_grid,
    plot_sdf_panel,
    plot_training_curves,
    plot_training_dashboard,
    save_json,
    to_numpy,
)


def build_diffusion_train_loader(
    train_recs,
    z_train_np: np.ndarray,
    pca_dim: int,
    batch_size: int,
) -> DataLoader:
    """由 AE latent records 构建 PCA 网格扩散训练 DataLoader。"""
    pca = LatentPCA.fit(z_train_np, pca_dim)
    codec = DiffusionLatentCodec.from_mode("pca_unet", pca, z_train_np.shape)
    z_enc = codec.encode_numpy(z_train_np)
    c_train = torch.from_numpy(np.stack([r.condition_norm for r in train_recs])).float()
    return DataLoader(TensorDataset(z_enc, c_train), batch_size=batch_size, shuffle=True)


def load_autoencoder_from_checkpoint(
    ckpt_path: Path,
    ae_cfg: dict[str, Any],
    device: torch.device,
) -> MeridianAutoEncoder:
    """按 checkpoint 保存的通道和尺寸配置恢复冻结 AE。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ae = MeridianAutoEncoder(
        ckpt["in_channels"],
        ae_cfg["latent_channels"],
        ckpt.get("latent_spatial", ae_cfg["latent_spatial"]),
        output_size=ckpt.get("input_size", ae_cfg.get("input_size", [256, 256])),
    ).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    return ae


def build_pca_unet_for_conditions(
    pca_dim: int,
    cond_stats: ConditionStats,
    base_ch: int = 64,
):
    """按 condition_stats 列数构建 PCA-UNet 去噪器。"""
    return build_pca_unet(pca_dim, len(cond_stats.columns), base_ch=base_ch)


@torch.no_grad()
def _eval_generation_l1(
    unet,
    ae,
    schedule,
    codec: DiffusionLatentCodec,
    test_recs,
    device,
    unet_cfg: dict,
    max_samples: int = 5,
) -> float:
    """在少量测试样本上评估生成质量：条件采样 z → 解码 SDF 与 GT 潜的 L1。

    用于训练过程中按 ``val_gen_l1`` 选取最优 UNet 检查点。
    """
    unet.eval()
    ae.eval()
    losses = []
    sample_shape = codec.sample_shape(1)
    for r in test_recs[:max_samples]:
        cond = torch.from_numpy(r.condition_norm).unsqueeze(0).to(device)
        # DDIM + CFG 从纯噪声采样 PCA 网格
        z_gen_enc = schedule.sample(
            unet,
            sample_shape,
            cond,
            cfg_scale=unet_cfg.get("cfg_scale", 1.5),
            steps=unet_cfg.get("sample_steps", 50),
            use_ddim=unet_cfg.get("use_ddim", True),
        )
        z_gen = codec.decode_to_raw(z_gen_enc)
        z_gt = torch.from_numpy(r.z_m).unsqueeze(0).to(device)
        sdf_gen = ae.decode(z_gen)[0, 0]
        sdf_gt = ae.decode(z_gt)[0, 0]
        losses.append(float(torch.nn.functional.l1_loss(sdf_gen, sdf_gt)))
    return float(np.mean(losses)) if losses else float("inf")


def train_diffusion(cfg: dict[str, Any], ae_run_dir: Path, run_dir: Path) -> Path:
    """在 AE 潜空间的 PCA 网格上训练条件 UNet 扩散模型。

    Args:
        cfg: 方案配置。
        ae_run_dir: 已完成 AE 训练目录（含检查点与 body_latent_*.json）。
        run_dir: 扩散阶段输出目录。

    Returns:
        扩散运行目录路径。
    """
    root = project_root()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")

    ae_cfg = cfg["autoencoder"]
    unet_cfg = cfg.get("unet", cfg.get("diffusion", {}))
    cond_stats = ConditionStats.from_dict(json.loads((ae_run_dir / "condition_stats.json").read_text()))
    train_recs = load_records(ae_run_dir / "body_latent_train.json")
    test_recs = load_records(ae_run_dir / "body_latent_test.json")

    # --- PCA 降维：将高维 z_m 压缩为 k 维，再重塑为 C×H×W 网格 ---
    z_train_np = np.stack([r.z_m for r in train_recs])
    z_shape = z_train_np.shape
    pca_dim = unet_cfg.get("pca_dim", 128)
    pca = LatentPCA.fit(z_train_np, pca_dim)
    save_latent_pca(pca, run_dir / "latent_pca.json")
    print(f"[UNet] PCA: {z_shape[1:]} -> {pca.dim} dims, std={pca.std:.4f}")

    codec = DiffusionLatentCodec.from_mode("pca_unet", pca, z_shape)
    assert codec.grid is not None
    print(f"[UNet] grid layout: {codec.grid.channels}×{codec.grid.height}×{codec.grid.width}")

    # 训练数据：编码后的 PCA 网格 + 标准化条件
    z_enc = codec.encode_numpy(z_train_np)
    c_train = torch.from_numpy(np.stack([r.condition_norm for r in train_recs])).float()
    train_loader = DataLoader(TensorDataset(z_enc, c_train), batch_size=unet_cfg["batch_size"], shuffle=True)

    # 加载冻结的 AE，用于物理引导与最终 SDF 解码
    ae = load_autoencoder_from_checkpoint(ae_run_dir / "best_autoencoder.pt", ae_cfg, device)

    unet, grid_spec = build_pca_unet_for_conditions(pca.dim, cond_stats, base_ch=unet_cfg.get("unet_base_ch", 64))
    unet = unet.to(device)
    schedule = DiffusionSchedule(unet_cfg["timesteps"], device)
    opt = torch.optim.AdamW(unet.parameters(), lr=unet_cfg["lr"], weight_decay=1e-4)
    pg_cfg = cfg.get("physics_guidance", {})
    pg_on, pg_w = pg_cfg.get("enabled", False), pg_cfg.get("weight", 0.02)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_gen_l1": []}
    best_gen = float("inf")
    sample_shape = codec.sample_shape(1)
    grid_meta = {
        "channels": grid_spec.channels,
        "height": grid_spec.height,
        "width": grid_spec.width,
    }

    def eval_noise_loss() -> float:
        """在测试集上评估噪声预测 MSE（未采样，直接加噪）。"""
        unet.eval()
        losses = []
        with torch.no_grad():
            for r in test_recs:
                z0 = codec.encode_raw(torch.from_numpy(r.z_m).unsqueeze(0).to(device))
                cond = torch.from_numpy(r.condition_norm).unsqueeze(0).to(device)
                t = torch.randint(0, schedule.timesteps, (1,), device=device)
                xt, noise = schedule.q_sample(z0, t)
                pred = unet(xt, t, cond)
                losses.append(float(torch.nn.functional.mse_loss(pred, noise)))
        return float(np.mean(losses))

    epochs = unet_cfg["epochs"]
    for epoch in range(1, epochs + 1):
        unet.train()
        ep_loss, nb = 0.0, 0
        for z0, cond in tqdm(train_loader, desc=f"UNet {epoch}/{epochs}", leave=False):
            z0, cond = z0.to(device), cond.to(device)
            t = torch.randint(0, schedule.timesteps, (z0.shape[0],), device=device)
            xt, noise = schedule.q_sample(z0, t)
            # CFG 训练：随机 dropout 条件，使模型同时学习有条件/无条件分支
            drop = torch.rand(z0.shape[0], device=device) < unet_cfg["cfg_dropout"]
            pred = unet(xt, t, cond, cond_mask=(~drop).float())

            loss = torch.nn.functional.mse_loss(pred, noise)
            # 训练后期可选物理引导：预测 x0 解码后的壁厚风险应与 GT 一致
            if pg_on and epoch > epochs // 3:
                x0_hat = schedule.predict_x0(xt, t, pred)
                risk_pred = physics_risk_proxy(ae.decode(codec.decode_to_raw(x0_hat)))
                with torch.no_grad():
                    risk_gt = physics_risk_proxy(ae.decode(codec.decode_to_raw(z0)))
                loss = loss + pg_w * torch.nn.functional.mse_loss(risk_pred, risk_gt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1

        val_loss = eval_noise_loss()
        gen_l1 = _eval_generation_l1(unet, ae, schedule, codec, test_recs, device, unet_cfg)
        history["train_loss"].append(ep_loss / max(nb, 1))
        history["val_loss"].append(val_loss)
        history["val_gen_l1"].append(gen_l1)

        # 按生成 L1 保存最优检查点（比纯噪声 MSE 更贴近下游任务）
        if gen_l1 < best_gen:
            best_gen = gen_l1
            torch.save(
                {
                    "model": unet.state_dict(),
                    "epoch": epoch,
                    "mode": "pca_unet",
                    "val_gen_l1": gen_l1,
                    "pca_dim": pca.dim,
                    "grid": grid_meta,
                },
                run_dir / "best_unet_diffusion.pt",
            )

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"[UNet] ep{epoch}: train={history['train_loss'][-1]:.5f} val={val_loss:.5f} gen_l1={gen_l1:.4f}")

    plot_training_curves(history, run_dir / "training_curves.png", "PCA-UNet Diffusion Training")
    plot_training_dashboard(history, run_dir / "training_dashboard.png", "PCA-UNet Training Dashboard")
    save_json(history, run_dir / "training_history.json")

    # --- 最优模型：全测试集条件生成与评估 ---
    unet.load_state_dict(torch.load(run_dir / "best_unet_diffusion.pt", map_location=device, weights_only=False)["model"])
    unet.eval()

    vis = run_dir / "generations"
    vis.mkdir(exist_ok=True)
    metrics, grid_items = [], []

    with torch.no_grad():
        for r in test_recs:
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
            z_gt = torch.from_numpy(r.z_m).unsqueeze(0).to(device)

            # 与 NPZ 原始 SDF 对比（非 AE 重建），衡量端到端生成质量
            npz_path = root / cfg["npz_dir"] / f"{r.sample_id}.npz"
            sdf_orig = torch.from_numpy(np.load(npz_path)["sdf2d_norm"].astype(np.float32)).to(device)
            sdf_gen = ae.decode(z_gen)[0, 0]
            cond_dict = {c: float(r.condition_raw[i]) for i, c in enumerate(cond_stats.columns)}
            gt_np, gen_np = to_numpy(sdf_orig), to_numpy(sdf_gen)
            plot_sdf_panel(gt_np, gen_np, f"UNet Gen {r.sample_id}", vis / f"{r.sample_id}.png", cond_dict)
            l1 = float(torch.nn.functional.l1_loss(sdf_gen, sdf_orig))
            l1_recon = float(torch.nn.functional.l1_loss(ae.decode(z_gt)[0, 0], sdf_orig))
            metrics.append({"sample_id": r.sample_id, "l1_vs_original": l1, "l1_ae_recon": l1_recon, "condition": cond_dict})
            grid_items.append({"sample_id": r.sample_id, "gt": gt_np, "pred": gen_np})

    plot_sample_grid(grid_items, run_dir / "grid_test.png", "PCA-UNet Generations vs Original SDF", max_samples=10)
    plot_per_sample_metrics(
        [{"sample_id": m["sample_id"], "l1": m["l1_vs_original"]} for m in metrics],
        run_dir / "metrics_test.png",
        "Per-sample UNet Generation L1",
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
            "denoiser": "pca_unet",
            "grid": grid_meta,
        },
        run_dir / "summary.json",
    )
    print(f"[UNet] done -> {run_dir} | mean_gen_l1={mean_l1:.4f} (AE recon={mean_recon:.4f})")
    return run_dir
