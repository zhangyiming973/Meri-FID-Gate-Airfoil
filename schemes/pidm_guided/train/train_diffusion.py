"""PIDM 风格条件扩散训练：PCA 潜空间 MLP 去噪 + 几何残差虚拟似然。

参考 PhysicsInformedDiffusionModels (ICLR 2025) 的损失设计：
  L = c_data * L_DDPM - c_residual * log p(r=0 | x0_pred, var_t)

几何残差在 AE 解码 SDF 上计算（壁厚、可选面积、Eikonal），目标趋近 0。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from schemes.pidm_guided.data.dataset import ConditionStats
from schemes.pidm_guided.models.autoencoder import MeridianAutoEncoder
from schemes.pidm_guided.models.diffusion import DiffusionSchedule
from schemes.pidm_guided.models.latent_pca import LatentPCA, save_latent_pca
from schemes.pidm_guided.models.mlp_denoiser import MLPDenoiser
from schemes.pidm_guided.physics.geometry_residual import GeometryResidualComputer
from schemes.pidm_guided.physics.pidm_loss import pidm_virtual_likelihood_loss
from schemes.pidm_guided.records.body_latent import load_records
from schemes.pidm_guided.train.diffusion_codec import DiffusionLatentCodec
from schemes.pidm_guided.utils.paths import project_root
from schemes.pidm_guided.utils.visualization import (
    plot_per_sample_metrics,
    plot_sample_grid,
    plot_sdf_panel,
    plot_training_curves,
    plot_training_dashboard,
    save_json,
    to_numpy,
)


def _build_denoiser(diff_cfg: dict, codec: DiffusionLatentCodec, cond_dim: int, device: torch.device):
    mode = diff_cfg.get("denoiser", "mlp")
    if mode == "unet":
        raise ValueError("Raw UNet denoiser is not supported in pidm_guided scheme; use pca_unet/dim_unet instead")
    return MLPDenoiser(codec.pca.dim, cond_dim, hidden=diff_cfg.get("mlp_hidden", 512)).to(device), "mlp"


def _load_area_targets(records, root: Path, cfg: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """从 NPZ physics 字段加载各样本目标截面积。"""
    targets: dict[str, torch.Tensor] = {}
    npz_dir = root / cfg["npz_dir"]
    for r in records:
        npz_path = npz_dir / f"{r.sample_id}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        if "physics" in data:
            phys = data["physics"].item() if data["physics"].ndim == 0 else data["physics"]
            if isinstance(phys, dict) and "area_mm2" in phys:
                targets[r.sample_id] = torch.tensor(float(phys["area_mm2"]), device=device)
    return targets


def _make_x0_corrector(
    ae: MeridianAutoEncoder,
    codec: DiffusionLatentCodec,
    residual_computer: GeometryResidualComputer,
    schedule: DiffusionSchedule,
    pidm_cfg: dict,
    area_by_id: dict[str, torch.Tensor],
    sample_ids: list[str],
):
    """构建 DDIM 采样时的 x0 残差梯度校正闭包。"""
    n_steps = int(pidm_cfg.get("n_correction", 0))
    m_steps = int(pidm_cfg.get("m_correction", 0))
    step_size = float(pidm_cfg.get("correction_step_size", 0.05))
    if n_steps <= 0 and m_steps <= 0:
        return None, 0, 0

    def _decode_residual(x0_enc: torch.Tensor, batch_idx: slice | None = None) -> torch.Tensor:
        z_raw = codec.decode_to_raw(x0_enc)
        sdf = ae.decode(z_raw)
        area_target = None
        if residual_computer.use_area_constraint and area_by_id:
            ids = sample_ids if batch_idx is None else sample_ids[batch_idx]
            if isinstance(ids, str):
                ids = [ids]
            area_vals = [area_by_id.get(sid) for sid in ids]
            if all(v is not None for v in area_vals):
                area_target = torch.stack(area_vals)
        return residual_computer(sdf, area_target=area_target)

    def corrector(x0: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        return schedule.correct_x0(
            x0,
            lambda enc: _decode_residual(enc).sum(),
            steps=max(n_steps, m_steps),
            step_size=step_size,
        )

    return corrector, n_steps, m_steps


@torch.no_grad()
def eval_generation_l1(
    denoiser,
    ae,
    schedule,
    codec: DiffusionLatentCodec,
    test_recs,
    device,
    diff_cfg,
    pidm_cfg: dict,
    residual_computer: GeometryResidualComputer,
    area_by_id: dict[str, torch.Tensor],
    max_samples: int = 5,
) -> tuple[float, float]:
    """快速生成评估：返回 (gen_l1, mean_physics_residual)。"""
    denoiser.eval()
    ae.eval()
    losses, res_vals = [], []
    sample_shape = codec.sample_shape(1)
    corrector, n_corr, m_corr = _make_x0_corrector(
        ae, codec, residual_computer, schedule, pidm_cfg, area_by_id, [r.sample_id for r in test_recs[:max_samples]]
    )

    for r in test_recs[:max_samples]:
        cond = torch.from_numpy(r.condition_norm).unsqueeze(0).to(device)
        z_gen_enc = schedule.sample(
            denoiser,
            sample_shape,
            cond,
            cfg_scale=diff_cfg.get("cfg_scale", 1.5),
            steps=diff_cfg.get("sample_steps", 50),
            use_ddim=diff_cfg.get("use_ddim", True),
            x0_corrector=corrector,
            n_correction=n_corr,
            m_correction=m_corr,
        )
        z_gen = codec.decode_to_raw(z_gen_enc)
        z_gt = torch.from_numpy(r.z_m).unsqueeze(0).to(device)
        sdf_gen = ae.decode(z_gen)[0, 0]
        sdf_gt = ae.decode(z_gt)[0, 0]
        losses.append(float(torch.nn.functional.l1_loss(sdf_gen, sdf_gt)))

        area_target = area_by_id.get(r.sample_id)
        with torch.enable_grad():
            z_enc = z_gen_enc.detach().requires_grad_(False)
            z_raw = codec.decode_to_raw(z_enc)
            sdf = ae.decode(z_raw)
            res = residual_computer(sdf, area_target=area_target.unsqueeze(0) if area_target is not None else None)
            res_vals.append(float(residual_computer.mean_abs(res)))

    return (
        float(np.mean(losses)) if losses else float("inf"),
        float(np.mean(res_vals)) if res_vals else float("inf"),
    )


def train_diffusion(cfg: dict[str, Any], ae_run_dir: Path, run_dir: Path) -> Path:
    """在 AE 潜空间 PCA 系数上训练 PIDM 风格条件扩散模型。"""
    root = project_root()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")

    ae_cfg, diff_cfg = cfg["autoencoder"], cfg["diffusion"]
    pidm_cfg = cfg.get("pidm", {})
    c_data = float(pidm_cfg.get("c_data", 1.0))
    c_residual = float(pidm_cfg.get("c_residual", 0.001))
    warmup_frac = float(pidm_cfg.get("warmup_frac", 0.33))

    residual_computer = GeometryResidualComputer(
        min_thickness_mm=float(pidm_cfg.get("min_thickness_mm", 2.0)),
        use_area_constraint=bool(pidm_cfg.get("use_area_constraint", False)),
        use_eikonal=bool(pidm_cfg.get("use_eikonal", True)),
        eikonal_weight=float(pidm_cfg.get("eikonal_weight", 0.1)),
        sdf_scale=float(pidm_cfg.get("sdf_scale", 12.0)),
    )

    cond_stats = ConditionStats.from_dict(json.loads((ae_run_dir / "condition_stats.json").read_text()))
    train_recs = load_records(ae_run_dir / "body_latent_train.json")
    test_recs = load_records(ae_run_dir / "body_latent_test.json")
    area_by_id = _load_area_targets(train_recs + test_recs, root, cfg, device)

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

    ckpt = torch.load(ae_run_dir / "best_autoencoder.pt", map_location=device, weights_only=False)
    ae = MeridianAutoEncoder(ckpt["in_channels"], ae_cfg["latent_channels"], ae_cfg["latent_spatial"]).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()

    denoiser, mode = _build_denoiser(diff_cfg, codec, len(cond_stats.columns), device)
    schedule = DiffusionSchedule(diff_cfg["timesteps"], device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=diff_cfg["lr"], weight_decay=1e-4)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_data_loss": [],
        "train_physics_loss": [],
        "train_residual_mean": [],
        "val_loss": [],
        "val_gen_l1": [],
        "val_physics_residual": [],
    }
    best_gen = float("inf")
    sample_shape = codec.sample_shape(1)

    def eval_noise_loss() -> float:
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
        ep_loss = ep_data = ep_phys = ep_res = 0.0
        nb = 0
        pidm_on = c_residual > 0 and epoch > diff_cfg["epochs"] * warmup_frac

        for z0, cond in tqdm(train_loader, desc=f"PIDM-Diff {epoch}/{diff_cfg['epochs']}", leave=False):
            z0, cond = z0.to(device), cond.to(device)
            t = torch.randint(0, schedule.timesteps, (z0.shape[0],), device=device)
            xt, noise = schedule.q_sample(z0, t)
            drop = torch.rand(z0.shape[0], device=device) < diff_cfg["cfg_dropout"]
            pred = denoiser(xt, t, cond, cond_mask=(~drop).float())

            data_loss = torch.nn.functional.mse_loss(pred, noise)
            loss = c_data * data_loss
            phys_loss = torch.tensor(0.0, device=device)
            res_mean = 0.0

            if pidm_on:
                x0_hat_enc = schedule.predict_x0(xt, t, pred)
                z_raw_hat = codec.decode_to_raw(x0_hat_enc)
                sdf_hat = ae.decode(z_raw_hat)
                residual = residual_computer(sdf_hat)
                post_var = schedule.posterior_var_at(t, z0)
                phys_loss = pidm_virtual_likelihood_loss(residual, post_var, c_residual)
                res_mean = float(residual_computer.mean_abs(residual).item())
                loss = loss + phys_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item()
            ep_data += data_loss.item()
            ep_phys += float(phys_loss.item())
            ep_res += res_mean
            nb += 1

        val_loss = eval_noise_loss()
        gen_l1, val_res = eval_generation_l1(
            denoiser, ae, schedule, codec, test_recs, device, diff_cfg, pidm_cfg,
            residual_computer, area_by_id,
        )
        history["train_loss"].append(ep_loss / max(nb, 1))
        history["train_data_loss"].append(ep_data / max(nb, 1))
        history["train_physics_loss"].append(ep_phys / max(nb, 1))
        history["train_residual_mean"].append(ep_res / max(nb, 1))
        history["val_loss"].append(val_loss)
        history["val_gen_l1"].append(gen_l1)
        history["val_physics_residual"].append(val_res)

        if gen_l1 < best_gen:
            best_gen = gen_l1
            torch.save(
                {"model": denoiser.state_dict(), "epoch": epoch, "mode": mode, "val_gen_l1": gen_l1, "pca_dim": pca.dim},
                run_dir / "best_diffusion.pt",
            )

        if epoch == 1 or epoch % 20 == 0 or epoch == diff_cfg["epochs"]:
            print(
                f"[PIDM-Diff] ep{epoch}: train={history['train_loss'][-1]:.5f} "
                f"data={history['train_data_loss'][-1]:.5f} phys={history['train_physics_loss'][-1]:.5f} "
                f"res={history['train_residual_mean'][-1]:.4f} val={val_loss:.5f} gen_l1={gen_l1:.4f} "
                f"val_res={val_res:.4f}"
            )

    plot_training_curves(history, run_dir / "training_curves.png", "PIDM Diffusion Training")
    plot_training_dashboard(history, run_dir / "training_dashboard.png", "PIDM Diffusion Training Dashboard")
    save_json(history, run_dir / "training_history.json")
    save_json({"pidm": pidm_cfg, "residual_components": _residual_component_names(residual_computer)}, run_dir / "pidm_config.json")

    denoiser.load_state_dict(torch.load(run_dir / "best_diffusion.pt", map_location=device, weights_only=False)["model"])
    denoiser.eval()

    corrector, n_corr, m_corr = _make_x0_corrector(
        ae, codec, residual_computer, schedule, pidm_cfg, area_by_id, [r.sample_id for r in test_recs]
    )

    vis = run_dir / "generations"
    vis.mkdir(exist_ok=True)
    metrics, grid_items = [], []

    with torch.no_grad() if (n_corr <= 0 and m_corr <= 0) else torch.enable_grad():
        for r in test_recs:
            cond = torch.from_numpy(r.condition_norm).unsqueeze(0).to(device)
            z_gen_enc = schedule.sample(
                denoiser,
                sample_shape,
                cond,
                diff_cfg.get("cfg_scale", 1.5),
                diff_cfg.get("sample_steps", 50),
                use_ddim=diff_cfg.get("use_ddim", True),
                x0_corrector=corrector,
                n_correction=n_corr,
                m_correction=m_corr,
            )
            z_gen = codec.decode_to_raw(z_gen_enc)
            z_gt = torch.from_numpy(r.z_m).unsqueeze(0).to(device)

            npz_path = root / cfg["npz_dir"] / f"{r.sample_id}.npz"
            sdf_orig = torch.from_numpy(np.load(npz_path)["sdf2d_norm"].astype(np.float32)).to(device)
            sdf_gen = ae.decode(z_gen)[0, 0]
            area_target = area_by_id.get(r.sample_id)
            res = residual_computer(
                sdf_gen.unsqueeze(0).unsqueeze(0),
                area_target=area_target.unsqueeze(0) if area_target is not None else None,
            )
            cond_dict = {c: float(r.condition_raw[i]) for i, c in enumerate(cond_stats.columns)}
            gt_np, gen_np = to_numpy(sdf_orig), to_numpy(sdf_gen)
            plot_sdf_panel(gt_np, gen_np, f"Gen {r.sample_id}", vis / f"{r.sample_id}.png", cond_dict)
            l1 = float(torch.nn.functional.l1_loss(sdf_gen, sdf_orig))
            l1_recon = float(torch.nn.functional.l1_loss(ae.decode(z_gt)[0, 0], sdf_orig))
            metrics.append({
                "sample_id": r.sample_id,
                "l1_vs_original": l1,
                "l1_ae_recon": l1_recon,
                "physics_residual_mean": float(residual_computer.mean_abs(res)),
                "condition": cond_dict,
            })
            grid_items.append({"sample_id": r.sample_id, "gt": gt_np, "pred": gen_np})

    plot_sample_grid(grid_items, run_dir / "grid_test.png", "PIDM Diffusion Generations vs Original SDF", max_samples=10)
    plot_per_sample_metrics(
        [{"sample_id": m["sample_id"], "l1": m["l1_vs_original"]} for m in metrics],
        run_dir / "metrics_test.png",
        "Per-sample Generation L1 vs Original",
    )
    mean_l1 = float(np.mean([m["l1_vs_original"] for m in metrics]))
    mean_recon = float(np.mean([m["l1_ae_recon"] for m in metrics]))
    mean_res = float(np.mean([m["physics_residual_mean"] for m in metrics]))
    save_json(
        {
            "generations": metrics,
            "mean_l1_vs_original": mean_l1,
            "mean_ae_recon_l1": mean_recon,
            "mean_physics_residual": mean_res,
        },
        run_dir / "generation_metrics.json",
    )
    save_json(
        {
            "ae_run_dir": str(ae_run_dir),
            "mean_gen_l1": mean_l1,
            "mean_ae_recon_l1": mean_recon,
            "mean_physics_residual": mean_res,
            "best_gen_l1": best_gen,
            "pca_dim": pca.dim,
            "denoiser": mode,
            "pidm": pidm_cfg,
        },
        run_dir / "summary.json",
    )
    print(f"PIDM Diffusion done -> {run_dir} | mean_gen_l1={mean_l1:.4f} mean_res={mean_res:.4f} (AE recon={mean_recon:.4f})")
    return run_dir


def _residual_component_names(computer: GeometryResidualComputer) -> list[str]:
    names = ["thickness_violation"]
    if computer.use_area_constraint:
        names.append("area_deviation")
    if computer.use_eikonal:
        names.append("eikonal")
    return names
