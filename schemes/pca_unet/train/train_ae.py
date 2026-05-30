"""PCA-UNet 第一阶段：训练子午面 SDF 自编码器。

流程：数据加载 → AE 训练 → 潜记录导出 → 门控评估 → 重建可视化。
训练产物供 ``train_diffusion`` 做 PCA 拟合与条件扩散建模。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from schemes.pca_unet.data.dataset import ConditionStats, MeridianSDFDataset
from schemes.pca_unet.data.splits import load_condition_stats, load_splits
from schemes.pca_unet.gate.latent_gate import GateConfig, LatentRepresentationGate
from schemes.pca_unet.losses.ae_losses import AELossWeights, compute_ae_losses
from schemes.pca_unet.models.autoencoder import MeridianAutoEncoder
from schemes.pca_unet.records.body_latent import BodyLatentRecord, save_records
from schemes.pca_unet.utils.paths import project_root
from schemes.pca_unet.utils.visualization import (
    plot_gate_report,
    plot_latent_pca,
    plot_per_sample_metrics,
    plot_sample_grid,
    plot_sdf_panel,
    plot_train_test_metric_compare,
    plot_training_curves,
    plot_training_dashboard,
    save_json,
    to_numpy,
)


@torch.no_grad()
def _physics_for_sample(batch_physics, index: int) -> dict:
    """从 batch 物理摘要中提取单样本 dict（兼容 list 与 collated dict）。"""
    if isinstance(batch_physics, list):
        return batch_physics[index]
    return {
        "min_thickness_mm": float(batch_physics["min_thickness_mm"][index]),
        "area_mm2": float(batch_physics["area_mm2"][index]),
    }


@torch.no_grad()
def build_latent_records(
    model: MeridianAutoEncoder,
    loader: DataLoader,
    device: torch.device,
) -> list[BodyLatentRecord]:
    """遍历 DataLoader，编码全部样本并记录潜向量与重建指标。

    用于生成 ``body_latent_train.json`` / ``body_latent_test.json``，
    供 PCA 拟合与扩散训练使用。
    """
    model.eval()
    w = AELossWeights()
    records: list[BodyLatentRecord] = []
    for batch in loader:
        x = batch["x"].to(device)
        sdf = batch["sdf"].to(device)
        sem = batch["semantic"].to(device)
        z_m, recon = model(x)
        for i in range(x.shape[0]):
            losses = compute_ae_losses(
                recon[i : i + 1], sdf[i : i + 1], sem[i : i + 1],
                _physics_for_sample(batch["physics"], i),
                w,
            )
            records.append(
                BodyLatentRecord(
                    sample_id=batch["sample_id"][i],
                    z_m=to_numpy(z_m[i]).astype(np.float32),
                    condition_raw=to_numpy(batch["condition_raw"][i]).astype(np.float32),
                    condition_norm=to_numpy(batch["condition"][i]).astype(np.float32),
                    recon_l1=float(losses["l1"]),
                    recon_l2=float(losses["l2"]),
                    zero_band_l1=float(losses["zero_band"]),
                    dim_error=float(losses["dimension"]),
                    semantic_error=float(losses["semantic"]),
                    npz_path=batch["npz_path"][i],
                )
            )
    return records


def train_autoencoder(cfg: dict[str, Any], run_dir: Path) -> Path:
    """执行自编码器完整训练流水线。

    Args:
        cfg: 合并后的方案配置（含 data_root、autoencoder、latent_gate 等）。
        run_dir: AE 阶段输出目录。

    Returns:
        AE 运行目录路径（含 best_autoencoder.pt、body_latent_*.json 等）。
    """
    root = project_root()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")

    # --- 数据准备 ---
    param_path = root / cfg.get("param_table_path", "")
    param_table = pd.read_csv(param_path) if param_path.is_file() else None
    condition_columns = cfg.get("condition_columns")
    processed_dir = Path(cfg["data_root"])
    train_df, test_df = load_splits(processed_dir)
    cond_stats = load_condition_stats(processed_dir, condition_columns)
    save_json(cond_stats.to_dict(), run_dir / "condition_stats.json")

    ae_cfg = cfg["autoencoder"]
    npz_dir = root / cfg["npz_dir"]
    train_ds = MeridianSDFDataset(train_df, npz_dir, param_table, cond_stats, ae_cfg["use_semantic"])
    test_ds = MeridianSDFDataset(test_df, npz_dir, param_table, cond_stats, ae_cfg["use_semantic"])
    nw = cfg.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=ae_cfg["batch_size"], shuffle=True, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=ae_cfg["batch_size"], shuffle=False, num_workers=nw)

    # --- 模型与优化器 ---
    in_ch = 2 if ae_cfg["use_semantic"] else 1
    model = MeridianAutoEncoder(in_ch, ae_cfg["latent_channels"], ae_cfg["latent_spatial"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ae_cfg["lr"])
    lw = AELossWeights(**ae_cfg["loss_weights"])
    history: dict[str, list[float]] = {"train_total": [], "train_l1": [], "val_l1": [], "val_total": []}
    best_val = float("inf")

    # --- 训练循环 ---
    for epoch in range(1, ae_cfg["epochs"] + 1):
        model.train()
        tr_tot, tr_l1, n = 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc=f"AE {epoch}/{ae_cfg['epochs']}", leave=False):
            x, sdf, sem = batch["x"].to(device), batch["sdf"].to(device), batch["semantic"].to(device)
            opt.zero_grad()
            _, recon = model(x)
            losses = compute_ae_losses(recon, sdf, sem, batch["physics"], lw)
            losses["total"].backward()
            opt.step()
            tr_tot += losses["total"].item()
            tr_l1 += losses["l1"].item()
            n += 1
        history["train_total"].append(tr_tot / n)
        history["train_l1"].append(tr_l1 / n)

        # 验证集评估
        model.eval()
        vl_tot, vl_l1, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                x, sdf, sem = batch["x"].to(device), batch["sdf"].to(device), batch["semantic"].to(device)
                _, recon = model(x)
                losses = compute_ae_losses(recon, sdf, sem, batch["physics"], lw)
                vl_tot += losses["total"].item()
                vl_l1 += losses["l1"].item()
                vn += 1
        history["val_total"].append(vl_tot / max(vn, 1))
        history["val_l1"].append(vl_l1 / max(vn, 1))

        # 按验证 L1 保存最优检查点
        if history["val_l1"][-1] < best_val:
            best_val = history["val_l1"][-1]
            torch.save({"model": model.state_dict(), "in_channels": in_ch, "epoch": epoch}, run_dir / "best_autoencoder.pt")

        if epoch == 1 or epoch % 10 == 0 or epoch == ae_cfg["epochs"]:
            print(f"[AE] ep{epoch}: train_l1={history['train_l1'][-1]:.4f} val_l1={history['val_l1'][-1]:.4f}")

    plot_training_curves(history, run_dir / "training_curves.png", "AutoEncoder Training")
    plot_training_dashboard(history, run_dir / "training_dashboard.png", "AutoEncoder Training Dashboard")
    save_json(history, run_dir / "training_history.json")
    
    print("已保存训练结果，正在加载最优权重...")
    # --- 最优模型：导出潜记录与门控 ---
    ckpt = torch.load(run_dir / "best_autoencoder.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    train_records = build_latent_records(model, train_loader, device)
    test_records = build_latent_records(model, test_loader, device)
    save_records(train_records, run_dir / "body_latent_train.json")
    save_records(test_records, run_dir / "body_latent_test.json")

    gate = LatentRepresentationGate(GateConfig(**cfg["latent_gate"]))
    gate_result = gate.evaluate(train_records)
    save_json(gate_result.report, run_dir / "latent_gate_report.json")
    plot_gate_report(gate_result.report, run_dir / "latent_gate_report.png")
    plot_latent_pca(np.stack([r.z_m.reshape(-1) for r in test_records]), run_dir / "latent_pca_test.png",
                    [r.sample_id for r in test_records])
    plot_train_test_metric_compare(
        [r.recon_l1 for r in train_records], [r.recon_l1 for r in test_records],
        run_dir / "recon_l1_train_vs_test.png",
    )

    # --- 重建可视化（仅测试集）---
    vis = run_dir / "visualizations"
    for split_name, records, loader in [
        ("test", test_records, test_loader),
    ]:
        out_dir = vis / f"reconstructions_{split_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_items: list[dict] = []
        metrics: list[dict] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x, sdf = batch["x"].to(device), batch["sdf"].to(device)
                _, recon = model(x)
                for i in range(x.shape[0]):
                    sid = batch["sample_id"][i]
                    gt_np, pred_np = to_numpy(sdf[i, 0]), to_numpy(recon[i, 0])
                    cond = {c: float(batch["condition_raw"][i][j]) for j, c in enumerate(cond_stats.columns)}
                    l1 = float(np.abs(gt_np - pred_np).mean())
                    plot_sdf_panel(gt_np, pred_np, f"Recon [{split_name}] {sid}", out_dir / f"{sid}.png", cond)
                    grid_items.append({"sample_id": sid, "gt": gt_np, "pred": pred_np})
                    metrics.append({"sample_id": sid, "l1": l1})
        plot_sample_grid(grid_items, vis / f"grid_{split_name}.png",
                         f"AE Recon ({split_name})", max_samples=10)
        plot_per_sample_metrics(metrics, vis / f"metrics_{split_name}.png", f"Recon L1 ({split_name})")
        save_json({"metrics": metrics, "mean_l1": float(np.mean([m["l1"] for m in metrics]))},
                  vis / f"metrics_{split_name}.json")

    save_json({"gate_passed": gate_result.passed, "pass_ratio": gate_result.pass_ratio, "best_val_l1": best_val}, run_dir / "summary.json")
    print(f"AE done -> {run_dir} | gate={gate_result.passed} ratio={gate_result.pass_ratio:.1%}")
    return run_dir
