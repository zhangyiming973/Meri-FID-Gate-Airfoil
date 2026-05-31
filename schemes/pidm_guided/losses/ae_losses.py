"""自编码器损失：SDF 重建、零等值面加权、物理量与语义一致性。

同时提供 physics_risk_proxy，供扩散阶段的物理引导（physics guidance）使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AELossWeights:
    """各损失项的标量权重。"""

    l1: float = 1.0
    l2: float = 0.5
    zero_level: float = 2.0  # 零等值面附近加权 L1
    zero_sigma_mm: float = 1.5  # 零带高斯权重宽度（毫米尺度）
    dimension: float = 0.3  # 最小厚度、面积物理约束
    semantic: float = 0.2  # 语义区域内外一致性


def zero_level_weights(sdf: torch.Tensor, sigma_mm: float = 1.5, sdf_scale: float = 12.0) -> torch.Tensor:
    """在 SDF≈0 附近赋予更大权重的空间权重图。

    sdf_scale 将归一化 SDF 映射到约 ±12mm 物理范围。
    """
    return torch.exp(-(sdf * sdf_scale).abs() / sigma_mm)


def estimate_min_thickness(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """从 SDF 估计最小壁厚（沿扫描方向的 inside 区域宽度代理）。"""
    inside = (sdf < 0).float()
    return inside.sum(dim=-1).amax(dim=-1) * (2 * sdf_scale / sdf.shape[-1])


def estimate_area(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """从 SDF 零等值面内区域估计 2D 截面积（mm²）。"""
    cell = (2 * sdf_scale / sdf.shape[-1]) ** 2
    return (sdf < 0).float().sum(dim=(-2, -1)) * cell


def semantic_consistency_loss(recon: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
    """重建 SDF 在语义分区内的“实体覆盖率”应与语义标签一致。

    将 semantic 分为低/中/高三档区域，惩罚 inside 覆盖率偏离 1 的情况。
    """
    inside = (recon < 0).float()
    regions = [
        (semantic > 0.1) & (semantic < 0.4),
        (semantic > 0.4) & (semantic < 0.8),
        semantic > 0.8,
    ]
    losses = []
    for mask in regions:
        m = mask.float()
        if m.sum() < 1:
            continue
        cov = (inside * m).sum(dim=(-2, -1)) / (m.sum(dim=(-2, -1)) + 1e-6)
        losses.append((1.0 - cov).mean())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=recon.device)


def _physics_tensors(
    physics: list[dict] | dict | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """将 batch 物理标注统一为 (min_thickness_mm, area_mm2) 张量对。"""
    if not physics:
        return None
    if isinstance(physics, dict) and "min_thickness_mm" in physics:
        gt_t = physics["min_thickness_mm"]
        gt_a = physics["area_mm2"]
        if not isinstance(gt_t, torch.Tensor):
            gt_t = torch.tensor(gt_t, device=device, dtype=dtype)
        if not isinstance(gt_a, torch.Tensor):
            gt_a = torch.tensor(gt_a, device=device, dtype=dtype)
        return gt_t.to(device=device, dtype=dtype), gt_a.to(device=device, dtype=dtype)
    if isinstance(physics, list):
        return (
            torch.tensor([p["min_thickness_mm"] for p in physics], device=device, dtype=dtype),
            torch.tensor([p["area_mm2"] for p in physics], device=device, dtype=dtype),
        )
    return None


def compute_ae_losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    semantic: torch.Tensor | None,
    physics: list[dict] | dict | None,
    w: AELossWeights,
    sdf_scale: float = 12.0,
) -> dict[str, torch.Tensor]:
    """计算 AE 训练/评估用的多项损失及加权和 total。

    Returns:
        含 total, l1, l2, zero_band, dimension, semantic 的字典。
    """
    l1 = F.l1_loss(recon, target)
    l2 = F.mse_loss(recon, target)
    zw = zero_level_weights(target, w.zero_sigma_mm, sdf_scale)
    zero_band = (zw * (recon - target).abs()).sum() / (zw.sum() + 1e-6)

    dim_loss = torch.tensor(0.0, device=recon.device)
    pt = _physics_tensors(physics, recon.device, recon.dtype)
    if pt is not None:
        gt_t, gt_a = pt
        est_thick = estimate_min_thickness(recon, sdf_scale)
        est_area = estimate_area(recon, sdf_scale)
        if est_thick.ndim > 1 and est_thick.shape[-1] == 1:
            est_thick = est_thick.squeeze(-1)
        if est_area.ndim > 1 and est_area.shape[-1] == 1:
            est_area = est_area.squeeze(-1)
        if gt_t.ndim == 0:
            gt_t = gt_t.unsqueeze(0)
        if gt_a.ndim == 0:
            gt_a = gt_a.unsqueeze(0)
        dim_loss = F.l1_loss(est_thick, gt_t) + F.l1_loss(est_area, gt_a) / 10000.0

    sem = torch.tensor(0.0, device=recon.device)
    if semantic is not None and w.semantic > 0:
        sem = semantic_consistency_loss(recon, semantic)

    total = w.l1 * l1 + w.l2 * l2 + w.zero_level * zero_band + w.dimension * dim_loss + w.semantic * sem
    return {"total": total, "l1": l1, "l2": l2, "zero_band": zero_band, "dimension": dim_loss, "semantic": sem}


def physics_risk_proxy(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """物理风险代理：壁厚越小风险越高（用于扩散阶段的辅助损失）。"""
    return 1.0 / (estimate_min_thickness(sdf, sdf_scale) + 1e-3)
