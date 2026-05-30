"""自编码器损失：SDF 重建、零等值面加权、轮缘区域加权、梯度与语义一致性。

轮缘（径向 r 较大 / 图像右侧）细节对生成质量影响大，通过 spatial_detail_weights
与 gradient 损失强化外缘重建。
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
    zero_level: float = 2.0
    zero_sigma_mm: float = 1.5
    dimension: float = 0.3
    semantic: float = 0.2
    rim: float = 1.5
    gradient: float = 0.5


def zero_level_weights(sdf: torch.Tensor, sigma_mm: float = 1.5, sdf_scale: float = 12.0) -> torch.Tensor:
    """在 SDF≈0 附近赋予更大权重的空间权重图。"""
    return torch.exp(-(sdf * sdf_scale).abs() / sigma_mm)


def spatial_detail_weights(
    semantic: torch.Tensor | None,
    r_grid: torch.Tensor | None,
    rim_boost: float = 4.0,
) -> torch.Tensor:
    """空间重建权重：语义轮缘区 + 径向靠外（图像列方向 r 增大）加权。

    Args:
        semantic: (B, 1, H, W) 归一化语义 mask。
        r_grid: (B, 1, H, W) 物理径向坐标网格；列方向对应 r 增大（右侧为轮缘）。
        rim_boost: 轮缘/外径向区域的额外权重倍率。

    Returns:
        逐像素权重，batch 内均值为 1。
    """
    if semantic is not None:
        w = 1.0 + rim_boost * semantic.clamp(0.0, 1.0)
    else:
        w = torch.ones(1, 1, 1, 1)

    if r_grid is not None:
        r = r_grid
        if r.ndim == 2:
            r = r.unsqueeze(0).unsqueeze(0)
        elif r.ndim == 3:
            r = r.unsqueeze(1)
        r_min = r.amin(dim=(-2, -1), keepdim=True)
        r_max = r.amax(dim=(-2, -1), keepdim=True)
        r_norm = (r - r_min) / (r_max - r_min + 1e-6)
        # 列方向 r 增大 → 右侧轮缘；平方使外缘权重更集中
        if semantic is not None:
            w = w + rim_boost * r_norm.pow(2)
        else:
            w = 1.0 + rim_boost * r_norm.pow(2)

    if semantic is None and r_grid is None:
        return w

    if w.ndim == 2:
        w = w.unsqueeze(0).unsqueeze(0)
    elif w.ndim == 3:
        w = w.unsqueeze(1)
    return w / (w.mean(dim=(-2, -1), keepdim=True) + 1e-6)


def rim_region_mask(
    semantic: torch.Tensor | None,
    r_grid: torch.Tensor | None,
    r_threshold: float = 0.72,
) -> torch.Tensor:
    """轮缘区域 mask：语义 web/rim + 径向外侧列。"""
    parts: list[torch.Tensor] = []
    if semantic is not None:
        parts.append((semantic > 0.55).float())
    if r_grid is not None:
        r = r_grid
        if r.ndim == 2:
            r = r.unsqueeze(0).unsqueeze(0)
        elif r.ndim == 3:
            r = r.unsqueeze(1)
        r_norm = (r - r.amin(dim=(-2, -1), keepdim=True)) / (
            r.amax(dim=(-2, -1), keepdim=True) - r.amin(dim=(-2, -1), keepdim=True) + 1e-6
        )
        parts.append((r_norm > r_threshold).float())
    if not parts:
        return torch.ones(1, 1, 1, 1)
    mask = parts[0]
    for p in parts[1:]:
        mask = torch.clamp(mask + p, 0.0, 1.0)
    return mask


def sdf_gradient_loss(recon: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """SDF 梯度匹配损失，强化零等值面附近的锐利边界。"""
    def _grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    gx_r, gy_r = _grad(recon)
    gx_t, gy_t = _grad(target)
    loss_x = (gx_r - gx_t).abs()
    loss_y = (gy_r - gy_t).abs()
    if weight is not None:
        wx = weight[..., :, 1:]
        wy = weight[..., 1:, :]
        loss_x = (loss_x * wx).sum() / (wx.sum() + 1e-6)
        loss_y = (loss_y * wy).sum() / (wy.sum() + 1e-6)
        return loss_x + loss_y
    return loss_x.mean() + loss_y.mean()


def weighted_l1(recon: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """逐像素加权 L1。"""
    return (weight * (recon - target).abs()).sum() / (weight.sum() + 1e-6)


def estimate_min_thickness(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """从 SDF 估计最小壁厚（沿扫描方向的 inside 区域宽度代理）。"""
    inside = (sdf < 0).float()
    return inside.sum(dim=-1).amax(dim=-1) * (2 * sdf_scale / sdf.shape[-1])


def estimate_area(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """从 SDF 零等值面内区域估计 2D 截面积（mm²）。"""
    cell = (2 * sdf_scale / sdf.shape[-1]) ** 2
    return (sdf < 0).float().sum(dim=(-2, -1)) * cell


def semantic_consistency_loss(recon: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
    """重建 SDF 在语义分区内的“实体覆盖率”应与语义标签一致。"""
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
    r_grid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """计算 AE 训练/评估用的多项损失及加权和 total。"""
    spatial_w = spatial_detail_weights(semantic, r_grid)
    l1 = F.l1_loss(recon, target)
    l2 = F.mse_loss(recon, target)

    zw = zero_level_weights(target, w.zero_sigma_mm, sdf_scale)
    zero_band = (zw * (recon - target).abs()).sum() / (zw.sum() + 1e-6)

    rim_mask = rim_region_mask(semantic, r_grid)
    rim_loss = weighted_l1(recon, target, rim_mask)

    grad_loss = torch.tensor(0.0, device=recon.device)
    if w.gradient > 0:
        grad_loss = sdf_gradient_loss(recon, target, rim_mask if rim_mask.numel() > 4 else spatial_w)

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

    total = (
        w.l1 * l1
        + w.l2 * l2
        + w.zero_level * zero_band
        + (w.rim * rim_loss if w.rim > 0 else 0.0)
        + w.gradient * grad_loss
        + w.dimension * dim_loss
        + w.semantic * sem
    )
    return {
        "total": total,
        "l1": l1,
        "l2": l2,
        "zero_band": zero_band,
        "rim": rim_loss,
        "gradient": grad_loss,
        "dimension": dim_loss,
        "semantic": sem,
    }


def physics_risk_proxy(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """物理风险代理：壁厚越小风险越高（用于扩散阶段的辅助损失）。"""
    return 1.0 / (estimate_min_thickness(sdf, sdf_scale) + 1e-3)
