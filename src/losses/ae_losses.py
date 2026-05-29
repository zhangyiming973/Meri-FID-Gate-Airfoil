"""自编码器（AE）损失函数。

除基础 L1/L2 重建外，还包含零等值面加权、物理尺寸约束与语义一致性损失。
physics_risk_proxy 供扩散训练中的物理引导（physics guidance）使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AELossWeights:
    """各损失项的加权系数。

    Attributes:
        l1: 全局 L1 重建权重。
        l2: MSE 重建权重。
        zero_level: 零等值面附近加权 L1 权重。
        zero_sigma_mm: 零等值面高斯权重的 sigma（mm 尺度）。
        dimension: 物理尺寸（最小厚度/面积）误差权重。
        semantic: 语义区域覆盖率一致性权重。
    """

    l1: float = 1.0
    l2: float = 0.5
    zero_level: float = 2.0
    zero_sigma_mm: float = 1.5
    dimension: float = 0.3
    semantic: float = 0.2


def zero_level_weights(sdf: torch.Tensor, sigma_mm: float = 1.5, sdf_scale: float = 12.0) -> torch.Tensor:
    """计算零等值面附近的空间权重图。

    在 SDF≈0 的区域赋予更高权重，使模型更关注轮廓边界精度。

    Args:
        sdf: 归一化 SDF 张量。
        sigma_mm: 高斯衰减的 sigma（物理 mm 单位）。
        sdf_scale: SDF 归一化到物理尺寸的缩放因子。

    Returns:
        与 sdf 同形状的权重张量，零等值面处接近 1。
    """
    return torch.exp(-(sdf * sdf_scale).abs() / sigma_mm)


def estimate_min_thickness(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """从重建 SDF 估计最小壁厚（mm）。

    沿径向取实体（sdf<0）区域的最大连续宽度作为厚度代理。
    """
    inside = (sdf < 0).float()
    return inside.sum(dim=-1).amax(dim=-1) * (2 * sdf_scale / sdf.shape[-1])


def estimate_area(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """从重建 SDF 估计截面实体面积（mm²）。"""
    cell = (2 * sdf_scale / sdf.shape[-1]) ** 2
    return (sdf < 0).float().sum(dim=(-2, -1)) * cell


def semantic_consistency_loss(recon: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
    """语义区域覆盖率一致性损失。

    将语义 mask 划分为 hub / web / rim 三个区间，
    要求重建 SDF 的实体区域在各语义区间内均有足够覆盖。
    """
    inside = (recon < 0).float()
    # 三个语义区间的 mask 阈值划分
    regions = [
        (semantic > 0.1) & (semantic < 0.4),   # hub 区域
        (semantic > 0.4) & (semantic < 0.8),   # web 区域
        semantic > 0.8,                          # rim 区域
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
    """将 batch 物理摘要转为 (min_thickness, area) 张量对。

    支持 DataLoader 拼成的 dict-of-tensors 或 list-of-dicts 两种格式。
    """
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
    """计算自编码器综合损失及各分项。

    Args:
        recon: 重建 SDF，形状 (B, 1, H, W)。
        target: 目标 SDF。
        semantic: 语义 mask，可为 None（关闭语义损失）。
        physics: GT 物理摘要，含 min_thickness_mm / area_mm2。
        w: 损失权重配置。
        sdf_scale: SDF 物理尺度因子。

    Returns:
        含 total、l1、l2、zero_band、dimension、semantic 的字典。
    """
    l1 = F.l1_loss(recon, target)
    l2 = F.mse_loss(recon, target)
    zw = zero_level_weights(target, w.zero_sigma_mm, sdf_scale)
    zero_band = (zw * (recon - target).abs()).sum() / (zw.sum() + 1e-6)

    dim_loss = torch.tensor(0.0, device=recon.device)
    pt = _physics_tensors(physics, recon.device, recon.dtype)
    if pt is not None:
        gt_t, gt_a = pt
        dim_loss = F.l1_loss(estimate_min_thickness(recon, sdf_scale).squeeze(-1), gt_t.squeeze(-1)) + \
                   F.l1_loss(estimate_area(recon, sdf_scale).squeeze(-1), gt_a.squeeze(-1)) / 10000.0

    sem = torch.tensor(0.0, device=recon.device)
    if semantic is not None and w.semantic > 0:
        sem = semantic_consistency_loss(recon, semantic)

    total = w.l1 * l1 + w.l2 * l2 + w.zero_level * zero_band + w.dimension * dim_loss + w.semantic * sem
    return {"total": total, "l1": l1, "l2": l2, "zero_band": zero_band, "dimension": dim_loss, "semantic": sem}


def physics_risk_proxy(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """物理风险代理：壁厚越薄，风险值越高。

    用于扩散训练后期的 physics guidance，约束生成潜变量解码后的几何合理性。
    """
    return 1.0 / (estimate_min_thickness(sdf, sdf_scale) + 1e-3)
