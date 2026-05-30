"""自编码器复合损失：L1/L2、零等值带、物理尺寸与语义一致性。

除像素级重建外，强调 SDF 零附近精度、最小壁厚/面积与 GT 一致，
以及语义区域内外的符号一致性。
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
    zero_sigma_mm: float = 1.5  # 零等值带高斯权重宽度（mm）
    dimension: float = 0.3
    semantic: float = 0.2


def zero_level_weights(sdf: torch.Tensor, sigma_mm: float = 1.5, sdf_scale: float = 12.0) -> torch.Tensor:
    """在 SDF 零等值面附近赋予更高权重的空间掩码。

    使用 exp(-|sdf·scale|/σ) 形式，使零附近像素在损失中占更大比重。
    """
    return torch.exp(-(sdf * sdf_scale).abs() / sigma_mm)


def estimate_min_thickness(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """从重建 SDF 估计最小壁厚（mm）：内部区域沿径向的最大深度。"""
    inside = (sdf < 0).float()
    return inside.sum(dim=-1).amax(dim=-1) * (2 * sdf_scale / sdf.shape[-1])


def estimate_area(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    """从重建 SDF 估计实体截面积（mm²）：内部像素数 × 单元格面积。"""
    cell = (2 * sdf_scale / sdf.shape[-1]) ** 2
    return (sdf < 0).float().sum(dim=(-2, -1)) * cell


def semantic_consistency_loss(recon: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
    """语义区域内外符号一致性：各语义带内「内部」像素覆盖率应接近 1。

    将语义掩码分为三个区域带，分别计算 recon<0 的覆盖率与 1 的偏差。
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
    """将 batch 物理摘要统一为 (min_thickness, area) 张量对。"""
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
    """计算自编码器训练所需的全部损失项及加权和。

    Returns:
        包含 total、l1、l2、zero_band、dimension、semantic 的字典。
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
    """物理风险代理：壁厚越小风险越高（1/thickness）。

    用于扩散训练后期的物理引导损失，使生成结果与 GT 的壁厚风险分布一致。
    """
    return 1.0 / (estimate_min_thickness(sdf, sdf_scale) + 1e-3)
