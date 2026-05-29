from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AELossWeights:
    l1: float = 1.0
    l2: float = 0.5
    zero_level: float = 2.0
    zero_sigma_mm: float = 1.5
    dimension: float = 0.3
    semantic: float = 0.2


def zero_level_weights(sdf: torch.Tensor, sigma_mm: float = 1.5, sdf_scale: float = 12.0) -> torch.Tensor:
    return torch.exp(-(sdf * sdf_scale).abs() / sigma_mm)


def estimate_min_thickness(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    inside = (sdf < 0).float()
    return inside.sum(dim=-1).amax(dim=-1) * (2 * sdf_scale / sdf.shape[-1])


def estimate_area(sdf: torch.Tensor, sdf_scale: float = 12.0) -> torch.Tensor:
    cell = (2 * sdf_scale / sdf.shape[-1]) ** 2
    return (sdf < 0).float().sum(dim=(-2, -1)) * cell


def semantic_consistency_loss(recon: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
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
) -> dict[str, torch.Tensor]:
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
    return 1.0 / (estimate_min_thickness(sdf, sdf_scale) + 1e-3)
