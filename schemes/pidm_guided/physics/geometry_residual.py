"""子午面 SDF 几何物理残差，对应 PIDM 中 PDE/FEM 残差的角色。

残差目标为 0（独立物理约束），而非对齐 GT 样本的代理量。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from schemes.pidm_guided.losses.ae_losses import estimate_area, estimate_min_thickness


def eikonal_residual(sdf: torch.Tensor) -> torch.Tensor:
    """SDF Eikonal 条件 |∇SDF| ≈ 1 的有限差分残差（逐样本标量）。"""
    dx = sdf[:, :, :, 1:] - sdf[:, :, :, :-1]
    dy = sdf[:, :, 1:, :] - sdf[:, :, :-1, :]
    dx_c = dx[:, :, :-1, :]
    dy_c = dy[:, :, :, :-1]
    grad_norm = torch.sqrt(dx_c.pow(2) + dy_c.pow(2) + 1e-6)
    return (grad_norm - 1.0).abs().mean(dim=(-2, -1))


@dataclass
class GeometryResidualComputer:
    """从解码 SDF 计算几何物理残差向量。"""

    min_thickness_mm: float = 2.0
    use_area_constraint: bool = False
    use_eikonal: bool = True
    eikonal_weight: float = 0.1
    sdf_scale: float = 12.0

    def __call__(
        self,
        sdf: torch.Tensor,
        area_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算残差 (B, K)，各分量目标趋近 0。"""
        if sdf.ndim == 3:
            sdf = sdf.unsqueeze(1)

        thick = estimate_min_thickness(sdf, self.sdf_scale).reshape(sdf.shape[0], -1).mean(dim=1)
        parts = [F.relu(self.min_thickness_mm - thick).unsqueeze(-1)]

        if self.use_area_constraint and area_target is not None:
            est_area = estimate_area(sdf, self.sdf_scale).reshape(sdf.shape[0], -1).mean(dim=1)
            if area_target.ndim == 0:
                area_target = area_target.unsqueeze(0)
            parts.append((est_area - area_target.reshape(-1)).abs().unsqueeze(-1))

        if self.use_eikonal:
            eik = eikonal_residual(sdf).reshape(sdf.shape[0], -1).mean(dim=1)
            parts.append((self.eikonal_weight * eik).unsqueeze(-1))

        return torch.cat(parts, dim=-1)

    def mean_abs(self, residual: torch.Tensor) -> torch.Tensor:
        """残差绝对值均值，用于监控。"""
        return residual.abs().mean()
