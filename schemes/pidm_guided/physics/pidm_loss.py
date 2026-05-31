"""PIDM 虚拟似然损失（ICLR 2025 Bastek et al.）。

参考 PhysicsInformedDiffusionModels/src/denoising_utils.py::gaussian_log_likelihood
与 model_estimation_loss 中的残差项。
"""
from __future__ import annotations

import torch


def gaussian_log_likelihood(x: torch.Tensor, means: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """高斯对数似然 log p(x | mean, var)，与 PIDM 实现一致。"""
    centered = x - means
    return -0.5 * (centered.pow(2) / variance)


def pidm_virtual_likelihood_loss(
    residual: torch.Tensor,
    posterior_variance: torch.Tensor,
    c_residual: float,
) -> torch.Tensor:
    """PIDM 残差虚拟似然：-c_residual * log p(r=0 | x0_pred, var_t)。

    Args:
        residual: 物理残差 (B, K) 或 (B,)。
        posterior_variance: 与 t 绑定的后验方差，形状 (B,) 或 (B, 1, ...)。
        c_residual: PIDM 配置中的 c_residual 缩放因子。
    """
    if c_residual <= 0:
        return residual.new_zeros(())

    var = posterior_variance
    while var.ndim < residual.ndim:
        var = var.unsqueeze(-1)
    if var.shape[-1] == 1 and residual.shape[-1] > 1:
        var = var.expand_as(residual)

    log_lik = gaussian_log_likelihood(torch.zeros_like(residual), means=residual, variance=var.clamp(min=1e-8))
    return c_residual * (-log_lik).mean()
