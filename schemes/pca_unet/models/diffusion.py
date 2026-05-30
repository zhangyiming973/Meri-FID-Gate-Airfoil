"""扩散模型核心：DDPM 噪声调度与 DDIM/DDPM 采样（含 CFG）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionSchedule:
    """DDPM 噪声调度与采样器。

    维护 β、α、ᾱ 序列，提供前向加噪 ``q_sample``、x0 预测及
    DDPM/DDIM + CFG 反向采样。
    """

    def __init__(self, timesteps: int = 200, device: torch.device | None = None) -> None:
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        if device:
            self.to(device)

    def to(self, device: torch.device) -> DiffusionSchedule:
        """将调度张量迁移到指定设备。"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def _ab_broadcast(self, ab: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """将 ᾱ_t 广播为与 x 可逐元素运算的形状。"""
        return ab.view(-1, *([1] * (x.ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """前向扩散：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε。"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self._ab_broadcast(self.alpha_bar[t], x0)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """由 x_t 与预测噪声 ε 反推 x_0（用于物理引导等）。"""
        ab = self._ab_broadcast(self.alpha_bar[t], xt)
        return (xt - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab).clamp(min=1e-6)

    @torch.no_grad()
    def p_sample_cfg(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: int,
        condition: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """单步 DDPM 采样，带 classifier-free guidance。"""
        tt = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        eps_c = model(xt, tt, condition)
        eps_u = model(xt, tt, condition, cond_mask=torch.zeros(xt.shape[0], device=xt.device))
        eps = eps_u + cfg_scale * (eps_c - eps_u)
        beta, alpha, ab = self.betas[t], self.alphas[t], self.alpha_bar[t]
        mean = (1 / torch.sqrt(alpha)) * (xt - (beta / torch.sqrt(1 - ab)) * eps)
        if t > 0:
            return mean + torch.sqrt(beta) * torch.randn_like(xt)
        return mean

    @torch.no_grad()
    def ddim_sample_cfg(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        condition: torch.Tensor,
        cfg_scale: float = 1.5,
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM 加速采样（η=0 为确定性），带 CFG。"""
        device = condition.device
        xt = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for i, t in enumerate(times):
            tt = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            eps_c = model(xt, tt, condition)
            eps_u = model(xt, tt, condition, cond_mask=torch.zeros(shape[0], device=device))
            eps = eps_u + cfg_scale * (eps_c - eps_u)

            ab_t = self.alpha_bar[t]
            x0 = (xt - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t).clamp(min=1e-6)

            if i == len(times) - 1:
                xt = x0
                break

            t_next = int(times[i + 1])
            ab_next = self.alpha_bar[t_next]
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next))
            dir_xt = torch.sqrt(ab_next) * x0
            noise = torch.randn_like(xt) if eta > 0 else 0.0
            xt = dir_xt + torch.sqrt(1 - ab_next - sigma**2) * eps + sigma * noise
        return xt

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        condition: torch.Tensor,
        cfg_scale: float = 1.5,
        steps: int | None = None,
        use_ddim: bool = True,
    ) -> torch.Tensor:
        """统一采样入口：默认 DDIM，否则逐步 DDPM。"""
        if use_ddim:
            return self.ddim_sample_cfg(model, shape, condition, cfg_scale, steps or 50)
        xt = torch.randn(shape, device=condition.device)
        for t in range(self.timesteps - 1, -1, -1):
            xt = self.p_sample_cfg(model, xt, t, condition, cfg_scale)
        return xt
