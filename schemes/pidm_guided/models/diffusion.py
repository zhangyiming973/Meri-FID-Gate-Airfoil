"""扩散过程核心：DDPM 噪声调度与 DDIM/DDPM 采样（含 CFG）。

扩展 PIDM 所需的后验方差 ``posterior_variance_clipped``，以及可选的
推理时 x0 残差梯度校正（N_correction / M_correction）。
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSchedule:
    """DDPM 噪声调度与采样：q 过程加噪、p/DDIM 反演，支持 classifier-free guidance。"""

    def __init__(self, timesteps: int = 200, device: torch.device | None = None) -> None:
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

        alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_variance = posterior_variance
        self.posterior_variance_clipped = posterior_variance.clone()
        self.posterior_variance_clipped[0] = posterior_variance[1]

        if device:
            self.to(device)

    def to(self, device: torch.device) -> DiffusionSchedule:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_variance_clipped = self.posterior_variance_clipped.to(device)
        return self

    def _ab_broadcast(self, ab: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """将 per-batch 的 alpha_bar 广播到与 x 同秩。"""
        return ab.view(-1, *([1] * (x.ndim - 1)))

    def posterior_var_at(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """提取与 batch 时间步 t 对应的后验方差，广播到 x 的 batch 维。"""
        return self._ab_broadcast(self.posterior_variance_clipped[t], x)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """前向扩散：x_t = sqrt(ab_t)*x0 + sqrt(1-ab_t)*noise。"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self._ab_broadcast(self.alpha_bar[t], x0)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """由 x_t 与预测噪声 ε 反推 x0。"""
        ab = self._ab_broadcast(self.alpha_bar[t], xt)
        return (xt - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab).clamp(min=1e-6)

    def correct_x0(
        self,
        x0: torch.Tensor,
        residual_fn: Callable[[torch.Tensor], torch.Tensor],
        steps: int = 1,
        step_size: float = 0.05,
    ) -> torch.Tensor:
        """PIDM/CoCoGen 式 x0 残差梯度校正（在潜空间系数上操作）。"""
        if steps <= 0:
            return x0
        corrected = x0
        for _ in range(steps):
            corrected = corrected.detach().requires_grad_(True)
            res = residual_fn(corrected)
            grad = torch.autograd.grad(res.sum(), corrected, retain_graph=False)[0]
            corrected = (corrected - step_size * grad).detach()
        return corrected

    @torch.no_grad()
    def p_sample_cfg(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: int,
        condition: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """单步 DDPM 采样，带 CFG：eps = eps_u + scale * (eps_c - eps_u)。"""
        tt = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        eps_c = model(xt, tt, condition)
        eps_u = model(xt, tt, condition, cond_mask=torch.zeros(xt.shape[0], device=xt.device))
        eps = eps_u + cfg_scale * (eps_c - eps_u)
        beta, alpha, ab = self.betas[t], self.alphas[t], self.alpha_bar[t]
        mean = (1 / torch.sqrt(alpha)) * (xt - (beta / torch.sqrt(1 - ab)) * eps)
        if t > 0:
            return mean + torch.sqrt(beta) * torch.randn_like(xt)
        return mean

    def ddim_sample_cfg(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        condition: torch.Tensor,
        cfg_scale: float = 1.5,
        steps: int = 50,
        eta: float = 0.0,
        x0_corrector: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        n_correction: int = 0,
        m_correction: int = 0,
    ) -> torch.Tensor:
        """DDIM 加速采样（默认 eta=0 确定性），支持 CFG 与 x0 残差校正。"""
        device = condition.device
        xt = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        use_correction = x0_corrector is not None and (n_correction > 0 or m_correction > 0)

        for i, t in enumerate(times):
            tt = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            with torch.no_grad():
                eps_c = model(xt, tt, condition)
                eps_u = model(xt, tt, condition, cond_mask=torch.zeros(shape[0], device=device))
                eps = eps_u + cfg_scale * (eps_c - eps_u)
                x0 = self.predict_x0(xt, tt, eps)
            if x0_corrector is not None and n_correction > 0:
                x0 = x0_corrector(x0, tt)
            if x0_corrector is not None and m_correction > 0 and i == len(times) - 1:
                x0 = x0_corrector(x0, tt)

            if i == len(times) - 1:
                xt = x0.detach() if use_correction else x0
                break

            t_next = int(times[i + 1])
            ab_t = self.alpha_bar[t]
            ab_next = self.alpha_bar[t_next]
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next))
            with torch.no_grad():
                dir_xt = torch.sqrt(ab_next) * x0
                noise = torch.randn_like(xt) if eta > 0 else 0.0
                xt = dir_xt + torch.sqrt(1 - ab_next - sigma**2) * eps + sigma * noise
        return xt

    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        condition: torch.Tensor,
        cfg_scale: float = 1.5,
        steps: int | None = None,
        use_ddim: bool = True,
        x0_corrector: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        n_correction: int = 0,
        m_correction: int = 0,
    ) -> torch.Tensor:
        """统一采样入口：默认 DDIM，否则逐步 DDPM。"""
        if use_ddim:
            return self.ddim_sample_cfg(
                model,
                shape,
                condition,
                cfg_scale,
                steps or 50,
                x0_corrector=x0_corrector,
                n_correction=n_correction,
                m_correction=m_correction,
            )
        with torch.no_grad():
            xt = torch.randn(shape, device=condition.device)
            for t in range(self.timesteps - 1, -1, -1):
                xt = self.p_sample_cfg(model, xt, t, condition, cfg_scale)
        return xt
