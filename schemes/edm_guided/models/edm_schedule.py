"""EDM 扩散调度：预条件化、对数正态噪声训练与 Heun 二阶采样。

参考 Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*
(NeurIPS 2022) 与 *Analyzing and Improving the Training Dynamics of Diffusion Models* (CVPR 2024)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def edm_precond_constants(sigma: torch.Tensor, sigma_data: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EDM 预条件化系数 c_skip, c_out, c_in。"""
    sd2 = sigma_data**2
    sigma2 = sigma**2
    c_skip = sd2 / (sigma2 + sd2)
    c_out = sigma * sigma_data / torch.sqrt(sigma2 + sd2)
    c_in = 1.0 / torch.sqrt(sigma2 + sd2)
    return c_skip, c_out, c_in


def edm_c_noise(sigma: torch.Tensor) -> torch.Tensor:
    """噪声水平嵌入：c_noise(σ) = (1/4) ln(σ)。"""
    return 0.25 * torch.log(sigma.clamp(min=1e-8))


def edm_training_target(x0: torch.Tensor, n: torch.Tensor, sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """F_θ 的有效训练目标（Eq. 8）。"""
    c_skip, c_out, _ = edm_precond_constants(sigma, sigma_data)
    while c_skip.ndim < x0.ndim:
        c_skip = c_skip.unsqueeze(-1)
        c_out = c_out.unsqueeze(-1)
    y_noisy = x0 + n
    return (x0 - c_skip * y_noisy) / c_out.clamp(min=1e-8)


def sample_training_sigma(
    batch_size: int,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> torch.Tensor:
    """训练噪声水平：ln(σ) ~ N(P_mean, P_std²)。"""
    log_sigma = torch.randn(batch_size, device=device) * p_std + p_mean
    return log_sigma.exp().clamp(sigma_min, sigma_max)


def edm_sigma_schedule(num_steps: int, sigma_min: float, sigma_max: float, rho: float) -> torch.Tensor:
    """EDM 采样 σ 序列（Eq. 5），末步 σ=0。"""
    ramp = torch.linspace(0, 1, num_steps, dtype=torch.float64)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, torch.zeros(1)]).float()


def resolve_latent_edm_config(edm_cfg: dict, sigma_data: float) -> dict:
    """为 PCA 潜空间自动适配 EDM 噪声范围。

    像素空间 EDM 默认 sigma_max=80（sigma_data≈0.5）；潜向量经 PCA 归一化后 std≈1，
    若仍用 sigma_max=80 会在过高噪声区间浪费采样步数并导致轮缘等细节被平滑。
    """
    resolved = dict(edm_cfg)
    resolved["sigma_data"] = float(resolved.get("sigma_data") or sigma_data)
    sd = resolved["sigma_data"]
    if resolved.get("sigma_min") is None:
        resolved["sigma_min"] = 0.002
    if resolved.get("sigma_max") is None:
        # 潜空间有效噪声上界 ≈ 16·σ_data（像素 EDM 比例 80/0.5=160 的 1/10）
        resolved["sigma_max"] = float(min(80.0, max(10.0, 16.0 * sd)))
    if resolved.get("p_mean") is None:
        resolved["p_mean"] = -1.2
    if resolved.get("p_std") is None:
        resolved["p_std"] = 1.2
    if resolved.get("rho") is None:
        resolved["rho"] = 7.0
    return resolved


class EDMDiffusionSchedule:
    """EDM 预条件化扩散：训练损失 + Heun ODE 采样（含 CFG）。"""

    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        rho: float = 7.0,
        device: torch.device | None = None,
    ) -> None:
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std
        self.rho = rho
        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> EDMDiffusionSchedule:
        return self

    def add_noise(self, x0: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor | None = None):
        """前向加噪：x = x0 + σ·ε。"""
        if noise is None:
            noise = torch.randn_like(x0)
        while sigma.ndim < x0.ndim:
            sigma = sigma.unsqueeze(-1)
        return x0 + sigma * noise, noise

    def wrap_denoiser(
        self,
        net: nn.Module,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """D_θ(x;σ) = c_skip·x + c_out·F_θ(c_in·x; c_noise)。"""
        c_skip, c_out, c_in = edm_precond_constants(sigma, self.sigma_data)
        c_noise = edm_c_noise(sigma)
        while c_skip.ndim < x.ndim:
            c_skip = c_skip.unsqueeze(-1)
            c_out = c_out.unsqueeze(-1)
            c_in = c_in.unsqueeze(-1)
        f_out = net(c_in * x, c_noise, condition, cond_mask=cond_mask)
        return c_skip * x + c_out * f_out

    def training_loss(
        self,
        net: nn.Module,
        x0: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """EDM 加权去噪损失（λ=1/c_out² 与预条件化合并后形式）。"""
        b = x0.shape[0]
        device = x0.device
        sigma = sample_training_sigma(b, device, self.sigma_min, self.sigma_max, self.p_mean, self.p_std)
        x_noisy, noise = self.add_noise(x0, sigma, None)
        c_noise = edm_c_noise(sigma)
        _, c_out, c_in = edm_precond_constants(sigma, self.sigma_data)
        while c_in.ndim < x0.ndim:
            c_in = c_in.unsqueeze(-1)
        target = edm_training_target(x0, noise, sigma, self.sigma_data)
        pred = net(c_in * x_noisy, c_noise, condition, cond_mask=cond_mask)
        return torch.nn.functional.mse_loss(pred, target)

    @torch.no_grad()
    def _denoise_cfg(
        self,
        net: nn.Module,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Classifier-free guidance on denoiser D_θ。"""
        d_c = self.wrap_denoiser(net, x, sigma, condition)
        d_u = self.wrap_denoiser(
            net,
            x,
            sigma,
            condition,
            cond_mask=torch.zeros(x.shape[0], device=x.device),
        )
        return d_u + cfg_scale * (d_c - d_u)

    @torch.no_grad()
    def heun_sample_cfg(
        self,
        net: nn.Module,
        shape: tuple[int, ...],
        condition: torch.Tensor,
        cfg_scale: float = 1.5,
        steps: int = 35,
    ) -> torch.Tensor:
        """Algorithm 1：Heun 二阶 ODE 采样，σ(t)=t, s(t)=1。"""
        device = condition.device
        sigmas = edm_sigma_schedule(steps, self.sigma_min, self.sigma_max, self.rho).to(device)
        xt = torch.randn(shape, device=device) * sigmas[0]

        for i in range(len(sigmas) - 1):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            t_cur = sigma_cur
            t_next = sigma_next
            dt = t_next - t_cur

            sigma_b = sigma_cur.expand(shape[0])
            d_cur = (xt - self._denoise_cfg(net, xt, sigma_b, condition, cfg_scale)) / t_cur.clamp(min=1e-8)
            x_euler = xt + dt * d_cur

            if sigma_next.item() == 0.0:
                xt = self._denoise_cfg(net, x_euler, sigma_b, condition, cfg_scale)
                break

            sigma_next_b = sigma_next.expand(shape[0])
            d_next = (x_euler - self._denoise_cfg(net, x_euler, sigma_next_b, condition, cfg_scale)) / t_next.clamp(
                min=1e-8
            )
            xt = xt + dt * 0.5 * (d_cur + d_next)

        return xt

    @torch.no_grad()
    def sample(
        self,
        net: nn.Module,
        shape: tuple[int, ...],
        condition: torch.Tensor,
        cfg_scale: float = 1.5,
        steps: int = 35,
    ) -> torch.Tensor:
        return self.heun_sample_cfg(net, shape, condition, cfg_scale, steps)

    @staticmethod
    def estimate_sigma_data(encoded: torch.Tensor) -> float:
        """从 PCA 编码训练集估计 σ_data。"""
        return float(encoded.std().clamp(min=1e-4))
