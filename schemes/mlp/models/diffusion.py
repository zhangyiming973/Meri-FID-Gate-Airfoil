"""扩散过程核心：噪声调度、UNet 去噪器（遗留）与 DDPM/DDIM 采样。

MLP 方案主要使用 DiffusionSchedule + MLPDenoiser；本模块中的 LatentDiffusionUNet
为空间 UNet 备选实现，train_diffusion 默认走 MLP 路径。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentStats:
    """潜向量全局均值/标准差，用于简单归一化（PCA 方案中较少使用）。"""

    mean: float
    std: float

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean) / self.std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatentStats:
        return cls(mean=float(data["mean"]), std=float(data["std"]))

    @classmethod
    def from_latents(cls, z: torch.Tensor) -> LatentStats:
        return cls(mean=float(z.mean()), std=max(float(z.std()), 1e-6))


def save_latent_stats(stats: LatentStats, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)


def load_latent_stats(path: Path) -> LatentStats:
    with open(path, encoding="utf-8") as f:
        return LatentStats.from_dict(json.load(f))


class SinusoidalPosEmb(nn.Module):
    """扩散时间步的正弦位置编码。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """带时间/条件嵌入的残差卷积块（AdaGN 式 scale-shift）。"""

    def __init__(self, ch: int, emb_dim: int) -> None:
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, ch * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=-1)
        h = self.n2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.c2(F.silu(h))


class LatentDiffusionUNet(nn.Module):
    """潜空间 UNet 去噪器：条件在空间维广播后与带噪潜码拼接。

    用于在完整 (C,H,W) 潜张量上直接扩散；MLP 方案改用 PCA+MLP。
    """

    def __init__(self, latent_channels: int = 64, cond_dim: int = 5, time_dim: int = 128, base_ch: int = 128) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.cond_dim = cond_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        emb_dim = time_dim * 2
        in_ch = latent_channels + cond_dim
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.rb1 = ResBlock(base_ch, emb_dim)
        self.rb2 = ResBlock(base_ch, emb_dim)
        self.mid = ResBlock(base_ch, emb_dim)
        self.rb3 = ResBlock(base_ch, emb_dim)
        self.rb4 = ResBlock(base_ch, emb_dim)
        self.out_conv = nn.Conv2d(base_ch, latent_channels, 3, padding=1)

    def _apply_cond_mask(self, condition: torch.Tensor, cond_mask: torch.Tensor | None) -> torch.Tensor:
        """CFG 训练/推理时按 mask 置零条件。"""
        if cond_mask is None:
            return condition
        if cond_mask.ndim == 1:
            return condition * cond_mask.unsqueeze(-1)
        return condition * cond_mask

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self._apply_cond_mask(condition, cond_mask)
        emb = torch.cat([self.time_mlp(t), self.cond_mlp(condition)], dim=-1)
        b, _, h, w = x.shape
        # 条件向量扩展为与潜特征同空间尺寸的特征图
        c_map = condition.unsqueeze(-1).unsqueeze(-1).expand(b, self.cond_dim, h, w)
        h0 = self.in_conv(torch.cat([x, c_map], dim=1))
        h0 = self.rb1(h0, emb)
        h0 = F.avg_pool2d(h0, 2)
        h0 = self.rb2(h0, emb)
        h0 = F.interpolate(h0, scale_factor=2, mode="nearest")
        h0 = self.mid(h0, emb)
        h0 = self.rb3(h0, emb)
        h0 = self.rb4(h0, emb)
        return self.out_conv(h0)


class DiffusionSchedule:
    """DDPM 噪声调度与采样：q 过程加噪、p/DDIM 反演，支持 classifier-free guidance。"""

    def __init__(self, timesteps: int = 200, device: torch.device | None = None) -> None:
        self.timesteps = timesteps
        # 线性 beta 调度
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        if device:
            self.to(device)

    def to(self, device: torch.device) -> DiffusionSchedule:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def _ab_broadcast(self, ab: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """将 per-batch 的 alpha_bar 广播到与 x 同秩。"""
        return ab.view(-1, *([1] * (x.ndim - 1)))

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
        """DDIM 加速采样（默认 eta=0 确定性），支持 CFG。"""
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
