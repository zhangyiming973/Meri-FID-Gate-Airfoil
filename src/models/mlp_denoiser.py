from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLPDenoiser(nn.Module):
    """Denoiser for PCA-compressed latent vector + condition."""

    def __init__(self, latent_dim: int, cond_dim: int = 5, hidden: int = 512, time_dim: int = 128) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond_mask is not None:
            if cond_mask.ndim == 1:
                condition = condition * cond_mask.unsqueeze(-1)
            else:
                condition = condition * cond_mask
        t_emb = self.time_mlp(t)
        h = torch.cat([x, condition, t_emb], dim=-1)
        return self.net(h)
