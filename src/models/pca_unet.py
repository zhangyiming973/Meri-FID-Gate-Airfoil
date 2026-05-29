from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PcaGridSpec:
    """Reshape PCA vector into a 2D grid for UNet diffusion."""

    channels: int
    height: int
    width: int

    @property
    def dim(self) -> int:
        return self.channels * self.height * self.width

    @classmethod
    def from_dim(cls, dim: int) -> PcaGridSpec:
        best: PcaGridSpec | None = None
        best_score = float("inf")
        for c in (1, 2, 4, 8):
            if dim % c != 0:
                continue
            spatial = dim // c
            for h in range(1, int(math.sqrt(spatial)) + 1):
                if spatial % h != 0:
                    continue
                w = spatial // h
                aspect = max(h, w) / max(min(h, w), 1)
                score = aspect + abs(h - w) * 0.1
                if score < best_score:
                    best_score = score
                    best = cls(channels=c, height=h, width=w)
        if best is None:
            raise ValueError(f"cannot factor pca_dim={dim} into C×H×W grid")
        return best

    def vec_to_grid(self, v: torch.Tensor) -> torch.Tensor:
        return v.view(v.shape[0], self.channels, self.height, self.width)

    def grid_to_vec(self, g: torch.Tensor) -> torch.Tensor:
        return g.reshape(g.shape[0], -1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, ch: int, emb_dim: int) -> None:
        super().__init__()
        g = min(8, ch)
        self.n1 = nn.GroupNorm(g, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(g, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, ch * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=-1)
        h = self.n2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.c2(F.silu(h))


class ConditionalUNet(nn.Module):
    """Encoder-decoder UNet with skip connections for PCA-grid latent diffusion."""

    def __init__(
        self,
        in_channels: int,
        cond_dim: int = 5,
        base_ch: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
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
        in_ch = in_channels + cond_dim

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.rb1 = ResBlock(base_ch, emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.rb2 = ResBlock(base_ch * 2, emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.rb_mid1 = ResBlock(base_ch * 4, emb_dim)
        self.rb_mid2 = ResBlock(base_ch * 4, emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.rb_dec2 = ResBlock(base_ch * 2, emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch, 4, stride=2, padding=1)
        self.rb_dec1 = ResBlock(base_ch, emb_dim)
        self.out_conv = nn.Conv2d(base_ch * 2, in_channels, 3, padding=1)

    def _apply_cond_mask(self, condition: torch.Tensor, cond_mask: torch.Tensor | None) -> torch.Tensor:
        if cond_mask is None:
            return condition
        if cond_mask.ndim == 1:
            return condition * cond_mask.unsqueeze(-1)
        return condition * cond_mask

    def _cond_map(self, condition: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b = condition.shape[0]
        return condition.unsqueeze(-1).unsqueeze(-1).expand(b, self.cond_dim, h, w)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self._apply_cond_mask(condition, cond_mask)
        emb = torch.cat([self.time_mlp(t), self.cond_mlp(condition)], dim=-1)

        _, _, h, w = x.shape
        x_in = torch.cat([x, self._cond_map(condition, h, w)], dim=1)

        e1 = self.rb1(self.in_conv(x_in), emb)
        e2 = self.rb2(self.down1(e1), emb)
        e3 = self.rb_mid1(self.down2(e2), emb)
        m = self.rb_mid2(e3, emb)

        d3 = self.rb_dec2(self.up2(m), emb)
        d2 = self.rb_dec1(self.up1(torch.cat([d3, e2], dim=1)), emb)
        d1 = self.out_conv(torch.cat([d2, e1], dim=1))
        return d1


def build_pca_unet(pca_dim: int, cond_dim: int, base_ch: int = 64) -> tuple[ConditionalUNet, PcaGridSpec]:
    spec = PcaGridSpec.from_dim(pca_dim)
    if spec.dim != pca_dim:
        raise ValueError(f"grid spec dim {spec.dim} != pca_dim {pca_dim}")
    model = ConditionalUNet(spec.channels, cond_dim, base_ch=base_ch)
    return model, spec
