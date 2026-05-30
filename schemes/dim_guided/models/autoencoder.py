"""子午线 2D SDF 卷积自编码器：编码/解码潜空间 z_m。

解码器使用 ConvTranspose2d 逐级上采样（替代双线性插值），以保留轮缘等外缘细节。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """双卷积 + GroupNorm + SiLU 的基础块。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        g = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """转置卷积上采样 + 卷积细化，比 bilinear Upsample 更利于锐化边界。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class MeridianAutoEncoder(nn.Module):
    """子午线 SDF 自编码器：sdf2d (+ semantic) -> z_m -> sdf2d_recon。

    默认潜空间：latent_channels × latent_spatial × latent_spatial（如 64×16×16）。
    """

    def __init__(self, in_channels: int = 2, latent_channels: int = 64, latent_spatial: int = 16) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial

        # 256 -> 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, latent_channels, 1),
        )
        # 16 -> 32 -> 64 -> 128 -> 256（转置卷积，保留高频）
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 1),
            UpBlock(128, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            UpBlock(32, 16),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, C_in, H, W) -> 潜码 (B, latent_channels, latent_spatial, latent_spatial)。"""
        return self.encoder(x)

    def decode(self, z_m: torch.Tensor) -> torch.Tensor:
        """潜码 -> 重建 SDF (B, 1, 256, 256)。"""
        return self.decoder(z_m)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """编码并解码，返回 (z_m, recon)。"""
        z_m = self.encode(x)
        return z_m, self.decode(z_m)
