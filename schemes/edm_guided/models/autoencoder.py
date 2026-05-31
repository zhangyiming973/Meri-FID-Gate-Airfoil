"""子午线 2D SDF 卷积自编码器：编码/解码潜空间 z_m。

编码器将 sdf2d（可选拼接 semantic_mask）下采样至固定空间分辨率的潜特征；
解码器上采样回 256×256 单通道 SDF。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 1),
            nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False),
            ConvBlock(128, 128),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
            nn.Upsample(size=(128, 128), mode="bilinear", align_corners=False),
            ConvBlock(64, 32),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
            ConvBlock(32, 16),
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
