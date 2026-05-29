"""PCA-UNet 自编码器：将子午面 SDF（及可选语义通道）编码为潜空间 z_m。

编码器通过多次池化将 256×256 输入压缩为 ``latent_channels × latent_spatial`` 的
特征图；解码器通过双线性上采样还原为单通道 SDF 重建。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """双卷积 + GroupNorm + SiLU 的基础卷积块。"""

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
    """子午面 SDF 自编码器：sdf2d (+ semantic_mask) → z_m → sdf2d_recon。

    Args:
        in_channels: 输入通道数（1=仅 SDF，2=SDF+语义）。
        latent_channels: 潜空间通道数 C。
        latent_spatial: 潜空间空间分辨率 H=W。
    """

    def __init__(self, in_channels: int = 2, latent_channels: int = 64, latent_spatial: int = 16) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial

        # 编码器：256 → 128 → 64 → 32 → 16
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
        # 解码器：16 → 32 → 64 → 128 → 256
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
        """将输入图像编码为潜特征图 z_m。"""
        return self.encoder(x)

    def decode(self, z_m: torch.Tensor) -> torch.Tensor:
        """将潜特征图解码为重建 SDF。"""
        return self.decoder(z_m)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向：编码后解码，返回 (z_m, recon)。"""
        z_m = self.encode(x)
        return z_m, self.decode(z_m)
