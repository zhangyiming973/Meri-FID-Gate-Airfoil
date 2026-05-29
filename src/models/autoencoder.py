"""Meridian SDF 卷积自编码器。

将 2D 符号距离场（SDF，可选语义 mask 通道）编码为低分辨率潜变量 z_m，
再解码回 256×256 SDF 重建。潜变量供后续 PCA + 扩散建模。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """双卷积 + GroupNorm + SiLU 的基础卷积块。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        g = min(8, out_ch)  # GroupNorm 组数不超过通道数
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
    """Meridian 2D SDF 自编码器。

    编码器：256→128→64→32→16 四级下采样，输出 (latent_channels, latent_spatial, latent_spatial)。
    解码器：对称上采样回 256×256 单通道 SDF。

    Args:
        in_channels: 输入通道数，1=仅 SDF，2=SDF+语义 mask。
        latent_channels: 潜变量通道数 C。
        latent_spatial: 潜变量空间分辨率 H=W。
    """

    def __init__(self, in_channels: int = 2, latent_channels: int = 64, latent_spatial: int = 16) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial

        # 编码器：4 次 MaxPool 将 256 降至 16
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
        # 解码器：双线性上采样 + 卷积块，逐步恢复分辨率
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
        """编码输入为潜变量 z_m。"""
        return self.encoder(x)

    def decode(self, z_m: torch.Tensor) -> torch.Tensor:
        """从潜变量解码重建 SDF。"""
        return self.decoder(z_m)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向：返回 (z_m, recon)。"""
        z_m = self.encode(x)
        return z_m, self.decode(z_m)
