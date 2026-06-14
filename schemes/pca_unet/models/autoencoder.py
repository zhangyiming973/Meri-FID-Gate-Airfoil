"""PCA-UNet 自编码器：将 SDF（及可选语义通道）编码为潜空间 z_m。

编码器通过多次池化将输入压缩为 ``latent_channels × latent_spatial`` 的
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
    """SDF 自编码器：sdf2d (+ semantic_mask) → z_m → sdf2d_recon。

    Args:
        in_channels: 输入通道数（1=仅 SDF，2=SDF+语义）。
        latent_channels: 潜空间通道数 C。
        latent_spatial: 潜空间空间分辨率（整数表示 H=W，也可传 ``[H, W]``）。
        output_size: 重建输出尺寸（H, W）；默认为 256×256。
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_channels: int = 64,
        latent_spatial: int | tuple[int, int] | list[int] = 16,
        output_size: tuple[int, int] | list[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        if isinstance(latent_spatial, int):
            self.latent_spatial = (latent_spatial, latent_spatial)
        else:
            self.latent_spatial = tuple(int(v) for v in latent_spatial)
        self.output_size = tuple(int(v) for v in output_size)
        if len(self.latent_spatial) != 2 or len(self.output_size) != 2:
            raise ValueError("latent_spatial and output_size must be 2D sizes")
        latent_h, latent_w = self.latent_spatial
        out_h, out_w = self.output_size
        if out_h % latent_h != 0 or out_w % latent_w != 0:
            raise ValueError(f"output_size {self.output_size} must be divisible by latent_spatial {self.latent_spatial}")
        if out_h // latent_h != 16 or out_w // latent_w != 16:
            raise ValueError(
                "Current encoder has four stride-2 pooling stages, so output_size / latent_spatial must be 16"
            )

        # 编码器：每个空间维度连续下采样 4 次，最终到 latent_spatial。
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
        h2, w2 = latent_h * 2, latent_w * 2
        h4, w4 = latent_h * 4, latent_w * 4
        h8, w8 = latent_h * 8, latent_w * 8
        # 解码器：latent_spatial → output_size。
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 1),
            nn.Upsample(size=(h2, w2), mode="bilinear", align_corners=False),
            ConvBlock(128, 128),
            nn.Upsample(size=(h4, w4), mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
            nn.Upsample(size=(h8, w8), mode="bilinear", align_corners=False),
            ConvBlock(64, 32),
            nn.Upsample(size=self.output_size, mode="bilinear", align_corners=False),
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
