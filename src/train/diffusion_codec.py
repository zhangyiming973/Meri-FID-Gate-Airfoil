"""扩散潜变量编解码器（Diffusion Latent Codec）。

在 AE 原始潜变量 z_m 与扩散模型工作空间（PCA 向量或 PCA 网格）之间转换。
支持两种模式：``mlp``（一维 PCA 向量）与 ``pca_unet``（重塑为 C×H×W 网格供 UNet 处理）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.latent_pca import LatentPCA
from src.models.pca_unet import PcaGridSpec


@dataclass
class DiffusionLatentCodec:
    """AE 潜变量 ↔ 扩散张量的双向编解码器。

    Attributes:
        mode: ``mlp`` 或 ``pca_unet``。
        pca: 拟合好的 LatentPCA 实例。
        z_shape: 原始潜变量形状 (B, C, H, W) 中除 batch 外的部分。
        grid: pca_unet 模式下的网格布局规格，mlp 模式为 None。
    """

    mode: str
    pca: LatentPCA
    z_shape: tuple[int, ...]
    grid: PcaGridSpec | None = None

    @classmethod
    def from_mode(cls, mode: str, pca: LatentPCA, z_shape: tuple[int, ...]) -> DiffusionLatentCodec:
        """根据扩散模式创建编解码器。

        pca_unet 模式会自动推导将 PCA 向量重塑为 2D 网格的 PcaGridSpec。
        """
        grid = PcaGridSpec.from_dim(pca.dim) if mode == "pca_unet" else None
        if mode == "pca_unet" and grid is not None:
            print(f"PCA grid: {pca.dim} -> {grid.channels}×{grid.height}×{grid.width}")
        return cls(mode=mode, pca=pca, z_shape=z_shape, grid=grid)

    def encode_raw(self, z_raw: torch.Tensor) -> torch.Tensor:
        """AE 原始潜变量 → 扩散工作空间张量（PCA 向量或网格）。"""
        w = self.pca.encode(z_raw)
        if self.mode == "pca_unet" and self.grid is not None:
            return self.grid.vec_to_grid(w)
        return w

    def decode_to_raw(self, z_enc: torch.Tensor) -> torch.Tensor:
        """扩散工作空间张量 → AE 原始潜变量 z_m。"""
        if self.mode == "pca_unet" and self.grid is not None:
            z_enc = self.grid.grid_to_vec(z_enc)
        return self.pca.decode(z_enc, self.z_shape[1:])

    def encode_numpy(self, z_raw_np: np.ndarray) -> torch.Tensor:
        """NumPy 数组版本的 encode_raw。"""
        return self.encode_raw(torch.from_numpy(z_raw_np))

    def sample_shape(self, batch: int = 1) -> tuple[int, ...]:
        """返回扩散采样时的张量形状（含 batch 维）。"""
        if self.mode == "pca_unet" and self.grid is not None:
            return (batch, self.grid.channels, self.grid.height, self.grid.width)
        return (batch, self.pca.dim)
