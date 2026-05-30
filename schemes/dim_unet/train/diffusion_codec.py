"""扩散潜编码器：在 AE 原始潜空间与 PCA 网格扩散张量之间转换。

训练流程：z_m → PCA 编码 → 重塑为 C×H×W 网格 → UNet 扩散；
采样流程：UNet 输出网格 → 展平 → PCA 解码 → z_m → AE 解码为 SDF。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from schemes.dim_unet.models.latent_pca import LatentPCA
from schemes.dim_unet.models.pca_unet import PcaGridSpec


@dataclass
class DiffusionLatentCodec:
    """AE 潜变量 ↔ PCA 网格扩散张量的双向编解码器。"""

    pca: LatentPCA
    z_shape: tuple[int, ...]
    grid: PcaGridSpec

    @classmethod
    def from_mode(cls, mode: str, pca: LatentPCA, z_shape: tuple[int, ...]) -> DiffusionLatentCodec:
        grid = PcaGridSpec.from_dim(pca.dim)
        print(f"PCA grid: {pca.dim} -> {grid.channels}×{grid.height}×{grid.width}")
        return cls(pca=pca, z_shape=z_shape, grid=grid)

    def encode_raw(self, z_raw: torch.Tensor) -> torch.Tensor:
        w = self.pca.encode(z_raw)
        return self.grid.vec_to_grid(w)

    def decode_to_raw(self, z_enc: torch.Tensor) -> torch.Tensor:
        return self.pca.decode(self.grid.grid_to_vec(z_enc), self.z_shape[1:])

    def encode_numpy(self, z_raw_np: np.ndarray) -> torch.Tensor:
        return self.encode_raw(torch.from_numpy(z_raw_np))

    def sample_shape(self, batch: int = 1) -> tuple[int, ...]:
        return (batch, self.grid.channels, self.grid.height, self.grid.width)
