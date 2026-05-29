"""扩散潜码编解码器：在 AE 原始潜空间与 PCA 压缩空间之间转换。

MLP 去噪器在 PCA 低维向量上操作；采样结果需 decode 回 (C,H,W) 再经 AE 解码为 SDF。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from schemes.mlp.models.latent_pca import LatentPCA


@dataclass
class DiffusionLatentCodec:
    """AE 潜码 <-> MLP 扩散向量的双向编解码。

    Attributes:
        pca: 在训练集 z_m 上拟合的主成分分析器。
        z_shape: 原始潜张量形状 (N, C, H, W) 中的 (C, H, W) 部分，用于 decode 时 reshape。
    """

    pca: LatentPCA
    z_shape: tuple[int, ...]

    @classmethod
    def from_mode(cls, mode: str, pca: LatentPCA, z_shape: tuple[int, ...]) -> DiffusionLatentCodec:
        """按模式构造编解码器（当前 MLP 方案仅使用 PCA 路径）。"""
        return cls(pca=pca, z_shape=z_shape)

    def encode_raw(self, z_raw: torch.Tensor) -> torch.Tensor:
        """将 AE 潜码 flatten 后 PCA 投影并标准化。"""
        return self.pca.encode(z_raw)

    def decode_to_raw(self, z_enc: torch.Tensor) -> torch.Tensor:
        """将 PCA 向量逆变换为 AE 潜张量形状。"""
        return self.pca.decode(z_enc, self.z_shape[1:])

    def encode_numpy(self, z_raw_np: np.ndarray) -> torch.Tensor:
        """NumPy 数组版本的 encode_raw。"""
        return self.encode_raw(torch.from_numpy(z_raw_np))

    def sample_shape(self, batch: int = 1) -> tuple[int, ...]:
        """扩散采样时噪声张量的形状：(batch, pca_dim)。"""
        return (batch, self.pca.dim)
