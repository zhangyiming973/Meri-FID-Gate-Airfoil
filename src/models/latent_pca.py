"""AE 潜变量 PCA 压缩与持久化。

将高维 z_m (C×H×W) 展平后做 SVD/PCA，降至可扩散建模的低维空间；
训练结束后保存 mean/components/std 供推理时编解码。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class LatentPCA:
    """潜变量 PCA 变换器。

    Attributes:
        mean: 训练集潜变量均值向量，形状 (D,)。
        components: 主成分矩阵，形状 (k, D)。
        std: 投影后系数的标准差，用于归一化。
    """

    mean: np.ndarray
    components: np.ndarray  # (k, dim)
    std: float = 1.0

    @property
    def dim(self) -> int:
        """PCA 保留的主成分数量 k。"""
        return self.components.shape[0]

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """将原始潜变量 z 投影到 PCA 空间并归一化。

        Args:
            z: 形状 (B, C, H, W) 或已展平 (B, D)。

        Returns:
            PCA 系数 (B, k)。
        """
        flat = z.flatten(1)
        mean = torch.from_numpy(self.mean).to(z.device, z.dtype)
        comp = torch.from_numpy(self.components).to(z.device, z.dtype)
        proj = (flat - mean) @ comp.T
        return proj / self.std

    def decode(self, w: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """从 PCA 系数重建原始潜变量形状。

        Args:
            w: PCA 系数 (B, k)。
            shape: 目标潜变量空间形状 (C, H, W)，不含 batch 维。

        Returns:
            重建的 z_m (B, C, H, W)。
        """
        comp = torch.from_numpy(self.components).to(w.device, w.dtype)
        mean = torch.from_numpy(self.mean).to(w.device, w.dtype)
        flat = w * self.std @ comp + mean
        return flat.view(w.shape[0], *shape)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "std": self.std,
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LatentPCA:
        """从字典反序列化。"""
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            components=np.array(data["components"], dtype=np.float32),
            std=float(data.get("std", 1.0)),
        )

    @classmethod
    def fit(cls, z: np.ndarray, n_components: int) -> LatentPCA:
        """在训练集潜变量上拟合 PCA。

        Args:
            z: 形状 (N, C, H, W) 的潜变量数组。
            n_components: 保留的主成分数。

        Returns:
            拟合好的 LatentPCA 实例。
        """
        flat = z.reshape(len(z), -1)
        mean = flat.mean(axis=0)
        x = flat - mean
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        k = min(n_components, vt.shape[0])
        components = vt[:k]
        proj = x @ components.T
        std = max(float(proj.std()), 1e-6)
        return cls(mean=mean.astype(np.float32), components=components.astype(np.float32), std=std)


def save_latent_pca(pca: LatentPCA, path: Path) -> None:
    """将 PCA 参数保存为 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pca.to_dict(), f, indent=2)


def load_latent_pca(path: Path) -> LatentPCA:
    """从 JSON 加载 PCA 参数。"""
    with open(path, encoding="utf-8") as f:
        return LatentPCA.from_dict(json.load(f))
