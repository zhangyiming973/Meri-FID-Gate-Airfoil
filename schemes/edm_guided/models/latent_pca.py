"""潜空间 PCA：将 AE 高维潜张量压缩为低维向量供 MLP 扩散。

在训练集 z_m 上 SVD 拟合主成分，投影后按标准差归一化，便于扩散模型学习。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class LatentPCA:
    """基于 SVD 的潜空间主成分分析。

    Attributes:
        mean: 训练集潜向量逐维均值，形状 (D,)。
        components: 前 k 个主成分，形状 (k, D)。
        std: 投影系数的全局标准差，用于 encode/decode 时的尺度归一化。
    """

    mean: np.ndarray
    components: np.ndarray  # (k, dim)
    std: float = 1.0

    @property
    def dim(self) -> int:
        """PCA 保留的主成分数量 k。"""
        return self.components.shape[0]

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, C, H, W) -> PCA 系数 (B, k)，经减均值、投影、除 std。"""
        flat = z.flatten(1)
        mean = torch.from_numpy(self.mean).to(z.device, z.dtype)
        comp = torch.from_numpy(self.components).to(z.device, z.dtype)
        proj = (flat - mean) @ comp.T
        return proj / self.std

    def decode(self, w: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """PCA 系数 (B, k) -> 重构潜张量 (B, *shape)。"""
        comp = torch.from_numpy(self.components).to(w.device, w.dtype)
        mean = torch.from_numpy(self.mean).to(w.device, w.dtype)
        flat = w * self.std @ comp + mean
        return flat.view(w.shape[0], *shape)

    def to_dict(self) -> dict:
        """序列化为 JSON 兼容字典。"""
        return {
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "std": self.std,
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LatentPCA:
        """从字典恢复 PCA 参数。"""
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            components=np.array(data["components"], dtype=np.float32),
            std=float(data.get("std", 1.0)),
        )

    @classmethod
    def fit(cls, z: np.ndarray, n_components: int) -> LatentPCA:
        """在样本集 z 上拟合 PCA。

        Args:
            z: 形状 (N, C, H, W) 的潜码数组。
            n_components: 保留的主成分数，不超过 min(N, D)。
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
