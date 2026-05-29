from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class LatentPCA:
    mean: np.ndarray
    components: np.ndarray  # (k, dim)
    std: float = 1.0

    @property
    def dim(self) -> int:
        return self.components.shape[0]

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        flat = z.flatten(1)
        mean = torch.from_numpy(self.mean).to(z.device, z.dtype)
        comp = torch.from_numpy(self.components).to(z.device, z.dtype)
        proj = (flat - mean) @ comp.T
        return proj / self.std

    def decode(self, w: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        comp = torch.from_numpy(self.components).to(w.device, w.dtype)
        mean = torch.from_numpy(self.mean).to(w.device, w.dtype)
        flat = w * self.std @ comp + mean
        return flat.view(w.shape[0], *shape)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "std": self.std,
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LatentPCA:
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            components=np.array(data["components"], dtype=np.float32),
            std=float(data.get("std", 1.0)),
        )

    @classmethod
    def fit(cls, z: np.ndarray, n_components: int) -> LatentPCA:
        """z: (N, C, H, W)"""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pca.to_dict(), f, indent=2)


def load_latent_pca(path: Path) -> LatentPCA:
    with open(path, encoding="utf-8") as f:
        return LatentPCA.from_dict(json.load(f))
