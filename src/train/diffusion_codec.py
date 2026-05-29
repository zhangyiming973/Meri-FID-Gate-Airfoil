from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.latent_pca import LatentPCA
from src.models.pca_unet import PcaGridSpec


@dataclass
class DiffusionLatentCodec:
    """Encode AE latent <-> diffusion tensor (vector or grid)."""

    mode: str
    pca: LatentPCA
    z_shape: tuple[int, ...]
    grid: PcaGridSpec | None = None

    @classmethod
    def from_mode(cls, mode: str, pca: LatentPCA, z_shape: tuple[int, ...]) -> DiffusionLatentCodec:
        grid = PcaGridSpec.from_dim(pca.dim) if mode == "pca_unet" else None
        if mode == "pca_unet" and grid is not None:
            print(f"PCA grid: {pca.dim} -> {grid.channels}×{grid.height}×{grid.width}")
        return cls(mode=mode, pca=pca, z_shape=z_shape, grid=grid)

    def encode_raw(self, z_raw: torch.Tensor) -> torch.Tensor:
        w = self.pca.encode(z_raw)
        if self.mode == "pca_unet" and self.grid is not None:
            return self.grid.vec_to_grid(w)
        return w

    def decode_to_raw(self, z_enc: torch.Tensor) -> torch.Tensor:
        if self.mode == "pca_unet" and self.grid is not None:
            z_enc = self.grid.grid_to_vec(z_enc)
        return self.pca.decode(z_enc, self.z_shape[1:])

    def encode_numpy(self, z_raw_np: np.ndarray) -> torch.Tensor:
        return self.encode_raw(torch.from_numpy(z_raw_np))

    def sample_shape(self, batch: int = 1) -> tuple[int, ...]:
        if self.mode == "pca_unet" and self.grid is not None:
            return (batch, self.grid.channels, self.grid.height, self.grid.width)
        return (batch, self.pca.dim)
