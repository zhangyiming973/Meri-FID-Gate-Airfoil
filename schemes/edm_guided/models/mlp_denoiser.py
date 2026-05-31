"""EDM 预条件化 MLP 去噪网络 F_θ。

输入为 c_in·x、对数噪声嵌入 c_noise(σ) 与设计条件；输出为 EDM 预条件化网络原始输出。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class NoiseLevelEmb(nn.Module):
    """EDM 噪声水平嵌入：c_noise(σ) = (1/4) ln(σ) 经 MLP 映射。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        return self.mlp(c_noise.unsqueeze(-1))


class EDMLDenoiser(nn.Module):
    """PCA 潜向量 + 设计条件的 EDM MLP 去噪器 F_θ。

    结构：噪声水平嵌入 + 三层全连接，输入拼接 [c_in·x_t, condition, noise_emb]。
    支持 cond_mask 实现 classifier-free guidance 训练时的条件 dropout。
    """

    def __init__(self, latent_dim: int, cond_dim: int = 5, hidden: int = 512, noise_dim: int = 128) -> None:
        super().__init__()
        self.noise_mlp = NoiseLevelEmb(noise_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + noise_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """零初始化最后一层，使初始 D_θ ≈ 恒等映射方向（EDM2 初始化思想）。"""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c_noise: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向：预测 EDM 预条件化目标 F_θ(c_in·x; c_noise, cond)。"""
        if cond_mask is not None:
            if cond_mask.ndim == 1:
                condition = condition * cond_mask.unsqueeze(-1)
            else:
                condition = condition * cond_mask
        n_emb = self.noise_mlp(c_noise)
        h = torch.cat([x, condition, n_emb], dim=-1)
        return self.net(h)
