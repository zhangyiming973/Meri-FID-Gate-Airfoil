"""PCA 潜向量 MLP 去噪器。

在 PCA 压缩后的低维潜向量空间上，以工况条件 + 时间步嵌入为输入，
预测 DDPM 噪声 epsilon。适用于 ``denoiser=mlp`` 扩散训练路径。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码，将离散时间步 t 映射为连续向量（Transformer 经典做法）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLPDenoiser(nn.Module):
    """PCA 压缩潜向量 + 工况条件的 MLP 去噪网络。

    输入拼接：[带噪潜向量 x_t | 工况 condition | 时间嵌入 t_emb]，
    输出与 x_t 同维的预测噪声。

    Args:
        latent_dim: PCA 压缩后的潜向量维度。
        cond_dim: 工况向量维度。
        hidden: MLP 隐层宽度。
        time_dim: 时间嵌入维度。
    """

    def __init__(self, latent_dim: int, cond_dim: int = 5, hidden: int = 512, time_dim: int = 128) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """预测噪声 epsilon。

        Args:
            x: 带噪潜向量 (B, latent_dim)。
            t: 时间步 (B,)。
            condition: 工况向量 (B, cond_dim)。
            cond_mask: CFG 训练时的条件 dropout mask，0 表示无条件。

        Returns:
            预测噪声，形状与 x 相同。
        """
        if cond_mask is not None:
            if cond_mask.ndim == 1:
                condition = condition * cond_mask.unsqueeze(-1)
            else:
                condition = condition * cond_mask
        t_emb = self.time_mlp(t)
        h = torch.cat([x, condition, t_emb], dim=-1)
        return self.net(h)
