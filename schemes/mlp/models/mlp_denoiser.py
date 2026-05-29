"""MLP 条件去噪网络：在 PCA 压缩潜向量上做扩散去噪。

输入为带噪潜向量、扩散时间步嵌入与设计条件；输出预测噪声 ε。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码：将整数时间步 t 映射为固定维度的连续嵌入。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        # 对数间隔频率，与 Transformer 位置编码类似
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLPDenoiser(nn.Module):
    """PCA 潜向量 + 设计条件的 MLP 去噪器。

    结构：时间嵌入 MLP + 三层全连接，输入拼接 [x_t, condition, t_emb]。
    支持 cond_mask 实现 classifier-free guidance 训练时的条件 dropout。
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
        """前向：预测加噪样本 x_t 在时间 t 下的噪声。

        cond_mask 为 0 时抹除条件（用于 CFG 的无条件分支或训练 dropout）。
        """
        if cond_mask is not None:
            if cond_mask.ndim == 1:
                condition = condition * cond_mask.unsqueeze(-1)
            else:
                condition = condition * cond_mask
        t_emb = self.time_mlp(t)
        h = torch.cat([x, condition, t_emb], dim=-1)
        return self.net(h)
