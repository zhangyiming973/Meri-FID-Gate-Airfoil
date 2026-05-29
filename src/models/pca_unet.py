"""PCA 网格布局与条件 UNet 去噪器。

将一维 PCA 系数重塑为 C×H×W 二维网格，使标准 UNet 架构可用于潜空间扩散。
包含 PcaGridSpec（因式分解布局）与 ConditionalUNet（带 skip 连接的条件去噪网络）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PcaGridSpec:
    """PCA 向量到 2D 网格的重塑规格。

    将 dim 维 PCA 系数分解为 channels × height × width，
    优先选择接近正方形的空间布局。

    Attributes:
        channels: 通道数 C。
        height: 高度 H。
        width: 宽度 W。
    """

    channels: int
    height: int
    width: int

    @property
    def dim(self) -> int:
        """网格元素总数 C×H×W，应等于 PCA 维度。"""
        return self.channels * self.height * self.width

    @classmethod
    def from_dim(cls, dim: int) -> PcaGridSpec:
        """从 PCA 维度自动搜索最优 C×H×W 分解。

        遍历候选通道数 {1,2,4,8}，在剩余空间维度上枚举因子对 (h,w)，
        以长宽比接近 1 为优先评分标准。

        Raises:
            ValueError: 无法将 dim 分解为有效网格时抛出。
        """
        best: PcaGridSpec | None = None
        best_score = float("inf")
        for c in (1, 2, 4, 8):
            if dim % c != 0:
                continue
            spatial = dim // c
            for h in range(1, int(math.sqrt(spatial)) + 1):
                if spatial % h != 0:
                    continue
                w = spatial // h
                aspect = max(h, w) / max(min(h, w), 1)
                score = aspect + abs(h - w) * 0.1
                if score < best_score:
                    best_score = score
                    best = cls(channels=c, height=h, width=w)
        if best is None:
            raise ValueError(f"cannot factor pca_dim={dim} into C×H×W grid")
        return best

    def vec_to_grid(self, v: torch.Tensor) -> torch.Tensor:
        """(B, dim) → (B, C, H, W)。"""
        return v.view(v.shape[0], self.channels, self.height, self.width)

    def grid_to_vec(self, g: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, dim)。"""
        return g.reshape(g.shape[0], -1)


class SinusoidalPosEmb(nn.Module):
    """正弦时间步位置编码。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """带时间/条件嵌入的残差卷积块（AdaGN 风格 scale-shift 调制）。"""

    def __init__(self, ch: int, emb_dim: int) -> None:
        super().__init__()
        g = min(8, ch)
        self.n1 = nn.GroupNorm(g, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(g, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, ch * 2)  # 输出 scale 与 shift

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=-1)
        h = self.n2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.c2(F.silu(h))


class ConditionalUNet(nn.Module):
    """带 skip 连接的条件 UNet，用于 PCA 网格潜空间扩散。

    工况向量在空间维度广播后与带噪潜变量拼接；
    时间嵌入与条件嵌入合并后注入各 ResBlock。

    Args:
        in_channels: 输入/输出通道数（等于 PcaGridSpec.channels）。
        cond_dim: 工况向量维度。
        base_ch: 基础通道宽度。
        time_dim: 时间嵌入维度。
    """

    def __init__(
        self,
        in_channels: int,
        cond_dim: int = 5,
        base_ch: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        emb_dim = time_dim * 2  # 时间 + 条件嵌入拼接
        in_ch = in_channels + cond_dim  # 输入 = 带噪潜变量 + 广播工况图

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.rb1 = ResBlock(base_ch, emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.rb2 = ResBlock(base_ch * 2, emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.rb_mid1 = ResBlock(base_ch * 4, emb_dim)
        self.rb_mid2 = ResBlock(base_ch * 4, emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.rb_dec2 = ResBlock(base_ch * 2, emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch, 4, stride=2, padding=1)
        self.rb_dec1 = ResBlock(base_ch, emb_dim)
        self.out_conv = nn.Conv2d(base_ch * 2, in_channels, 3, padding=1)

    def _apply_cond_mask(self, condition: torch.Tensor, cond_mask: torch.Tensor | None) -> torch.Tensor:
        """CFG 训练：按 mask 将工况置零（模拟无条件生成）。"""
        if cond_mask is None:
            return condition
        if cond_mask.ndim == 1:
            return condition * cond_mask.unsqueeze(-1)
        return condition * cond_mask

    def _cond_map(self, condition: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """将 (B, cond_dim) 工况广播为 (B, cond_dim, H, W) 空间图。"""
        b = condition.shape[0]
        return condition.unsqueeze(-1).unsqueeze(-1).expand(b, self.cond_dim, h, w)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """预测噪声 epsilon。

        UNet 结构：enc1 → down → mid → up+skip → out。
        """
        condition = self._apply_cond_mask(condition, cond_mask)
        emb = torch.cat([self.time_mlp(t), self.cond_mlp(condition)], dim=-1)

        _, _, h, w = x.shape
        x_in = torch.cat([x, self._cond_map(condition, h, w)], dim=1)

        e1 = self.rb1(self.in_conv(x_in), emb)
        e2 = self.rb2(self.down1(e1), emb)
        e3 = self.rb_mid1(self.down2(e2), emb)
        m = self.rb_mid2(e3, emb)

        d3 = self.rb_dec2(self.up2(m), emb)
        d2 = self.rb_dec1(self.up1(torch.cat([d3, e2], dim=1)), emb)  # skip from e2
        d1 = self.out_conv(torch.cat([d2, e1], dim=1))                  # skip from e1
        return d1


def build_pca_unet(pca_dim: int, cond_dim: int, base_ch: int = 64) -> tuple[ConditionalUNet, PcaGridSpec]:
    """根据 PCA 维度构建匹配的 ConditionalUNet 与网格规格。

    Returns:
        (unet 模型, 网格布局规格) 元组。
    """
    spec = PcaGridSpec.from_dim(pca_dim)
    if spec.dim != pca_dim:
        raise ValueError(f"grid spec dim {spec.dim} != pca_dim {pca_dim}")
    model = ConditionalUNet(spec.channels, cond_dim, base_ch=base_ch)
    return model, spec
