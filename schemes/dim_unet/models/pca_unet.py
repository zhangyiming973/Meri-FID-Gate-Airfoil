"""PCA 网格 UNet：在 PCA 系数重塑的 2D 网格上做条件扩散去噪。

与直接在 AE 潜特征图上扩散不同，本模块先将 k 维 PCA 向量因子分解为
C×H×W 网格，再用带跳跃连接的条件 UNet 预测噪声。
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
    """将 PCA 系数向量重塑为 2D 网格的规格说明。

    通过枚举通道数 C∈{1,2,4,8} 及因子分解 H×W，选取长宽比最接近方形的布局，
    以便 UNet 在下采样/上采样时保持合理的空间结构。
    """

    channels: int
    height: int
    width: int

    @property
    def dim(self) -> int:
        """网格元素总数 C×H×W，应等于 PCA 维度 k。"""
        return self.channels * self.height * self.width

    @classmethod
    def from_dim(cls, dim: int) -> PcaGridSpec:
        """根据 PCA 维度 k 自动搜索最优 C×H×W 分解。

        评分准则：UNet 需两次 stride=2 下采样，优先 H/W 为 4 的倍数且接近正方形。
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
                # UNet 两次 /2 下采样：H、W 为 4 的倍数时 skip 尺寸最稳定
                if h % 4 != 0 or w % 4 != 0:
                    score += 5.0
                if score < best_score:
                    best_score = score
                    best = cls(channels=c, height=h, width=w)
        if best is None:
            raise ValueError(f"cannot factor pca_dim={dim} into C×H×W grid")
        return best

    def vec_to_grid(self, v: torch.Tensor) -> torch.Tensor:
        """(B, k) → (B, C, H, W)。"""
        return v.view(v.shape[0], self.channels, self.height, self.width)

    def grid_to_vec(self, g: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, k)。"""
        return g.reshape(g.shape[0], -1)


class SinusoidalPosEmb(nn.Module):
    """扩散时间步的正弦位置编码（Transformer 风格）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """带时间/条件嵌入的 FiLM 调制残差块。"""

    def __init__(self, ch: int, emb_dim: int) -> None:
        super().__init__()
        g = min(8, ch)
        self.n1 = nn.GroupNorm(g, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(g, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        # 嵌入向量生成 scale/shift，对中间特征做仿射调制
        self.emb = nn.Linear(emb_dim, ch * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=-1)
        h = self.n2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.c2(F.silu(h))


class ConditionalUNet(nn.Module):
    """用于 PCA 网格潜扩散的编码器-解码器 UNet（含跳跃连接）。

    输入为带噪 PCA 网格与广播后的条件图拼接；时间步与条件向量分别经 MLP
    嵌入后拼接，注入各 ResBlock。支持 classifier-free guidance 的条件 dropout。
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
        emb_dim = time_dim * 2
        in_ch = in_channels + cond_dim  # 噪声网格 + 条件空间广播

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
        """CFG 训练时对条件做 dropout：mask=0 表示无条件分支。"""
        if cond_mask is None:
            return condition
        if cond_mask.ndim == 1:
            return condition * cond_mask.unsqueeze(-1)
        return condition * cond_mask

    def _cond_map(self, condition: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """将条件向量 (B, cond_dim) 广播为 (B, cond_dim, H, W) 空间图。"""
        b = condition.shape[0]
        return condition.unsqueeze(-1).unsqueeze(-1).expand(b, self.cond_dim, h, w)

    def _align_skip(self, skip: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """将跳跃连接特征的空间尺寸对齐到 ref（处理非 2 幂网格）。"""
        if skip.shape[2:] == ref.shape[2:]:
            return skip
        return F.interpolate(skip, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """预测加噪样本 x 在时间步 t、条件 condition 下的噪声 ε。

        Args:
            x: 带噪 PCA 网格 (B, C, H, W)。
            t: 扩散时间步 (B,)。
            condition: 标准化条件向量 (B, cond_dim)。
            cond_mask: 可选，CFG 条件掩码。

        Returns:
            预测的噪声，形状与 x 相同。
        """
        condition = self._apply_cond_mask(condition, cond_mask)
        emb = torch.cat([self.time_mlp(t), self.cond_mlp(condition)], dim=-1)

        _, _, h, w = x.shape
        x_in = torch.cat([x, self._cond_map(condition, h, w)], dim=1)

        # UNet 编码路径
        e1 = self.rb1(self.in_conv(x_in), emb)
        e2 = self.rb2(self.down1(e1), emb)
        e3 = self.rb_mid1(self.down2(e2), emb)
        m = self.rb_mid2(e3, emb)

        # 解码路径 + 跳跃连接（非 2 幂网格时用插值对齐 skip 尺寸）
        d3 = self.rb_dec2(self.up2(m), emb)
        d2 = self.rb_dec1(self.up1(torch.cat([d3, self._align_skip(e2, d3)], dim=1)), emb)
        d1 = self.out_conv(torch.cat([d2, self._align_skip(e1, d2)], dim=1))
        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)
        return d1


def build_pca_unet(pca_dim: int, cond_dim: int, base_ch: int = 64) -> tuple[ConditionalUNet, PcaGridSpec]:
    """根据 PCA 维度构建匹配的 ConditionalUNet 与网格规格。

    Args:
        pca_dim: PCA 主成分数 k。
        cond_dim: 条件向量维度。
        base_ch: UNet 基础通道数。

    Returns:
        (模型, 网格规格) 元组。
    """
    spec = PcaGridSpec.from_dim(pca_dim)
    if spec.dim != pca_dim:
        raise ValueError(f"grid spec dim {spec.dim} != pca_dim {pca_dim}")
    model = ConditionalUNet(spec.channels, cond_dim, base_ch=base_ch)
    return model, spec
