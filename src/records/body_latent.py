"""车身潜变量记录（Body Latent Record）。

将自编码器编码后的潜变量 z_m 与重建质量指标、工况向量一并序列化，
供潜变量质量门控（gate）与扩散模型训练使用。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BodyLatentRecord:
    """单样本的潜变量与重建评估记录。

    Attributes:
        sample_id: 样本唯一标识。
        z_m: 自编码器潜变量，形状 (C, H, W)。
        condition_raw: 原始工况向量（未归一化）。
        condition_norm: 归一化后的工况向量。
        recon_l1: 重建 L1 误差。
        recon_l2: 重建 L2（MSE）误差。
        zero_band_l1: 零等值面附近的加权 L1 误差。
        dim_error: 物理尺寸（厚度/面积）估计误差。
        semantic_error: 语义区域一致性误差。
        gate_passed: 是否通过潜变量质量门控。
        npz_path: 源 NPZ 文件路径。
        extra: 扩展字段，供后续流水线附加信息。
    """

    sample_id: str
    z_m: np.ndarray
    condition_raw: np.ndarray
    condition_norm: np.ndarray
    recon_l1: float
    recon_l2: float
    zero_band_l1: float
    dim_error: float
    semantic_error: float
    gate_passed: bool = False
    npz_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典（ndarray 转 list）。"""
        d = asdict(self)
        d["z_m"] = self.z_m.tolist()
        d["condition_raw"] = self.condition_raw.tolist()
        d["condition_norm"] = self.condition_norm.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyLatentRecord:
        """从 JSON 反序列化字典恢复记录对象。"""
        d = dict(data)
        d["z_m"] = np.array(d["z_m"], dtype=np.float32)
        d["condition_raw"] = np.array(d["condition_raw"], dtype=np.float32)
        d["condition_norm"] = np.array(d["condition_norm"], dtype=np.float32)
        return cls(**d)


def save_records(records: list[BodyLatentRecord], path: Path) -> None:
    """将记录列表写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def load_records(path: Path) -> list[BodyLatentRecord]:
    """从 JSON 文件加载记录列表。"""
    with open(path, encoding="utf-8") as f:
        return [BodyLatentRecord.from_dict(x) for x in json.load(f)]
