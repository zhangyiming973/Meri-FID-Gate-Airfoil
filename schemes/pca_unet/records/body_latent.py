"""潜向量记录：AE 编码结果与重建指标的持久化格式。

每条 ``BodyLatentRecord`` 对应一个样本，保存 z_m、条件向量、
各项重建误差及门控状态，供 PCA 拟合与扩散训练加载。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BodyLatentRecord:
    """单样本的 AE 潜表示与质量指标。

    Attributes:
        sample_id: 样本标识。
        z_m: AE 编码潜张量 (C, H, W)。
        condition_raw / condition_norm: 原始与标准化设计条件。
        recon_l1/l2, zero_band_l1, dim_error, semantic_error: 重建分项误差。
        gate_passed: 是否通过潜表示门控。
        npz_path: 源 NPZ 路径。
        extra: 扩展字段。
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
        """转为 JSON 可序列化字典（ndarray → list）。"""
        d = asdict(self)
        d["z_m"] = self.z_m.tolist()
        d["condition_raw"] = self.condition_raw.tolist()
        d["condition_norm"] = self.condition_norm.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyLatentRecord:
        """从字典反序列化。"""
        d = dict(data)
        d["z_m"] = np.array(d["z_m"], dtype=np.float32)
        d["condition_raw"] = np.array(d["condition_raw"], dtype=np.float32)
        d["condition_norm"] = np.array(d["condition_norm"], dtype=np.float32)
        return cls(**d)


def save_records(records: list[BodyLatentRecord], path: Path) -> None:
    """将记录列表保存为 JSON 数组文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def load_records(path: Path) -> list[BodyLatentRecord]:
    """从 JSON 文件加载记录列表。"""
    with open(path, encoding="utf-8") as f:
        return [BodyLatentRecord.from_dict(x) for x in json.load(f)]
