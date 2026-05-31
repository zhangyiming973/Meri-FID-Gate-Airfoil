"""体潜码记录：AE 编码后每个样本的潜向量、条件与重建误差。

序列化为 JSON，供扩散训练与潜空间质量门控使用。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BodyLatentRecord:
    """单样本的体潜码与元数据。

    Attributes:
        sample_id: 样本唯一标识。
        z_m: AE 编码得到的潜张量，形状通常为 (C, H, W)。
        condition_raw: 未归一化的设计条件向量（如厚度、宽度等）。
        condition_norm: 归一化后的条件，供扩散模型作为条件输入。
        recon_l1/l2: 重建 SDF 与真值的 L1/L2 误差。
        zero_band_l1: 零等值面附近的加权 L1 误差。
        dim_error: 由 SDF 估计的最小厚度、面积与真值物理量的偏差。
        semantic_error: 语义区域覆盖率一致性损失。
        gate_passed: 是否通过潜表示质量门控（由 gate 模块写入）。
        npz_path: 对应 NPZ 数据文件路径。
        extra: 扩展字段，便于后续追加指标。
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
        """转为可 JSON 序列化的字典（数组转为列表）。"""
        d = asdict(self)
        d["z_m"] = self.z_m.tolist()
        d["condition_raw"] = self.condition_raw.tolist()
        d["condition_norm"] = self.condition_norm.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyLatentRecord:
        """从 JSON 反序列化，列表字段恢复为 float32 ndarray。"""
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
    """从 JSON 文件加载体潜码记录列表。"""
    with open(path, encoding="utf-8") as f:
        return [BodyLatentRecord.from_dict(x) for x in json.load(f)]
