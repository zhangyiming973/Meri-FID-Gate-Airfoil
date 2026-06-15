"""MLP 方案数据集：从 NPZ 加载 SDF、语义掩码与条件向量。

``MeridianSDFDataset`` 为 AE 训练提供 (x, sdf, semantic, condition) 批次；
条件向量经 ``ConditionStats`` 标准化后供 MLP 扩散去噪器使用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from schemes.edm_guided.data import CONDITION_COLUMNS


@dataclass
class ConditionStats:
    """条件向量的逐维均值与标准差，用于 z-score 标准化。"""

    mean: np.ndarray
    std: np.ndarray
    columns: list[str]

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """将原始条件值标准化为零均值、单位方差。"""
        return (values - self.mean) / (self.std + 1e-8)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """将标准化条件还原为物理量（mm、度等）。"""
        return values * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 可存储的字典。"""
        return {"columns": self.columns, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionStats:
        """从字典反序列化（读取 condition_stats.json）。"""
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
            columns=list(data["columns"]),
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, columns: list[str]) -> ConditionStats:
        """从训练集 DataFrame 统计各条件列的均值与标准差。"""
        arr = df[columns].astype(float).values
        return cls(mean=arr.mean(axis=0), std=arr.std(axis=0), columns=columns)


def resolve_npz_path(row: pd.Series, npz_dir: Path) -> Path:
    """解析样本对应的 NPZ 文件路径。

    优先使用 ``npz_dir/<sample_id>.npz``，否则回退到索引表中的 ``npz_path`` 字段。
    """
    local = npz_dir / f"{row['sample_id']}.npz"
    if local.exists():
        return local
    alt = Path(str(row.get("npz_path", "")))
    if alt.exists():
        return alt
    raise FileNotFoundError(f"NPZ not found for {row['sample_id']}")


def load_condition_vector(
    sample_id: str,
    index_row: pd.Series | None,
    param_table: pd.DataFrame | None = None,
    npz_path: Path | None = None,
) -> dict[str, float]:
    """从多种来源加载样本的设计条件向量。

    查找顺序：索引行 → 参数表 CSV → NPZ 内 ``condition_json``。

    Args:
        sample_id: 样本 ID。
        index_row: 数据划分索引中的一行（train_split / test_split）。
        param_table: 可选的全局参数表。
        npz_path: NPZ 路径，用于读取内嵌条件 JSON。

    Returns:
        列名到浮点值的字典，必须覆盖 ``CONDITION_COLUMNS`` 全部列。
    """
    values: dict[str, float] = {}
    if index_row is not None:
        for col in CONDITION_COLUMNS:
            if col in index_row.index and pd.notna(index_row[col]):
                values[col] = float(index_row[col])
    if len(values) == len(CONDITION_COLUMNS):
        return values

    if param_table is not None:
        row = param_table[param_table["id"] == sample_id]
        if not row.empty:
            for col in CONDITION_COLUMNS:
                if col in row.columns:
                    values[col] = float(row.iloc[0][col])
            if len(values) == len(CONDITION_COLUMNS):
                return values

    if npz_path and npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        if "condition_json" in data:
            cond = json.loads(str(data["condition_json"]))
            for col in CONDITION_COLUMNS:
                if col in cond:
                    values[col] = float(cond[col])

    missing = [c for c in CONDITION_COLUMNS if c not in values]
    if missing:
        raise KeyError(f"Missing condition for {sample_id}: {missing}")
    return values


class MeridianSDFDataset(Dataset):
    """子午面 SDF 数据集，供 MLP 方案自编码器训练使用。

    每个样本返回：
      - ``x``: AE 输入（SDF ± 语义掩码，1 或 2 通道）
      - ``sdf`` / ``semantic``: 重建目标与语义区域监督
      - ``condition`` / ``condition_raw``: 标准化与原始设计条件
      - ``physics``: 最小壁厚、面积等物理摘要（用于复合损失）
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        npz_dir: Path,
        param_table: pd.DataFrame | None = None,
        condition_stats: ConditionStats | None = None,
        use_semantic: bool = True,
    ) -> None:
        self.index_df = index_df.reset_index(drop=True)
        self.npz_dir = Path(npz_dir)
        self.param_table = param_table
        self.condition_stats = condition_stats
        self.use_semantic = use_semantic

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.index_df.iloc[idx]
        sample_id = str(row["sample_id"])
        npz_path = resolve_npz_path(row, self.npz_dir)
        data = np.load(npz_path)

        sdf = data["sdf2d_norm"].astype(np.float32)
        # 语义掩码归一化到 [0, 1]（原始值为 0–5 的区域标签：hub/web/rim 等）
        semantic = (data["semantic_mask"].astype(np.float32) / 5.0)
        r_grid = data["r_grid"].astype(np.float32)
        physics = json.loads(str(data["physics_summary_json"]))

        cond_raw = load_condition_vector(sample_id, row, self.param_table, npz_path)
        cond_arr = np.array([cond_raw[c] for c in CONDITION_COLUMNS], dtype=np.float32)
        cond_norm = (
            self.condition_stats.normalize(cond_arr)
            if self.condition_stats
            else cond_arr
        )

        # AE 输入通道：仅 SDF，或 SDF + 语义 mask 拼接
        channels = [sdf[None, ...]]
        if self.use_semantic:
            channels.append(semantic[None, ...])
        x = np.concatenate(channels, axis=0)

        return {
            "sample_id": sample_id,
            "x": torch.from_numpy(x),
            "sdf": torch.from_numpy(sdf[None, ...]),
            "semantic": torch.from_numpy(semantic[None, ...]),
            "r_grid": torch.from_numpy(r_grid[None, ...]),
            "condition_raw": torch.from_numpy(cond_arr),
            "condition": torch.from_numpy(cond_norm.astype(np.float32)),
            "physics": physics,
            "npz_path": str(npz_path),
        }
