"""dim_guided 方案数据集：从 NPZ / Excel 加载 SDF 与 ConditionVector。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from schemes.dim_guided.data import CONDITION_COLUMNS
from schemes.dim_guided.data.condition_vector import load_condition_vector


@dataclass
class ConditionStats:
    """条件向量的逐维均值与标准差，用于 z-score 标准化。"""

    mean: np.ndarray
    std: np.ndarray
    columns: list[str]

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / (self.std + 1e-8)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionStats:
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
            columns=list(data["columns"]),
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, columns: list[str]) -> ConditionStats:
        arr = df[columns].astype(float).values
        return cls(mean=arr.mean(axis=0), std=arr.std(axis=0), columns=columns)


def resolve_npz_path(row: pd.Series, npz_dir: Path) -> Path:
    local = npz_dir / f"{row['sample_id']}.npz"
    if local.exists():
        return local
    alt = Path(str(row.get("npz_path", "")))
    if alt.exists():
        return alt
    raise FileNotFoundError(f"NPZ not found for {row['sample_id']}")


def _load_param_table(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


class MeridianSDFDataset(Dataset):
    """子午面 SDF 数据集，条件来自 ConditionVector（Excel / NPZ / 索引）。"""

    def __init__(
        self,
        index_df: pd.DataFrame,
        npz_dir: Path,
        param_table: pd.DataFrame | None = None,
        param_table_path: Path | None = None,
        condition_stats: ConditionStats | None = None,
        use_semantic: bool = True,
    ) -> None:
        self.index_df = index_df.reset_index(drop=True)
        self.npz_dir = Path(npz_dir)
        self.param_table = param_table
        self.param_table_path = Path(param_table_path) if param_table_path else None
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
        semantic = (data["semantic_mask"].astype(np.float32) / 5.0)
        r_grid = data["r_grid"].astype(np.float32)
        physics = json.loads(str(data["physics_summary_json"]))

        cv = load_condition_vector(
            sample_id,
            index_row=row,
            param_table=self.param_table,
            param_table_path=self.param_table_path,
            npz_path=npz_path,
        )
        cond_arr = cv.as_array()
        cond_norm = (
            self.condition_stats.normalize(cond_arr)
            if self.condition_stats
            else cond_arr
        )

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
            "condition_source": cv.source,
            "physics": physics,
            "npz_path": str(npz_path),
        }
