"""数据集划分与条件向量（condition）处理的共享工具。

Meridian 叶轮几何由若干标量条件驱动扩散模型；本模块负责：

- 从索引 CSV、参数表或 condition_specs JSONL 补全条件列
- 按固定随机种子划分 train/test 并落盘
- 计算训练集条件列的均值/标准差，供 ``ConditionStats`` 归一化

被 ``scripts/prepare_splits.py`` 与各 scheme 的 ``data/splits.py`` 复用。
"""
from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 默认条件列：轮毂/轮缘半径、腹板倾角、过渡半径、轴向范围下界
CONDITION_COLUMNS = [
    "hub_r_end_mm",
    "rim_r_start_mm",
    "angle_web_deg",
    "r_trans_bore_web_mm",
    "z_min",
]

# 索引列名 -> param_table.csv 中对应字段名的映射
PARAM_TABLE_FIELD_MAP = {
    "hub_r_end_mm": "r_bore2_mm",
    "rim_r_start_mm": "r_outer2_mm",
    "angle_web_deg": "angle_web_deg",
    "r_trans_bore_web_mm": "r_trans_bore_web_mm",
    "z_min": "z_min",
}


class ConditionStats:
    """训练集条件列的逐维均值与标准差，用于推理/训练时 Z-score 归一化。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray, columns: list[str]) -> None:
        """Args:
            mean: 各条件维度的样本均值，形状 ``(n_conditions,)``。
            std: 各条件维度的样本标准差。
            columns: 与 mean/std 对齐的列名列表。
        """
        self.mean = mean
        self.std = std
        self.columns = columns

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """对条件向量做 Z-score 归一化。

        Args:
            values: 形状 ``(..., n_conditions)`` 的浮点数组。

        Returns:
            归一化后的数组；分母加 ``1e-8`` 防止除零。
        """
        return (values - self.mean) / (self.std + 1e-8)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 可写的字典（列表形式的 mean/std）。"""
        return {"columns": self.columns, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionStats:
        """从 ``condition_stats.json`` 反序列化。"""
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
            columns=list(data["columns"]),
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, columns: list[str]) -> ConditionStats:
        """在 DataFrame 指定列上计算均值与标准差。"""
        arr = df[columns].astype(float).values
        return cls(mean=arr.mean(axis=0), std=arr.std(axis=0), columns=columns)


def _load_condition_specs(path: Path) -> dict[str, dict[str, Any]]:
    """读取 JSONL 格式的 per-sample 几何语义规格。

    每行一个 JSON 对象，须含 ``sample_id`` 字段。

    Returns:
        ``sample_id`` -> 规格字典 的查找表。
    """
    specs: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spec = json.loads(line)
            specs[str(spec["sample_id"])] = spec
    return specs


def _derive_conditions_from_spec(spec: dict[str, Any]) -> dict[str, float]:
    """从单条 condition_spec 的几何语义区域推导标量条件向量。

    根据 hub/rim 半径范围、轴向宽度等计算与 ``CONDITION_COLUMNS`` 对齐的五个量；
    ``angle_web_deg`` 由轮缘与轮毂径向差相对轴向宽度用 ``arctan2`` 估算。

    Args:
        spec: 含 ``semantic_regions``、``z_width_mm`` 等的规格字典。

    Returns:
        列名 -> 浮点值 的字典。
    """
    regions = {r["name"]: r for r in spec["semantic_regions"]}
    hub_end = float(regions["hub"]["r_range_mm"][1])
    rim_start = float(regions["rim"]["r_range_mm"][0])
    r_trans = float(regions["hub_web_transition"]["r_range_mm"][0])
    z_width = float(spec["z_width_mm"])
    z_min = -z_width / 2.0  # 轴向对称，以中心为原点
    angle_web_deg = float(np.degrees(np.arctan2(rim_start - hub_end, z_width)))
    return {
        "hub_r_end_mm": hub_end,
        "rim_r_start_mm": rim_start,
        "angle_web_deg": angle_web_deg,
        "r_trans_bore_web_mm": r_trans,
        "z_min": z_min,
    }


def _merge_param_table(df: pd.DataFrame, param_table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """用外部参数表按 sample_id 填充索引中缺失的条件列。

    已存在且非空的列不会被覆盖；仅对全 NaN 或缺失的列尝试映射。

    Args:
        df: 样本索引 DataFrame，须含 ``sample_id``。
        param_table: 参数表，首列或 ``id`` 列作为关联键。
        columns: 需要补全的条件列名列表。

    Returns:
        补全后的 DataFrame 副本。
    """
    df = df.copy()
    id_col = "id" if "id" in param_table.columns else param_table.columns[0]
    lookup = param_table.set_index(id_col)
    for col in columns:
        if col in df.columns and df[col].notna().any():
            continue
        src_col = PARAM_TABLE_FIELD_MAP.get(col, col)
        if src_col not in lookup.columns:
            continue
        mapped = df["sample_id"].map(lookup[src_col])
        if col not in df.columns:
            df[col] = mapped
        else:
            df[col] = df[col].fillna(mapped)
    return df


def enrich_index_conditions(
    df: pd.DataFrame,
    columns: list[str],
    root: Path,
    condition_specs_path: str | None = None,
    param_table_path: str | None = None,
) -> pd.DataFrame:
    """多级回退：为索引 DataFrame 补全条件列。

    优先级：
        1. 索引 CSV 中已有列则跳过
        2. ``param_table_path`` 指向的 CSV（若存在）
        3. ``condition_specs_path`` JSONL 逐样本推导

    Args:
        df: 原始索引表。
        columns: 目标条件列名。
        root: 项目根目录，用于解析相对路径。
        condition_specs_path: 相对 root 的 JSONL 路径，可选。
        param_table_path: 相对 root 的参数表路径，可选。

    Returns:
        条件列齐全的 DataFrame。

    Raises:
        ValueError: 仍无法补全任一必需列。
    """
    missing = [c for c in columns if c not in df.columns or df[c].isna().all()]
    if not missing:
        return df

    df = df.copy()
    if param_table_path:
        pt_path = root / param_table_path
        if pt_path.exists():
            df = _merge_param_table(df, pd.read_csv(pt_path), columns)

    missing = [c for c in columns if c not in df.columns or df[c].isna().all()]
    if missing and condition_specs_path:
        specs_path = root / condition_specs_path
        if specs_path.exists():
            specs = _load_condition_specs(specs_path)
            derived_rows = []
            for _, row in df.iterrows():
                spec = specs.get(str(row["sample_id"]))
                derived_rows.append(_derive_conditions_from_spec(spec) if spec else {})
            for col in missing:
                df[col] = [d.get(col, np.nan) for d in derived_rows]

    still_missing = [c for c in columns if c not in df.columns or df[c].isna().any()]
    if still_missing:
        raise ValueError(
            f"Index is missing condition columns: {still_missing}. "
            "Provide param_table.csv or condition_specs file in dataset metadata."
        )
    return df


def _resolve_test_count(test_size: int | float, total: int) -> int:
    """将 ``test_size`` 解析为测试集样本数。

    ``0 < test_size < 1`` 表示比例；其他数值按样本数处理。
    """
    if isinstance(test_size, bool) or not isinstance(test_size, Real):
        raise TypeError(f"test_size must be an int count or float ratio, got {type(test_size).__name__}")
    if test_size < 0:
        raise ValueError(f"test_size must be non-negative, got {test_size}")
    if 0 < float(test_size) < 1:
        n_test = int(math.ceil(total * float(test_size)))
        return min(max(n_test, 1), max(total - 1, 1)) if total > 1 else total
    return min(int(test_size), total)


def make_train_test_split(
    index_path: Path,
    test_size: int | float = 10,
    seed: int = 42,
    output_dir: Path | None = None,
    filter_passed_quality: bool = True,
    condition_columns: list[str] | None = None,
    root: Path | None = None,
    condition_specs_path: str | None = None,
    param_table_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取索引 CSV，可选质量过滤与条件补全，再随机划分 train/test。

    Args:
        index_path: 全量样本索引 CSV 路径。
        test_size: 测试集样本数上限；``0 < test_size < 1`` 时按比例划分。
        seed: ``pandas.DataFrame.sample`` 的随机种子，保证可复现。
        output_dir: 若给定，写入 train/test CSV 与 ``split_meta.json``。
        filter_passed_quality: 为 True 时仅保留 ``passed_quality==true`` 的行。
        condition_columns: 条件列名；默认 ``CONDITION_COLUMNS``。
        root: 项目根，用于 ``enrich_index_conditions``；为 None 则跳过补全。
        condition_specs_path: JSONL 规格文件相对路径。
        param_table_path: 参数表相对路径。

    Returns:
        ``(train_df, test_df)`` 元组，均已 ``reset_index``。

    Raises:
        ValueError: 过滤后无样本、或索引为空。
    """
    columns = condition_columns or CONDITION_COLUMNS
    df = pd.read_csv(index_path)

    if filter_passed_quality and "passed_quality" in df.columns:
        before = len(df)
        df = df[df["passed_quality"].astype(str).str.lower() == "true"].copy()
        if len(df) == 0:
            raise ValueError(
                f"No samples left after passed_quality filter (removed {before} rows). "
                "Set split.filter_passed_quality=false in data/{dataset}/dataset.json."
            )

    if root is not None:
        df = enrich_index_conditions(df, columns, root, condition_specs_path, param_table_path)

    # 按 sample_id 排序，保证相同种子下划分稳定（与行顺序无关）
    df = df.sort_values("sample_id").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"Index is empty: {index_path}")

    n_test = _resolve_test_count(test_size, len(df))
    test_idx = df.sample(n=n_test, random_state=seed).index
    test_df = df.loc[test_idx].reset_index(drop=True)
    train_df = df.drop(index=test_idx).reset_index(drop=True)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / "train_split.csv", index=False)
        test_df.to_csv(output_dir / "test_split.csv", index=False)
        with open(output_dir / "split_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_size": n_test,
                    "requested_test_size": test_size,
                    "seed": seed,
                    "filter_passed_quality": filter_passed_quality,
                    "train_count": len(train_df),
                    "test_count": len(test_df),
                    "test_ids": test_df["sample_id"].tolist(),
                },
                f,
                indent=2,
            )
    return train_df, test_df


def load_fixed_splits(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从 processed 目录加载已固定的 train/test 划分。

    Args:
        processed_dir: 含 ``train_split.csv`` 与 ``test_split.csv`` 的目录。

    Returns:
        ``(train_df, test_df)``。

    Raises:
        FileNotFoundError: 划分文件缺失（需先运行 prepare-splits）。
    """
    train_path = processed_dir / "train_split.csv"
    test_path = processed_dir / "test_split.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing split files in {processed_dir}. "
            "Run: python run.py prepare-splits --dataset <name>"
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_condition_stats(train_df: pd.DataFrame, columns: list[str] | None = None) -> ConditionStats:
    """在训练集 DataFrame 上构建 ``ConditionStats``。

    Args:
        train_df: 训练划分表。
        columns: 条件列；默认 ``CONDITION_COLUMNS``。
    """
    return ConditionStats.from_dataframe(train_df, columns or CONDITION_COLUMNS)
