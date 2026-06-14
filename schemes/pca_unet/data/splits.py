"""PCA-UNet 数据划分与条件统计加载。

封装项目级 ``scripts/split_utils``，为方案提供统一的 train/test 划分
与 ``ConditionStats`` 构建接口。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from schemes.pca_unet.data import CONDITION_COLUMNS
from schemes.pca_unet.data.dataset import ConditionStats
from scripts.split_utils import load_fixed_splits, build_condition_stats as _build_stats


def load_splits(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载固定的训练集与测试集索引表。"""
    return load_fixed_splits(processed_dir)


def load_condition_stats(processed_dir: Path, columns: list[str] | None = None) -> ConditionStats:
    """加载或计算条件标准化统计量。

    若 ``processed_dir/condition_stats.json`` 已存在则直接读取；
    否则从训练集重新统计。
    """
    path = processed_dir / "condition_stats.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return ConditionStats.from_dict(json.load(f))
    train_df, _ = load_splits(processed_dir)
    return _build_stats(train_df, columns or CONDITION_COLUMNS)


def build_condition_stats(train_df: pd.DataFrame, columns: list[str] | None = None) -> ConditionStats:
    """从训练集 DataFrame 构建条件统计量。"""
    return _build_stats(train_df, columns or CONDITION_COLUMNS)
