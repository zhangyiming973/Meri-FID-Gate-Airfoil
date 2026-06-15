"""MLP 方案数据划分与条件统计加载。

从 ``data/{dataset}/processed/`` 读取固定的 train/test 划分 CSV，
并加载或计算条件向量的标准化统计量。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from schemes.edm_guided.data import CONDITION_COLUMNS
from schemes.edm_guided.data.dataset import ConditionStats
from scripts.split_utils import load_fixed_splits, build_condition_stats as _build_stats


def load_splits(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载固定的训练集与测试集索引表（train_split.csv / test_split.csv）。"""
    return load_fixed_splits(processed_dir)


def load_condition_stats(processed_dir: Path, columns: list[str] | None = None) -> ConditionStats:
    """加载或计算条件标准化统计量。

    若 ``processed_dir/condition_stats.json`` 已存在则直接读取（由 prepare-splits 生成）；
    否则从训练集重新统计均值与标准差。
    """
    path = processed_dir / "condition_stats.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return ConditionStats.from_dict(json.load(f))
    train_df, _ = load_splits(processed_dir)
    return _build_stats(train_df, columns or CONDITION_COLUMNS)


def build_condition_stats(train_df: pd.DataFrame, columns: list[str] | None = None) -> ConditionStats:
    """从训练集 DataFrame 构建条件统计量（仅训练集，避免测试集信息泄漏）。"""
    return _build_stats(train_df, columns or CONDITION_COLUMNS)
