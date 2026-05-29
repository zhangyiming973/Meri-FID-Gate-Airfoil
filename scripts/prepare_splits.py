#!/usr/bin/env python3
"""为指定数据集生成固定的 train/test 划分文件。

输出目录（默认）::

    data/{dataset}/processed/
        train_split.csv      # 训练集样本索引与条件列
        test_split.csv       # 测试集样本索引与条件列
        condition_stats.json # 训练集条件列均值/标准差（供归一化）
        split_meta.json      # 划分元信息（种子、测试 ID 列表等）

可由 ``python run.py prepare-splits`` 或本脚本直接调用。
划分逻辑与随机种子由 ``data/{dataset}/dataset.json`` 中的 ``split`` 段配置。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.split_utils import build_condition_stats, make_train_test_split  # noqa: E402


def load_dataset_meta(dataset: str) -> dict:
    """读取 ``data/{dataset}/dataset.json`` 并注入数据集名称。

    Args:
        dataset: 数据集目录名，如 ``single``、``F404``。

    Returns:
        解析后的元数据字典，含 ``processed_dir``、``index_file``、``split`` 等字段。

    Raises:
        FileNotFoundError: 元数据文件不存在。
    """
    meta_path = ROOT / "data" / dataset / "dataset.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["name"] = dataset
    return meta


def prepare_splits(dataset: str, force: bool = False) -> Path:
    """生成或跳过已存在的 train/test 划分，并写入条件统计量。

    Args:
        dataset: 数据集名称。
        force: 为 True 时覆盖已有 ``train_split.csv`` / ``test_split.csv``。

    Returns:
        处理后数据目录路径（``processed`` 目录）。

    Raises:
        FileNotFoundError: 缺少 dataset.json 或索引文件。
        ValueError: 质量过滤后无样本、或条件列无法补全。
    """
    meta = load_dataset_meta(dataset)
    processed = ROOT / "data" / dataset / meta["processed_dir"]
    train_path = processed / "train_split.csv"
    test_path = processed / "test_split.csv"

    # 幂等：已有划分且未指定 --force 则直接返回
    if train_path.exists() and test_path.exists() and not force:
        print(f"Splits already exist for {dataset}: {train_path}")
        return processed

    index_path = processed / meta["index_file"]
    split_cfg = meta.get("split", {})
    condition_columns = meta.get("condition_columns")
    # 条件规格 JSONL 路径（相对项目根），用于从几何语义推导条件向量
    specs_rel = None
    if meta.get("condition_specs_file"):
        specs_rel = str(processed.relative_to(ROOT) / meta["condition_specs_file"])

    train_df, test_df = make_train_test_split(
        index_path,
        test_size=split_cfg.get("test_size", 10),
        seed=split_cfg.get("seed", 42),
        output_dir=processed,
        filter_passed_quality=split_cfg.get("filter_passed_quality", True),
        condition_columns=condition_columns,
        root=ROOT,
        condition_specs_path=specs_rel,
        param_table_path=meta.get("param_table_path"),
    )

    # 仅用训练集统计条件列均值/方差，避免测试集信息泄漏
    cond_stats = build_condition_stats(train_df, condition_columns)
    with open(processed / "condition_stats.json", "w", encoding="utf-8") as f:
        json.dump(cond_stats.to_dict(), f, indent=2)

    print(f"Prepared splits for {dataset}:")
    print(f"  train: {len(train_df)} -> {train_path}")
    print(f"  test:  {len(test_df)} -> {test_path}")
    return processed


def main() -> None:
    """命令行入口：``python scripts/prepare_splits.py --dataset <name>``。"""
    parser = argparse.ArgumentParser(description="Prepare fixed dataset train/test splits")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. single or F404")
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files")
    args = parser.parse_args()
    prepare_splits(args.dataset, force=args.force)


if __name__ == "__main__":
    main()
