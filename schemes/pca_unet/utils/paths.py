"""PCA-UNet 方案路径工具：项目根目录、运行输出目录等。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """返回 meri-fid-gate 项目根目录（向上三级自本文件）。"""
    return Path(__file__).resolve().parents[3]


def scheme_name() -> str:
    """当前方案标识符，用于输出目录命名。"""
    return "pca_unet"


def make_run_dir(dataset: str) -> Path:
    """创建带时间戳的训练运行目录。

    路径格式：``outputs/pca_unet/<dataset>/<YYYYMMDD_HHMMSS>/``。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root() / "outputs" / scheme_name() / dataset / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


import json


def data_processed_dir(dataset: str) -> Path:
    """根据 dataset.json 元信息解析处理后数据目录。"""
    root = project_root()
    meta_path = root / "data" / dataset / "dataset.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return root / "data" / dataset / meta["processed_dir"]
