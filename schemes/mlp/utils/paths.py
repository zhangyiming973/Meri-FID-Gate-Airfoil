"""路径与运行目录工具：定位项目根目录、方案名及带时间戳的输出目录。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """返回 meri-fid-gate 项目根目录（utils 上溯三级）。"""
    return Path(__file__).resolve().parents[3]


def scheme_name() -> str:
    """当前方案标识，固定为 ``mlp``。"""
    return "mlp"


def make_run_dir(dataset: str) -> Path:
    """创建带时间戳的运行输出目录。

    路径形如 ``outputs/mlp/<dataset>/YYYYMMDD_HHMMSS/``。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root() / "outputs" / scheme_name() / dataset / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def data_processed_dir(dataset: str) -> Path:
    """根据 dataset.json 元信息返回已处理数据目录。"""
    root = project_root()
    meta_path = root / "data" / dataset / "dataset.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return root / "data" / dataset / meta["processed_dir"]
