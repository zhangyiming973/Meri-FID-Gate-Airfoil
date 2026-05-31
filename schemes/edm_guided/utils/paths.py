"""路径与运行目录工具：定位项目根目录、方案名及带时间戳的输出目录。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """返回 meri-fid-gate 项目根目录（utils 上溯三级）。"""
    return Path(__file__).resolve().parents[3]


def scheme_name() -> str:
    """当前方案标识，固定为 ``edm_guided``。"""
    return "edm_guided"


def make_run_dir(dataset: str) -> Path:
    """创建带时间戳的运行输出目录。

    路径形如 ``outputs/edm_guided/<dataset>/YYYYMMDD_HHMMSS/``。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root() / "outputs" / scheme_name() / dataset / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
