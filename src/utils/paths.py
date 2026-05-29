"""项目路径工具。

提供项目根目录解析与训练运行输出目录的自动创建。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """返回项目根目录（src/utils 的上两级）。"""
    return Path(__file__).resolve().parents[2]


def make_run_dir(dataset_name: str, module: str, base: Path | None = None) -> Path:
    """创建带时间戳的训练运行目录。

    目录结构：``outputs/<dataset_name>/<module>/<YYYYMMDD_HHMMSS>/``

    Args:
        dataset_name: 数据集名称，用于区分不同数据配置。
        module: 模块标识，如 ``autoencoder``、``diffusion``。
        base: 输出根目录，默认 ``<project_root>/outputs``。

    Returns:
        已创建的 run 目录路径。
    """
    root = base or (project_root() / "outputs")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / dataset_name / module / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
