"""配置文件加载工具。

支持 JSON 与 YAML 格式的训练/推理配置读取。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """从 JSON 或 YAML 文件加载配置字典。

    Args:
        path: 配置文件路径（``.json`` / ``.yaml`` / ``.yml``）。

    Returns:
        解析后的配置字典。

    Raises:
        ImportError: 读取 YAML 时未安装 PyYAML。
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML not installed. Use configs/train.json or pip install pyyaml."
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)
