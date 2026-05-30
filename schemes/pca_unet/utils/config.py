"""PCA-UNet 配置加载：支持 JSON 与 YAML。"""
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
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML not installed. Use a .json config or pip install pyyaml."
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)
