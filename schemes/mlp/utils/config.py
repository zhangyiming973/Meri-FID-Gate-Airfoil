"""配置加载工具：支持 JSON / YAML 格式的方案与训练配置。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """从文件加载配置字典。

    根据后缀自动选择解析器：``.json`` 用标准库，``.yaml`` / ``.yml`` 需 PyYAML。
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
