#!/usr/bin/env python3
"""将 ``src/`` 下的模块复制并改写为自包含的 scheme 包（``schemes/mlp``、``schemes/pca_unet``）。

开发时在 ``src/`` 维护单一源码；发布训练/推理前运行本脚本生成各方案独立目录树，
并把 ``from src.xxx`` 替换为 ``from schemes.{scheme}.xxx``，使两套方案可并行演进且互不污染。

每个 scheme 包含：
    - 共享模型/损失/工具（COMMON_FILES）
    - 方案专属去噪网络（MLP 或 PCA-UNet）
    - 裁剪后的训练配置 JSON
    - 数据加载与路径工具（``data/``、``utils/paths.py``）

用法::

    python scripts/build_schemes.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# 两个 scheme 共用的源文件（按子目录分组）
COMMON_FILES = {
    "models": ["autoencoder.py", "diffusion.py", "latent_pca.py"],
    "losses": ["ae_losses.py"],
    "gate": ["latent_gate.py"],
    "records": ["body_latent.py"],
    "utils": ["config.py", "visualization.py"],
    "train": ["diffusion_codec.py"],
}

# 各方案差异：专属模型文件与扩散训练脚本文件名
SCHEME_SPEC = {
    "mlp": {
        "models": ["mlp_denoiser.py"],
        "train_ae": "train_autoencoder.py",
        "train_diff": "train_diffusion.py",
    },
    "pca_unet": {
        "models": ["pca_unet.py"],
        "train_ae": "train_autoencoder.py",
        "train_diff": "train_unet_diffusion.py",
    },
}


def rewrite(content: str, scheme: str) -> str:
    """把源码中的 ``from src.`` 导入改写为 ``from schemes.{scheme}.``。"""
    content = content.replace("from src.", f"from schemes.{scheme}.")
    return content


def copy_file(src: Path, dst: Path, scheme: str) -> None:
    """复制单个源文件到 scheme 目录，并应用 import 改写。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    dst.write_text(rewrite(text, scheme), encoding="utf-8")


def build_scheme(scheme: str) -> None:
    """为指定方案重建 ``schemes/{scheme}/`` 完整目录树。

    若目标目录已存在则先删除，保证与 ``src/`` 完全同步。

    Args:
        scheme: ``mlp`` 或 ``pca_unet``。
    """
    spec = SCHEME_SPEC[scheme]
    base = ROOT / "schemes" / scheme
    if base.exists():
        shutil.rmtree(base)
    (base / "config").mkdir(parents=True)

    # 复制共享模块并写入空 __init__.py 使其成为 Python 包
    for subdir, files in COMMON_FILES.items():
        for fname in files:
            copy_file(SRC / subdir / fname, base / subdir / fname, scheme)
        (base / subdir / "__init__.py").write_text("", encoding="utf-8")

    # 方案专属模型与训练入口（扩散脚本统一命名为 train_diffusion.py）
    for fname in spec["models"]:
        copy_file(SRC / "models" / fname, base / "models" / fname, scheme)

    copy_file(SRC / "train" / spec["train_ae"], base / "train" / "train_ae.py", scheme)
    copy_file(SRC / "train" / spec["train_diff"], base / "train" / "train_diffusion.py", scheme)
    (base / "train" / "__init__.py").write_text("", encoding="utf-8")
    (base / "__init__.py").write_text("", encoding="utf-8")

    # 生成 scheme 专属的数据层、路径工具与精简配置
    write_scheme_data(base, scheme)
    write_scheme_paths(base, scheme)
    write_scheme_configs(base, scheme)


def write_scheme_data(base: Path, scheme: str) -> None:
    """写入 ``data/``：条件列常量、dataset 加载器、splits 封装。

    ``splits.py`` 委托 ``scripts.split_utils`` 读取固定划分，
    训练代码仅依赖 ``schemes.{scheme}.data`` 接口。
    """
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "data" / "__init__.py").write_text(
        'CONDITION_COLUMNS = [\n'
        '    "hub_r_end_mm",\n'
        '    "rim_r_start_mm",\n'
        '    "angle_web_deg",\n'
        '    "r_trans_bore_web_mm",\n'
        '    "z_min",\n'
        ']\n',
        encoding="utf-8",
    )
    dataset_py = (SRC / "data" / "dataset.py").read_text(encoding="utf-8")
    dataset_py = rewrite(dataset_py, scheme)
    dataset_py = dataset_py.replace(
        "from src.data import CONDITION_COLUMNS",
        f"from schemes.{scheme}.data import CONDITION_COLUMNS",
    )
    (base / "data" / "dataset.py").write_text(dataset_py, encoding="utf-8")

    splits_py = f'''from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from schemes.{scheme}.data import CONDITION_COLUMNS
from schemes.{scheme}.data.dataset import ConditionStats
from scripts.split_utils import load_fixed_splits, build_condition_stats as _build_stats


def load_splits(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_fixed_splits(processed_dir)


def load_condition_stats(processed_dir: Path, columns: list[str] | None = None) -> ConditionStats:
    path = processed_dir / "condition_stats.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return ConditionStats.from_dict(json.load(f))
    train_df, _ = load_splits(processed_dir)
    return _build_stats(train_df, columns or CONDITION_COLUMNS)


def build_condition_stats(train_df: pd.DataFrame, columns: list[str] | None = None) -> ConditionStats:
    return _build_stats(train_df, columns or CONDITION_COLUMNS)
'''
    (base / "data" / "splits.py").write_text(splits_py, encoding="utf-8")


def write_scheme_paths(base: Path, scheme: str) -> None:
    """生成 ``utils/paths.py``：项目根、run 目录命名、processed 数据路径解析。"""
    paths_py = f'''from __future__ import annotations

from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def scheme_name() -> str:
    return "{scheme}"


def make_run_dir(dataset: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root() / "outputs" / scheme_name() / dataset / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def data_processed_dir(dataset: str) -> Path:
    root = project_root()
    meta_path = root / "data" / dataset / "dataset.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return root / "data" / dataset / meta["processed_dir"]
'''
    # 模板中 data_processed_dir 使用 json，需在函数前插入 import
    paths_py = paths_py.replace(
        "def data_processed_dir",
        "import json\n\n\ndef data_processed_dir",
    )
    (base / "utils" / "paths.py").write_text(paths_py, encoding="utf-8")


def write_scheme_configs(base: Path, scheme: str) -> None:
    """从 ``configs/`` 裁剪出 scheme 包内仅训练所需的 JSON 配置。

    mlp 方案保留 ``diffusion`` 段；pca_unet 保留 ``unet`` 段。
    分别为 ``single`` 与 ``F404`` 数据集各写一份。
    """
    if scheme == "mlp":
        for ds, src in [("single", "train_single.json"), ("F404", "train.json")]:
            cfg = json.loads((ROOT / "configs" / src).read_text(encoding="utf-8"))
            slim = {
                "dataset": ds,
                "autoencoder": cfg["autoencoder"],
                "latent_gate": cfg["latent_gate"],
                "diffusion": cfg["diffusion"],
                "physics_guidance": cfg.get("physics_guidance", {}),
                "device": cfg.get("device", "cuda"),
                "num_workers": cfg.get("num_workers", 0),
            }
            (base / "config" / f"{ds}.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    else:
        for ds, src in [("single", "train_unet.json"), ("F404", "train_f404_unet.json")]:
            cfg = json.loads((ROOT / "configs" / src).read_text(encoding="utf-8"))
            slim = {
                "dataset": ds,
                "autoencoder": cfg["autoencoder"],
                "latent_gate": cfg["latent_gate"],
                "unet": cfg["unet"],
                "physics_guidance": cfg.get("physics_guidance", {}),
                "device": cfg.get("device", "cuda"),
                "num_workers": cfg.get("num_workers", 0),
            }
            (base / "config" / f"{ds}.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """构建 ``SCHEME_SPEC`` 中列出的全部方案包。"""
    for scheme in SCHEME_SPEC:
        build_scheme(scheme)
        print(f"Built scheme: {scheme}")


if __name__ == "__main__":
    main()
