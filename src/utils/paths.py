from __future__ import annotations

from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def make_run_dir(dataset_name: str, module: str, base: Path | None = None) -> Path:
    root = base or (project_root() / "outputs")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / dataset_name / module / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
