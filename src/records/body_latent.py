from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BodyLatentRecord:
    sample_id: str
    z_m: np.ndarray
    condition_raw: np.ndarray
    condition_norm: np.ndarray
    recon_l1: float
    recon_l2: float
    zero_band_l1: float
    dim_error: float
    semantic_error: float
    gate_passed: bool = False
    npz_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["z_m"] = self.z_m.tolist()
        d["condition_raw"] = self.condition_raw.tolist()
        d["condition_norm"] = self.condition_norm.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyLatentRecord:
        d = dict(data)
        d["z_m"] = np.array(d["z_m"], dtype=np.float32)
        d["condition_raw"] = np.array(d["condition_raw"], dtype=np.float32)
        d["condition_norm"] = np.array(d["condition_norm"], dtype=np.float32)
        return cls(**d)


def save_records(records: list[BodyLatentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def load_records(path: Path) -> list[BodyLatentRecord]:
    with open(path, encoding="utf-8") as f:
        return [BodyLatentRecord.from_dict(x) for x in json.load(f)]
