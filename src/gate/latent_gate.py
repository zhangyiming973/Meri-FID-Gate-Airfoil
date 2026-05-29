from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.records.body_latent import BodyLatentRecord


@dataclass
class GateConfig:
    max_recon_l1: float = 0.08
    min_latent_std: float = 0.05
    min_pass_ratio: float = 0.85


@dataclass
class GateResult:
    passed: bool
    pass_ratio: float
    report: dict[str, Any]


class LatentRepresentationGate:
    def __init__(self, cfg: GateConfig) -> None:
        self.cfg = cfg

    def evaluate(self, records: list[BodyLatentRecord]) -> GateResult:
        latents = np.stack([r.z_m.reshape(-1) for r in records])
        global_std = float(latents.std())
        passed = 0
        recon_l1s = []
        for r in records:
            r.gate_passed = r.recon_l1 <= self.cfg.max_recon_l1
            recon_l1s.append(r.recon_l1)
            if r.gate_passed:
                passed += 1
        ratio = passed / max(len(records), 1)
        ok = global_std >= self.cfg.min_latent_std and ratio >= self.cfg.min_pass_ratio
        report = {
            "passed_count": passed,
            "failed_count": len(records) - passed,
            "pass_ratio": ratio,
            "global_latent_std": global_std,
            "max_recon_l1": self.cfg.max_recon_l1,
            "min_latent_std": self.cfg.min_latent_std,
            "min_pass_ratio": self.cfg.min_pass_ratio,
            "recon_l1_per_sample": recon_l1s,
            "failed_samples": [r.sample_id for r in records if not r.gate_passed],
        }
        return GateResult(passed=ok, pass_ratio=ratio, report=report)
