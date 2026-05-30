"""潜表示质量门控：在扩散训练前校验 AE 潜空间是否可用。

门控标准：
- 全局潜向量标准差不低于阈值（避免塌缩）；
- 足够比例的样本重建 L1 低于上限。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from schemes.dim_guided.records.body_latent import BodyLatentRecord


@dataclass
class GateConfig:
    """门控阈值配置。"""

    max_recon_l1: float = 0.08  # 单样本重建 L1 上限
    min_latent_std: float = 0.05  # 训练集潜向量全局标准差下限
    min_pass_ratio: float = 0.85  # 通过单样本 L1 检查的最低比例


@dataclass
class GateResult:
    """门控评估结果。"""

    passed: bool  # 整体是否通过（比例 + 全局 std 同时满足）
    pass_ratio: float  # 单样本 L1 达标比例
    report: dict[str, Any]  # 详细报告，含失败样本列表等


class LatentRepresentationGate:
    """AE 潜空间质量门控器。"""

    def __init__(self, cfg: GateConfig) -> None:
        self.cfg = cfg

    def evaluate(self, records: list[BodyLatentRecord]) -> GateResult:
        """对训练集体潜码记录执行门控评估。

        逐样本标记 gate_passed，并统计全局潜向量标准差与通过率。
        """
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
        # 需同时满足：潜空间有足够方差，且多数样本重建达标
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
