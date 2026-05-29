"""潜表示质量门控：AE 训练后检验重建与潜空间是否满足扩散前置条件。

在进入 PCA+UNet 扩散阶段前，检查训练集样本的重建 L1 与全局潜标准差，
确保足够比例的样本通过阈值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from schemes.pca_unet.records.body_latent import BodyLatentRecord


@dataclass
class GateConfig:
    """门控阈值配置。"""

    max_recon_l1: float = 0.08       # 单样本重建 L1 上限
    min_latent_std: float = 0.05     # 全局潜向量标准差下限（避免退化）
    min_pass_ratio: float = 0.85     # 样本通过率下限


@dataclass
class GateResult:
    """门控评估结果。"""

    passed: bool                     # 是否整体通过
    pass_ratio: float                # 样本通过率
    report: dict[str, Any]           # 详细报告（供可视化）


class LatentRepresentationGate:
    """AE 潜表示质量门控。

    逐样本标记 ``gate_passed``，并汇总通过率与全局潜标准差，
    判断是否满足进入扩散训练的条件。
    """

    def __init__(self, cfg: GateConfig) -> None:
        self.cfg = cfg

    def evaluate(self, records: list[BodyLatentRecord]) -> GateResult:
        """对训练集潜记录列表执行门控评估。

        Args:
            records: ``build_latent_records`` 产出的训练集记录。

        Returns:
            含通过标志、通过率与详细报告的结果对象。
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
        # 同时要求潜空间有足够方差且多数样本重建达标
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
