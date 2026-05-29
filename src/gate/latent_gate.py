"""潜变量表示质量门控（Latent Representation Gate）。

在自编码器训练完成后，评估潜空间是否满足扩散建模的前提条件：
重建误差足够低、潜变量具有足够方差、足够比例的样本通过单项阈值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.records.body_latent import BodyLatentRecord


@dataclass
class GateConfig:
    """门控阈值配置。

    Attributes:
        max_recon_l1: 单样本重建 L1 上限，超过则该样本 gate_passed=False。
        min_latent_std: 全局潜变量标准差下限，防止塌缩到常数。
        min_pass_ratio: 通过单样本阈值的样本比例下限。
    """

    max_recon_l1: float = 0.08
    min_latent_std: float = 0.05
    min_pass_ratio: float = 0.85


@dataclass
class GateResult:
    """门控评估结果。

    Attributes:
        passed: 整体是否通过（全局 std 与 pass_ratio 均达标）。
        pass_ratio: 单样本重建达标比例。
        report: 详细报告字典，含逐样本 L1、失败样本列表等。
    """

    passed: bool
    pass_ratio: float
    report: dict[str, Any]


class LatentRepresentationGate:
    """潜变量质量门控评估器。

    在训练集潜变量记录上计算全局统计量与逐样本达标情况，
    决定是否允许进入后续扩散训练阶段。
    """

    def __init__(self, cfg: GateConfig) -> None:
        self.cfg = cfg

    def evaluate(self, records: list[BodyLatentRecord]) -> GateResult:
        """对记录列表执行门控评估，并原地更新每条记录的 gate_passed 字段。

        Args:
            records: 自编码器编码后生成的 BodyLatentRecord 列表。

        Returns:
            GateResult，含整体通过标志与详细报告。
        """
        # 展平所有潜变量，计算全局标准差
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
        # 同时满足：潜空间有方差 + 足够多样本重建达标
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
