"""训练/测试阶段计时工具，供各 scheme 与 run.py 汇报统计使用。"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    """将秒数格式化为可读字符串（如 ``1h 2m 3.4s``）。"""
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, rem_m = divmod(int(minutes), 60)
    return f"{hours}h {rem_m}m {sec:.1f}s"


def duration_record(seconds: float) -> dict[str, Any]:
    return {"sec": round(seconds, 3), "human": format_duration(seconds)}


class StageTimer:
    """多阶段计时器：记录各阶段耗时并生成 JSON 报告。"""

    def __init__(self) -> None:
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._t0: float | None = None
        self._total_sec: float | None = None
        self._stages: dict[str, float] = {}

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._started_at = _utc_now_iso()
        self._total_sec = None

    def finish(self) -> float:
        if self._t0 is None:
            self.start()
        self._total_sec = time.perf_counter() - self._t0
        self._finished_at = _utc_now_iso()
        return self._total_sec

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = self._stages.get(name, 0.0) + (time.perf_counter() - t0)

    def build_section(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        total_sec = self._total_sec if self._total_sec is not None else sum(self._stages.values())
        section: dict[str, Any] = {
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "total_sec": round(total_sec, 3),
            "total_human": format_duration(total_sec),
            "stages": {k: duration_record(v) for k, v in self._stages.items()},
        }
        if extra:
            section.update(extra)
        return section


def load_timing_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_timing_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def merge_timing_section(path: Path, key: str, section: dict[str, Any]) -> dict[str, Any]:
    """向已有 timing.json 合并 train/test 等分区。"""
    report = load_timing_report(path)
    report[key] = section
    save_timing_report(path, report)
    return report


def finalize_train_timing(
    run_dir: Path,
    scheme: str,
    dataset: str,
    stage: str,
    fast: bool,
    timer: StageTimer,
) -> dict[str, Any]:
    """写入/更新 run_dir/timing.json 的训练分区并返回完整报告。"""
    timer.finish()
    report = load_timing_report(run_dir / "timing.json")
    report.update({"scheme": scheme, "dataset": dataset, "run_dir": str(run_dir)})
    report["train"] = timer.build_section({"stage_requested": stage, "fast": fast})
    save_timing_report(run_dir / "timing.json", report)
    return report


def finalize_test_timing(run_dir: Path, scheme: str, timer: StageTimer) -> dict[str, Any]:
    """写入/更新 run_dir/timing.json 的测试分区并返回完整报告。"""
    timer.finish()
    report = load_timing_report(run_dir / "timing.json")
    report.setdefault("scheme", scheme)
    report.setdefault("run_dir", str(run_dir))
    if "dataset" not in report:
        # outputs/{scheme}/{dataset}/{timestamp}
        parts = run_dir.parts
        if len(parts) >= 2:
            report["dataset"] = parts[-2]
    report["test"] = timer.build_section()
    save_timing_report(run_dir / "timing.json", report)
    return report


def print_timing_summary(report: dict[str, Any], title: str = "Timing Summary") -> None:
    """在控制台打印计时摘要。"""
    print(f"\n=== {title} ===")
    scheme = report.get("scheme", "?")
    dataset = report.get("dataset", "?")
    print(f"scheme={scheme}  dataset={dataset}  run_dir={report.get('run_dir', '?')}")
    if train := report.get("train"):
        print(f"  [train] total={train.get('total_human', '?')}  stages={list(train.get('stages', {}).keys())}")
        for name, rec in train.get("stages", {}).items():
            print(f"    - {name}: {rec.get('human', '?')}")
    if test := report.get("test"):
        print(f"  [test]  total={test.get('total_human', '?')}")
    print()


def collect_timing_reports(
    outputs_root: Path,
    scheme: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """扫描 outputs/ 下各 run 的 timing.json，汇总为列表。"""
    rows: list[dict[str, Any]] = []
    if not outputs_root.is_dir():
        return rows
    scheme_dirs = [outputs_root / scheme] if scheme else sorted(outputs_root.iterdir())
    for scheme_dir in scheme_dirs:
        if not scheme_dir.is_dir():
            continue
        scheme_name = scheme_dir.name
        ds_dirs = [scheme_dir / dataset] if dataset else sorted(scheme_dir.iterdir())
        for ds_dir in ds_dirs:
            if not ds_dir.is_dir():
                continue
            for run_dir in sorted(ds_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                timing_path = run_dir / "timing.json"
                if not timing_path.exists():
                    continue
                rec = load_timing_report(timing_path)
                rec.setdefault("scheme", scheme_name)
                rec.setdefault("dataset", ds_dir.name)
                rec.setdefault("run_dir", str(run_dir))
                rec.setdefault("run_id", run_dir.name)
                rows.append(rec)
    return rows


def export_timing_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """将汇总记录导出为 CSV（便于汇报）。"""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scheme",
        "dataset",
        "run_id",
        "train_total_sec",
        "train_total_human",
        "ae_sec",
        "diffusion_sec",
        "test_total_sec",
        "test_total_human",
        "run_dir",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in rows:
            train = rec.get("train") or {}
            test = rec.get("test") or {}
            stages = train.get("stages") or {}
            writer.writerow(
                {
                    "scheme": rec.get("scheme", ""),
                    "dataset": rec.get("dataset", ""),
                    "run_id": rec.get("run_id", ""),
                    "train_total_sec": train.get("total_sec", ""),
                    "train_total_human": train.get("total_human", ""),
                    "ae_sec": stages.get("autoencoder", {}).get("sec", ""),
                    "diffusion_sec": stages.get("diffusion", {}).get("sec", ""),
                    "test_total_sec": test.get("total_sec", ""),
                    "test_total_human": test.get("total_human", ""),
                    "run_dir": rec.get("run_dir", ""),
                }
            )
