#!/usr/bin/env python3
"""Plot summary figures for a batch XFOIL evaluation directory."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _case_dir(root: Path, group: str, sample_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in sample_id)
    return root / "cases" / f"{group}__{safe}"


def _read_xy_dat(path: Path) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if math.isfinite(x) and math.isfinite(y):
                rows.append((x, y))
    return np.asarray(rows, dtype=float)


def _read_polar(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    for row in _read_csv(path):
        parsed = {k: _float(v) for k, v in row.items()}
        if math.isfinite(parsed.get("alpha", math.nan)):
            rows.append(parsed)
    return rows


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_convergence(summary: list[dict[str, str]], out_dir: Path) -> Path:
    groups = ["generated", "baseline"]
    labels = ["Generated", "Baseline"]
    totals = [sum(1 for r in summary if r["group"] == g) for g in groups]
    any_ok = [sum(1 for r in summary if r["group"] == g and r["converged_any"] == "True") for g in groups]
    full_ok = [sum(1 for r in summary if r["group"] == g and r["converged_all"] == "True") for g in groups]

    x = np.arange(len(groups))
    width = 0.28
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - width, totals, width, label="Total", color="#8f9aa8")
    ax.bar(x, any_ok, width, label="Any polar point", color="#2f6f9f")
    ax.bar(x + width, full_ok, width, label="Full alpha sweep", color="#2f9f74")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cases")
    ax.set_title("XFOIL Convergence Coverage")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for bars in ax.containers:
        ax.bar_label(bars, padding=2, fontsize=9)
    path = out_dir / "xfoil_convergence_summary.png"
    _save(fig, path)
    return path


def plot_ld_distribution(summary: list[dict[str, str]], out_dir: Path) -> Path:
    generated = _finite([_float(r.get("best_ld")) for r in summary if r["group"] == "generated"])
    baseline = _finite([_float(r.get("best_ld")) for r in summary if r["group"] == "baseline"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0, max(generated + baseline + [1.0]), 18)
    ax.hist(baseline, bins=bins, alpha=0.65, label="Baseline", color="#5f6f52")
    ax.hist(generated, bins=bins, alpha=0.70, label="Generated", color="#2f6f9f")
    ax.axvline(np.mean(generated), color="#2f6f9f", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(baseline), color="#5f6f52", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Best L/D in alpha sweep")
    ax.set_ylabel("Count")
    ax.set_title("Best L/D Distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    path = out_dir / "xfoil_best_ld_distribution.png"
    _save(fig, path)
    return path


def plot_paired_metrics(paired: list[dict[str, str]], out_dir: Path) -> Path:
    ld_x: list[float] = []
    ld_y: list[float] = []
    cl_x: list[float] = []
    cl_y: list[float] = []
    for row in paired:
        g_ld = _float(row.get("generated_best_ld"))
        b_ld = _float(row.get("baseline_best_ld"))
        g_cl = _float(row.get("generated_max_cl"))
        b_cl = _float(row.get("baseline_max_cl"))
        if math.isfinite(g_ld) and math.isfinite(b_ld):
            ld_x.append(b_ld)
            ld_y.append(g_ld)
        if math.isfinite(g_cl) and math.isfinite(b_cl):
            cl_x.append(b_cl)
            cl_y.append(g_cl)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].scatter(ld_x, ld_y, s=34, color="#2f6f9f", alpha=0.85)
    max_ld = max(ld_x + ld_y + [1.0])
    axes[0].plot([0, max_ld], [0, max_ld], color="#777777", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Baseline best L/D")
    axes[0].set_ylabel("Generated best L/D")
    axes[0].set_title("Paired Best L/D")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(cl_x, cl_y, s=34, color="#b8613b", alpha=0.85)
    max_cl = max(cl_x + cl_y + [1.0])
    axes[1].plot([0, max_cl], [0, max_cl], color="#777777", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Baseline max CL")
    axes[1].set_ylabel("Generated max CL")
    axes[1].set_title("Paired Max CL")
    axes[1].grid(alpha=0.25)

    path = out_dir / "xfoil_paired_metric_scatter.png"
    _save(fig, path)
    return path


def plot_top_generated(summary: list[dict[str, str]], out_dir: Path, top_k: int) -> Path:
    generated = [r for r in summary if r["group"] == "generated" and math.isfinite(_float(r.get("best_ld")))]
    top = sorted(generated, key=lambda r: _float(r.get("best_ld")), reverse=True)[:top_k]
    labels = [r["sample_id"] for r in top][::-1]
    values = [_float(r.get("best_ld")) for r in top][::-1]
    colors = ["#2f6f9f" if _float(r.get("num_points")) >= _float(r.get("expected_points")) else "#7aa6c2" for r in top][::-1]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Best L/D")
    ax.set_title(f"Top {len(top)} Generated Airfoils by XFOIL Best L/D")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v:.1f}", va="center", fontsize=9)
    path = out_dir / "xfoil_top_generated_best_ld.png"
    _save(fig, path)
    return path


def plot_top_polars(root: Path, summary: list[dict[str, str]], out_dir: Path, top_k: int) -> Path:
    generated = [r for r in summary if r["group"] == "generated" and math.isfinite(_float(r.get("best_ld")))]
    top = sorted(generated, key=lambda r: _float(r.get("best_ld")), reverse=True)[:top_k]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for row in top:
        sample_id = row["sample_id"]
        polar = _read_polar(_case_dir(root, "generated", sample_id) / "polar.csv")
        if not polar:
            continue
        alpha = np.asarray([r["alpha"] for r in polar], dtype=float)
        cl = np.asarray([r["cl"] for r in polar], dtype=float)
        cd = np.asarray([r["cd"] for r in polar], dtype=float)
        ld = np.divide(cl, cd, out=np.full_like(cl, np.nan), where=cd > 0)
        axes[0].plot(alpha, cl, marker="o", linewidth=1.5, label=sample_id)
        axes[1].plot(alpha, cd, marker="o", linewidth=1.5)
        axes[2].plot(alpha, ld, marker="o", linewidth=1.5)

    axes[0].set_title("CL vs Alpha")
    axes[0].set_xlabel("Alpha [deg]")
    axes[0].set_ylabel("CL")
    axes[1].set_title("CD vs Alpha")
    axes[1].set_xlabel("Alpha [deg]")
    axes[1].set_ylabel("CD")
    axes[2].set_title("L/D vs Alpha")
    axes[2].set_xlabel("Alpha [deg]")
    axes[2].set_ylabel("L/D")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    path = out_dir / "xfoil_top_generated_polar_curves.png"
    _save(fig, path)
    return path


def plot_airfoil_shapes(root: Path, summary: list[dict[str, str]], out_dir: Path, top_k: int) -> Path:
    generated = [r for r in summary if r["group"] == "generated" and math.isfinite(_float(r.get("best_ld")))]
    top = sorted(generated, key=lambda r: _float(r.get("best_ld")), reverse=True)[:top_k]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    offset = 0.0
    for row in top:
        sample_id = row["sample_id"]
        coords = _read_xy_dat(_case_dir(root, "generated", sample_id) / "airfoil.dat")
        if len(coords) == 0:
            continue
        ax.plot(coords[:, 0], coords[:, 1] + offset, linewidth=1.2, label=sample_id)
        ax.text(1.02, offset, sample_id, va="center", fontsize=8)
        offset -= 0.09
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.03, 1.22)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c with vertical offsets")
    ax.set_title(f"Top {len(top)} Generated Airfoil Shapes Sent to XFOIL")
    ax.grid(alpha=0.2)
    path = out_dir / "xfoil_top_generated_airfoil_shapes.png"
    _save(fig, path)
    return path


def write_index(root: Path, out_dir: Path, paths: list[Path]) -> Path:
    lines = [
        "# XFOIL 可视化索引",
        "",
        "这些图基于本目录的 `xfoil_summary.csv`、`xfoil_paired_comparison.csv` 和各 case 的 `polar.csv` 生成。",
        "",
    ]
    for path in paths:
        rel = path.relative_to(root)
        title = path.stem.replace("_", " ")
        lines.append(f"- [{title}]({rel})")
    index = out_dir / "visualization_index.md"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot XFOIL batch result visualizations.")
    parser.add_argument("--xfoil-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    root = args.xfoil_dir
    out_dir = args.out_dir or root / "visualizations"
    summary = _read_csv(root / "xfoil_summary.csv")
    paired = _read_csv(root / "xfoil_paired_comparison.csv")

    paths = [
        plot_convergence(summary, out_dir),
        plot_ld_distribution(summary, out_dir),
        plot_paired_metrics(paired, out_dir),
        plot_top_generated(summary, out_dir, args.top_k),
        plot_top_polars(root, summary, out_dir, min(args.top_k, 8)),
        plot_airfoil_shapes(root, summary, out_dir, min(args.top_k, 10)),
    ]
    index = write_index(root, out_dir, paths)
    print(f"Wrote {len(paths)} figures and index to {out_dir}")
    print(index)


if __name__ == "__main__":
    main()
