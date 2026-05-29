"""PCA-UNet 训练可视化：SDF 对比、训练曲线、门控报告、潜空间 PCA 等。

为 AE 与扩散两阶段提供统一的 matplotlib 绘图与 JSON 保存工具。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sdf_panel(
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    save_path: Path,
    condition: dict[str, float] | None = None,
) -> None:
    """绘制三联图：GT SDF | 重建/生成 SDF | 绝对误差热力图。

    Args:
        gt:  ground truth SDF 二维数组。
        pred: 预测/生成 SDF。
        title: 图标题。
        save_path: 输出 PNG 路径。
        condition: 可选，在标题下方显示设计条件。
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = max(np.abs(gt).max(), np.abs(pred).max(), 1e-6)
    for ax, arr, lbl in zip(axes[:2], [gt, pred], ["GT SDF", "Recon / Gen SDF"]):
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(lbl)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    err = np.abs(gt - pred)
    im2 = axes[2].imshow(err, cmap="magma", origin="lower")
    axes[2].set_title(f"|Error| mean={err.mean():.4f}")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    if condition:
        cond = ", ".join(f"{k}={v:.2f}" for k, v in condition.items())
        fig.suptitle(f"{title}\n{cond}", fontsize=9)
    else:
        fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict[str, list[float]], save_path: Path, title: str) -> None:
    """在同一张图上绘制多条训练/验证曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, v in history.items():
        if v:
            # 验证曲线用虚线区分
            ax.plot(v, label=k, linestyle="--" if k.startswith("val") else "-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_training_dashboard(history: dict[str, list[float]], save_path: Path, title: str) -> None:
    """多子面板训练损失仪表盘，每项指标独立子图。"""
    keys = [k for k, v in history.items() if v]
    n = len(keys)
    if n == 0:
        return
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, key in zip(axes_flat, keys):
        vals = history[key]
        ax.plot(vals, color="#2980b9" if key.startswith("train") else "#e67e22", linewidth=2)
        ax.set_title(key.replace("_", " "))
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        if len(vals) > 1:
            ax.annotate(f"final={vals[-1]:.4f}", xy=(len(vals) - 1, vals[-1]),
                        fontsize=8, ha="right", va="bottom")
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(
    items: list[dict[str, Any]],
    save_path: Path,
    title: str,
    max_samples: int = 10,
) -> None:
    """多样本网格：每行 [GT | Pred | Error] 三列。

    Args:
        items: 含 ``gt``、``pred``、``sample_id`` 的字典列表。
        max_samples: 最多展示行数。
    """
    items = items[:max_samples]
    n = len(items)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(10, 2.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, item in enumerate(items):
        gt, pred = item["gt"], item["pred"]
        err = np.abs(gt - pred)
        vmax = max(np.abs(gt).max(), np.abs(pred).max(), 1e-6)
        for col, (arr, cmap, vmin, vmax_) in enumerate([
            (gt, "RdBu_r", -vmax, vmax),
            (pred, "RdBu_r", -vmax, vmax),
            (err, "magma", None, None),
        ]):
            ax = axes[row, col]
            kw = {"cmap": cmap, "origin": "lower"}
            if vmin is not None:
                kw.update(vmin=vmin, vmax=vmax_)
            ax.imshow(arr, **kw)
            ax.axis("off")
            if row == 0:
                ax.set_title(["GT", "Pred", "Error"][col])
        l1 = float(err.mean())
        sid = item.get("sample_id", f"#{row}")
        axes[row, 0].set_ylabel(sid, fontsize=8, rotation=0, ha="right", va="center")
        axes[row, 2].set_xlabel(f"L1={l1:.4f}", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample_metrics(
    metrics: list[dict[str, Any]],
    save_path: Path,
    title: str,
    value_key: str = "l1",
) -> None:
    """逐样本指标柱状图，高于均值的柱子标红。"""
    if not metrics:
        return
    ids = [m["sample_id"] for m in metrics]
    vals = [m[value_key] for m in metrics]
    mean_v = float(np.mean(vals))
    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.6), 4))
    colors = ["#e74c3c" if v > mean_v else "#3498db" for v in vals]
    ax.bar(ids, vals, color=colors, edgecolor="white")
    ax.axhline(mean_v, color="black", linestyle="--", label=f"mean={mean_v:.4f}")
    ax.set_ylabel(value_key)
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_gate_report(report: dict[str, Any], save_path: Path) -> None:
    """潜表示门控报告：通过/失败计数 + 重建 L1 直方图。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["passed", "failed"], [report["passed_count"], report["failed_count"]],
                color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Gate Pass Count")
    recon = report.get("recon_l1_per_sample", [])
    if recon:
        axes[1].hist(recon, bins=20, color="#3498db", edgecolor="white")
        axes[1].axvline(report["max_recon_l1"], color="red", linestyle="--", label="threshold")
        axes[1].legend()
        axes[1].set_title("Recon L1 Distribution")
    fig.suptitle(f"Pass ratio: {report['pass_ratio']:.1%}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_latent_pca(latents: np.ndarray, save_path: Path, labels: list[str] | None = None) -> None:
    """对潜向量做 PCA 降维到 2D 散点图（可视化潜空间分布）。"""
    if latents.shape[0] < 3:
        return
    x = latents - latents.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    xy = x @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.7, c=np.arange(len(xy)), cmap="viridis")
    if labels:
        for i, lbl in enumerate(labels[: min(15, len(labels))]):
            ax.annotate(lbl.replace("sample_", ""), (xy[i, 0], xy[i, 1]), fontsize=6, alpha=0.8)
    ax.set_title("Latent PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_train_test_metric_compare(
    train_vals: list[float],
    test_vals: list[float],
    save_path: Path,
    title: str = "Train vs Test Recon L1",
) -> None:
    """训练集与测试集重建 L1 箱线图对比（过拟合诊断）。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot([train_vals, test_vals], labels=["train", "test"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][1].set_facecolor("#e67e22")
    ax.set_ylabel("L1")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_json(data: dict[str, Any], path: Path) -> None:
    """将字典保存为 UTF-8 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """将 GPU 张量 detach 并转为 NumPy 数组。"""
    return x.detach().cpu().numpy()
