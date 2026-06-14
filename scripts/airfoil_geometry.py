"""UIUC/Selig 翼型坐标解析、归一化与几何条件计算。

本模块只处理数组级几何逻辑，不负责批量文件布局或训练配置。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AirfoilSurfaces:
    """按 x 升序排列的上下表面坐标。"""

    upper: np.ndarray
    lower: np.ndarray


@dataclass(frozen=True)
class ResampledAirfoil:
    """重采样后的翼型上下表面。"""

    x: np.ndarray
    y_upper: np.ndarray
    y_lower: np.ndarray

    @property
    def thickness(self) -> np.ndarray:
        return self.y_upper - self.y_lower

    @property
    def camber(self) -> np.ndarray:
        return 0.5 * (self.y_upper + self.y_lower)


def parse_selig_dat(path: Path) -> np.ndarray:
    """解析 UIUC/Selig 坐标文件，跳过标题和非数值行。"""
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
            if np.isfinite(x) and np.isfinite(y):
                rows.append((x, y))
    if len(rows) < 3:
        raise ValueError(f"Not enough coordinate rows in {path}")
    return np.asarray(rows, dtype=np.float64)


def _drop_consecutive_duplicates(coords: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    if len(coords) == 0:
        return coords
    keep = [0]
    for i in range(1, len(coords)):
        if np.linalg.norm(coords[i] - coords[keep[-1]]) > tol:
            keep.append(i)
    return coords[keep]


def normalize_airfoil(coords: np.ndarray) -> np.ndarray:
    """将翼型平移缩放到单位弦长，前缘在 x=0，尾缘中心附近 y=0。"""
    arr = np.asarray(coords, dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = _drop_consecutive_duplicates(arr)
    if len(arr) < 3:
        raise ValueError("Need at least three finite airfoil coordinates")

    x_min = float(arr[:, 0].min())
    x_max = float(arr[:, 0].max())
    chord = x_max - x_min
    if chord <= 0:
        raise ValueError("Airfoil chord must be positive")

    norm = arr.copy()
    norm[:, 0] = (norm[:, 0] - x_min) / chord
    norm[:, 1] = norm[:, 1] / chord

    leading = norm[np.argmin(norm[:, 0])]
    trailing_mask = np.isclose(norm[:, 0], 1.0, atol=1e-6)
    trailing_y = float(norm[trailing_mask, 1].mean()) if trailing_mask.any() else float(norm[np.argmax(norm[:, 0]), 1])
    # 线性弦线校正：让前缘与尾缘中心落到 y=0。
    chordline_y = leading[1] + (trailing_y - leading[1]) * norm[:, 0]
    norm[:, 1] = norm[:, 1] - chordline_y
    return norm


def _sort_unique_by_x(points: np.ndarray) -> np.ndarray:
    order = np.argsort(points[:, 0])
    pts = points[order]
    xs: list[float] = []
    ys: list[float] = []
    for x in np.unique(pts[:, 0]):
        y_vals = pts[np.isclose(pts[:, 0], x), 1]
        xs.append(float(x))
        ys.append(float(y_vals.mean()))
    return np.column_stack([xs, ys]).astype(np.float64)


def split_surfaces(coords: np.ndarray) -> AirfoilSurfaces:
    """按前缘索引将闭合翼型拆成上下表面，并按 x 升序返回。"""
    arr = _drop_consecutive_duplicates(np.asarray(coords, dtype=np.float64))
    if len(arr) < 4:
        raise ValueError("Need at least four coordinates to split surfaces")

    le_idx = int(np.argmin(arr[:, 0]))
    first = arr[: le_idx + 1]
    second = arr[le_idx:]
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Cannot split airfoil surfaces from coordinate order")

    first_sorted = _sort_unique_by_x(first)
    second_sorted = _sort_unique_by_x(second)
    probe_x = np.linspace(0.05, 0.95, 19)
    first_y = np.interp(probe_x, first_sorted[:, 0], first_sorted[:, 1])
    second_y = np.interp(probe_x, second_sorted[:, 0], second_sorted[:, 1])
    if float(first_y.mean()) >= float(second_y.mean()):
        upper, lower = first_sorted, second_sorted
    else:
        upper, lower = second_sorted, first_sorted
    return AirfoilSurfaces(upper=upper, lower=lower)


def cosine_x_grid(n: int = 257) -> np.ndarray:
    """返回 [0, 1] 上的 cosine spacing x 网格。"""
    if n < 2:
        raise ValueError("n must be at least 2")
    i = np.arange(n, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(np.pi * i / (n - 1)))


def resample_surfaces(surfaces: AirfoilSurfaces, x_grid: np.ndarray) -> ResampledAirfoil:
    """将上下表面插值到统一 x 网格。"""
    x = np.asarray(x_grid, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x_grid must be a one-dimensional array with at least two points")
    upper = _sort_unique_by_x(surfaces.upper)
    lower = _sort_unique_by_x(surfaces.lower)
    y_upper = np.interp(x, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x, lower[:, 0], lower[:, 1])
    return ResampledAirfoil(x=x, y_upper=y_upper, y_lower=y_lower)


def compute_airfoil_conditions(resampled: ResampledAirfoil) -> dict[str, float]:
    """计算第一版条件扩散使用的五维翼型几何条件。"""
    thickness = resampled.thickness
    camber_abs = np.abs(resampled.camber)
    t_idx = int(np.argmax(thickness))
    c_idx = int(np.argmax(camber_abs))
    return {
        "t_max": float(thickness[t_idx]),
        "x_tmax": float(resampled.x[t_idx]),
        "camber_max": float(camber_abs[c_idx]),
        "x_camber_max": float(resampled.x[c_idx]),
        "te_gap": float(thickness[-1]),
    }


def validate_airfoil(resampled: ResampledAirfoil, conditions: dict[str, float]) -> list[str]:
    """返回翼型过滤原因列表；空列表表示通过基础几何检查。"""
    reasons: list[str] = []
    thickness = resampled.thickness
    if not np.isfinite(resampled.x).all() or not np.isfinite(resampled.y_upper).all() or not np.isfinite(resampled.y_lower).all():
        reasons.append("non_finite")
    if conditions.get("t_max", 0.0) <= 0:
        reasons.append("nonpositive_tmax")
    if conditions.get("t_max", 0.0) > 0.35:
        reasons.append("excessive_tmax")
    if conditions.get("t_max", 0.0) < 0.02:
        reasons.append("thin_airfoil")
    if conditions.get("te_gap", 0.0) < -1e-4:
        reasons.append("negative_te_gap")
    if np.mean(thickness < -1e-5) > 0.02:
        reasons.append("negative_thickness")
    return reasons
