"""翼型二维 SDF 栅格化工具。"""
from __future__ import annotations

import numpy as np

from scripts.airfoil_geometry import ResampledAirfoil


def build_airfoil_polygon(resampled: ResampledAirfoil) -> np.ndarray:
    """由重采样上下表面构造闭合翼型多边形。"""
    upper = np.column_stack([resampled.x, resampled.y_upper])
    lower = np.column_stack([resampled.x[-2:0:-1], resampled.y_lower[-2:0:-1]])
    polygon = np.vstack([upper, lower, upper[:1]])
    return polygon.astype(np.float64)


def make_sdf_grid(
    height: int,
    width: int,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """构造 SDF 采样网格，返回形状为 ``(height, width)`` 的 x/y 坐标。"""
    if height < 2 or width < 2:
        raise ValueError("height and width must be at least 2")
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, width, dtype=np.float64)
    ys = np.linspace(y_min, y_max, height, dtype=np.float64)
    return np.meshgrid(xs, ys)


def _point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    seg = end - start
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 <= 1e-24:
        return np.linalg.norm(points - start, axis=1)
    t = ((points - start) @ seg) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    projection = start + t[:, None] * seg
    return np.linalg.norm(points - projection, axis=1)


def _unsigned_distance_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    distances = np.full(points.shape[0], np.inf, dtype=np.float64)
    for i in range(len(polygon) - 1):
        d = _point_segment_distances(points, polygon[i], polygon[i + 1])
        distances = np.minimum(distances, d)
    return distances


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Ray casting point-in-polygon test."""
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    px = polygon[:, 0]
    py = polygon[:, 1]
    j = len(polygon) - 1
    for i in range(len(polygon)):
        yi, yj = py[i], py[j]
        xi, xj = px[i], px[j]
        intersects = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-24) + xi)
        inside ^= intersects
        j = i
    return inside


def signed_distance_to_polygon(points: np.ndarray, polygon: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
    """计算点到闭合多边形的有符号距离，内部为负。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=np.float64)
    if len(poly) < 4:
        raise ValueError("polygon must contain at least three vertices plus closure")
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[:1]])

    out = np.empty(pts.shape[0], dtype=np.float64)
    for start in range(0, pts.shape[0], chunk_size):
        chunk = pts[start : start + chunk_size]
        unsigned = _unsigned_distance_to_polygon(chunk, poly)
        inside = _points_in_polygon(chunk, poly)
        out[start : start + chunk_size] = np.where(inside, -unsigned, unsigned)
    return out


def rasterize_airfoil_sdf(
    resampled: ResampledAirfoil,
    height: int = 128,
    width: int = 256,
    bounds: tuple[float, float, float, float] = (-0.05, 1.05, -0.25, 0.25),
    sdf_scale: float = 0.08,
) -> dict[str, np.ndarray]:
    """将重采样翼型 rasterize 为归一化二维 SDF 与二值内部 mask。"""
    if sdf_scale <= 0:
        raise ValueError("sdf_scale must be positive")
    polygon = build_airfoil_polygon(resampled)
    grid_x, grid_y = make_sdf_grid(height, width, bounds)
    points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    sdf = signed_distance_to_polygon(points, polygon).reshape(height, width)
    sdf_norm = np.clip(sdf / sdf_scale, -1.0, 1.0).astype(np.float32)
    semantic_mask = (sdf < 0).astype(np.uint8)
    return {
        "sdf2d_norm": sdf_norm,
        "semantic_mask": semantic_mask,
        "grid_x": grid_x.astype(np.float32),
        "grid_y": grid_y.astype(np.float32),
    }
