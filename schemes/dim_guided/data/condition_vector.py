"""ConditionVector：从 Excel / NPZ / CSV 读取并校验 5 维尺寸条件。

五维参数与子午面几何设计一一对应，作为潜空间扩散的条件输入：
  hub_r_end_mm, rim_r_start_mm, angle_web_deg, r_trans_bore_web_mm, z_min
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from schemes.dim_guided.data import CONDITION_COLUMNS

# Excel / 参数表中列名别名 -> 标准条件列
FIELD_ALIASES: dict[str, list[str]] = {
    "hub_r_end_mm": ["hub_r_end_mm", "r_bore2_mm", "hub_r_end", "bore_r_end"],
    "rim_r_start_mm": ["rim_r_start_mm", "r_outer2_mm", "rim_r_start", "outer_r_start"],
    "angle_web_deg": ["angle_web_deg", "web_angle_deg", "angle_web"],
    "r_trans_bore_web_mm": ["r_trans_bore_web_mm", "r_trans", "trans_r"],
    "z_min": ["z_min", "z_min_mm", "axial_z_min"],
}

# 物理合理范围（mm / deg），用于单样本校验
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "hub_r_end_mm": (50.0, 200.0),
    "rim_r_start_mm": (150.0, 500.0),
    "angle_web_deg": (0.5, 90.0),
    "r_trans_bore_web_mm": (50.0, 200.0),
    "z_min": (-60.0, 60.0),
}


@dataclass
class ConditionVector:
    """5 维尺寸条件向量。"""

    hub_r_end_mm: float
    rim_r_start_mm: float
    angle_web_deg: float
    r_trans_bore_web_mm: float
    z_min: float
    source: str = ""

    def as_dict(self) -> dict[str, float]:
        return {c: float(getattr(self, c)) for c in CONDITION_COLUMNS}

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, c) for c in CONDITION_COLUMNS], dtype=np.float32)

    @classmethod
    def from_dict(cls, values: dict[str, Any], source: str = "") -> ConditionVector:
        missing = [c for c in CONDITION_COLUMNS if c not in values or values[c] is None or pd.isna(values[c])]
        if missing:
            raise KeyError(f"ConditionVector missing fields: {missing}")
        return cls(
            hub_r_end_mm=float(values["hub_r_end_mm"]),
            rim_r_start_mm=float(values["rim_r_start_mm"]),
            angle_web_deg=float(values["angle_web_deg"]),
            r_trans_bore_web_mm=float(values["r_trans_bore_web_mm"]),
            z_min=float(values["z_min"]),
            source=source,
        )

    def validate_physical(self) -> list[str]:
        """检查单样本物理约束，返回问题列表（空表示通过）。"""
        issues: list[str] = []
        d = self.as_dict()
        if d["hub_r_end_mm"] >= d["rim_r_start_mm"]:
            issues.append("hub_r_end_mm 应小于 rim_r_start_mm")
        if d["r_trans_bore_web_mm"] < d["hub_r_end_mm"] * 0.5:
            issues.append("r_trans_bore_web_mm 相对 hub 半径过小")
        for col, (lo, hi) in PHYSICAL_BOUNDS.items():
            v = d[col]
            if v < lo or v > hi:
                issues.append(f"{col}={v:.4f} 超出合理范围 [{lo}, {hi}]")
        return issues


@dataclass
class ApplicabilityReport:
    """数据集级条件适用性评估报告。"""

    n_samples: int
    columns: list[str]
    per_column_std: dict[str, float]
    per_column_range: dict[str, tuple[float, float]]
    low_variance_columns: list[str]
    missing_rate: dict[str, float]
    physical_violations: int
    applicable: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "columns": self.columns,
            "per_column_std": self.per_column_std,
            "per_column_range": {k: list(v) for k, v in self.per_column_range.items()},
            "low_variance_columns": self.low_variance_columns,
            "missing_rate": self.missing_rate,
            "physical_violations": self.physical_violations,
            "applicable": self.applicable,
            "notes": self.notes,
        }


def _resolve_column(row: pd.Series | dict[str, Any], col: str) -> float | None:
    """从一行数据中按别名解析单个条件列。"""
    data = row if isinstance(row, dict) else row.to_dict()
    for alias in FIELD_ALIASES.get(col, [col]):
        if alias in data and data[alias] is not None and not (isinstance(data[alias], float) and np.isnan(data[alias])):
            if pd.notna(data[alias]):
                return float(data[alias])
    return None


def _read_tabular(path: Path) -> pd.DataFrame:
    """读取 CSV 或 Excel 参数表。"""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular format: {path}")


def load_from_excel(path: Path, sample_id: str | None = None, id_column: str = "sample_id") -> ConditionVector:
    """从 Excel / CSV 参数表读取单条 ConditionVector。

    Args:
        path: .xlsx / .xls / .csv 路径。
        sample_id: 样本 ID；为 None 时取首行。
        id_column: 关联键列名，回退 ``id``。
    """
    df = _read_tabular(path)
    id_col = id_column if id_column in df.columns else ("id" if "id" in df.columns else df.columns[0])
    if sample_id is not None:
        rows = df[df[id_col].astype(str) == str(sample_id)]
        if rows.empty:
            raise KeyError(f"sample_id '{sample_id}' not found in {path}")
        row = rows.iloc[0]
    else:
        row = df.iloc[0]
    values: dict[str, float] = {}
    for col in CONDITION_COLUMNS:
        v = _resolve_column(row, col)
        if v is not None:
            values[col] = v
    return ConditionVector.from_dict(values, source=f"excel:{path.name}")


def _derive_from_engineering_curves(curves: dict[str, Any]) -> dict[str, float] | None:
    """从 NPZ 内 engineering_curves_json 推导条件（无 condition_json 时的回退）。"""
    try:
        regions = curves.get("semantic_regions") or curves.get("regions")
        if not regions:
            return None
        if isinstance(regions, list):
            region_map = {r["name"]: r for r in regions}
        else:
            region_map = regions
        hub = region_map.get("hub") or region_map.get("bore")
        rim = region_map.get("rim") or region_map.get("outer")
        trans = region_map.get("hub_web_transition") or region_map.get("transition")
        if not hub or not rim:
            return None
        hub_end = float(hub["r_range_mm"][1] if "r_range_mm" in hub else hub["r_end_mm"])
        rim_start = float(rim["r_range_mm"][0] if "r_range_mm" in rim else rim["r_start_mm"])
        r_trans = float(trans["r_range_mm"][0]) if trans else hub_end
        z_width = float(curves.get("z_width_mm", abs(float(curves.get("z_max", 20)) - float(curves.get("z_min", -20)))))
        z_min = float(curves.get("z_min", -z_width / 2.0))
        angle = float(np.degrees(np.arctan2(rim_start - hub_end, max(z_width, 1e-6))))
        return {
            "hub_r_end_mm": hub_end,
            "rim_r_start_mm": rim_start,
            "angle_web_deg": angle,
            "r_trans_bore_web_mm": r_trans,
            "z_min": z_min,
        }
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def load_from_npz(npz_path: Path) -> ConditionVector:
    """从 NPZ 读取 ConditionVector。

    优先级：``condition_json`` → ``engineering_curves_json`` 推导。
    """
    data = np.load(npz_path, allow_pickle=True)
    if "condition_json" in data:
        cond = json.loads(str(data["condition_json"]))
        return ConditionVector.from_dict(cond, source=f"npz:condition_json:{npz_path.name}")

    if "engineering_curves_json" in data:
        curves = json.loads(str(data["engineering_curves_json"]))
        derived = _derive_from_engineering_curves(curves)
        if derived:
            return ConditionVector.from_dict(derived, source=f"npz:engineering_curves:{npz_path.name}")

    raise KeyError(f"NPZ {npz_path} has no condition_json or derivable engineering_curves_json")


def load_condition_vector(
    sample_id: str,
    index_row: pd.Series | None = None,
    param_table: pd.DataFrame | None = None,
    param_table_path: Path | None = None,
    npz_path: Path | None = None,
) -> ConditionVector:
    """多级回退加载 ConditionVector。

    优先级：索引行 → 内存参数表 → Excel/CSV 文件 → NPZ。
    """
    values: dict[str, float] = {}
    if index_row is not None:
        for col in CONDITION_COLUMNS:
            v = _resolve_column(index_row, col)
            if v is not None:
                values[col] = v
    if len(values) == len(CONDITION_COLUMNS):
        return ConditionVector.from_dict(values, source="index_row")

    if param_table is not None:
        id_col = "id" if "id" in param_table.columns else (
            "sample_id" if "sample_id" in param_table.columns else param_table.columns[0]
        )
        rows = param_table[param_table[id_col].astype(str) == str(sample_id)]
        if not rows.empty:
            for col in CONDITION_COLUMNS:
                v = _resolve_column(rows.iloc[0], col)
                if v is not None:
                    values[col] = v
            if len(values) == len(CONDITION_COLUMNS):
                return ConditionVector.from_dict(values, source="param_table")

    if param_table_path and param_table_path.exists():
        return load_from_excel(param_table_path, sample_id)

    if npz_path and npz_path.exists():
        return load_from_npz(npz_path)

    missing = [c for c in CONDITION_COLUMNS if c not in values]
    raise KeyError(f"Missing condition for {sample_id}: {missing}")


def evaluate_applicability(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    min_std_ratio: float = 0.01,
    min_abs_std: float = 1e-6,
) -> ApplicabilityReport:
    """评估条件列是否适用于尺寸引导扩散。

    判定逻辑：
    - 各列缺失率应为 0
    - 训练集内标准差过小（近常量）的列不适合作为条件
    - 几何物理约束违反样本计数

    Args:
        df: 含条件列的样本索引表。
        columns: 条件列名，默认 CONDITION_COLUMNS。
        min_std_ratio: 相对均值的标准差下限比例。
        min_abs_std: 绝对标准差下限。
    """
    cols = columns or CONDITION_COLUMNS
    n = len(df)
    notes: list[str] = []
    missing_rate: dict[str, float] = {}
    per_std: dict[str, float] = {}
    per_range: dict[str, tuple[float, float]] = {}
    low_var: list[str] = []

    for col in cols:
        if col not in df.columns:
            missing_rate[col] = 1.0
            low_var.append(col)
            continue
        series = df[col].astype(float)
        missing_rate[col] = float(series.isna().mean())
        valid = series.dropna()
        if len(valid) == 0:
            low_var.append(col)
            per_std[col] = 0.0
            per_range[col] = (float("nan"), float("nan"))
            continue
        std = float(valid.std())
        mean = float(valid.mean())
        per_std[col] = std
        per_range[col] = (float(valid.min()), float(valid.max()))
        rel = std / (abs(mean) + 1e-8)
        if std < min_abs_std or rel < min_std_ratio:
            low_var.append(col)
            notes.append(f"{col}: std={std:.6g} 过小，扩散条件区分度不足")

    phys_violations = 0
    for _, row in df.iterrows():
        try:
            cv = ConditionVector.from_dict({c: row[c] for c in cols if c in row.index and pd.notna(row[c])})
            if cv.validate_physical():
                phys_violations += 1
        except KeyError:
            phys_violations += 1

    applicable = (
        n > 0
        and all(missing_rate.get(c, 1.0) == 0.0 for c in cols)
        and len(low_var) < len(cols)
        and phys_violations < n * 0.5
    )
    if low_var and applicable:
        notes.append(f"部分列方差偏低 ({low_var})，CFG 仍可训练但条件控制可能较弱")
    if not applicable and n == 0:
        notes.append("样本数为 0")

    return ApplicabilityReport(
        n_samples=n,
        columns=cols,
        per_column_std=per_std,
        per_column_range=per_range,
        low_variance_columns=low_var,
        missing_rate=missing_rate,
        physical_violations=phys_violations,
        applicable=applicable,
        notes=notes,
    )
