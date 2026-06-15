"""Generate a one-sided tapered wing from a Selig/UIUC airfoil.

The geometry backend is CadQuery, which uses OpenCascade underneath.  The
parameter calculation and section generation are kept dependency-light so they
can be tested without a CAD runtime.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.airfoil_geometry import cosine_x_grid, normalize_airfoil, parse_selig_dat, resample_surfaces, split_surfaces


DEFAULT_EXAMPLE_AIRFOILS = [
    Path("data/airfoil/raw/uiuc/coord_seligFmt/coord_seligFmt/n0012.dat"),
    Path("data/airfoil/raw/uiuc/coord_seligFmt/coord_seligFmt/naca2412.dat"),
    Path("data/airfoil/raw/uiuc/coord_seligFmt/coord_seligFmt/s1223.dat"),
]


@dataclass(frozen=True)
class WingInputs:
    """User-facing planform inputs.

    ``total_length`` is interpreted as one-sided span length, measured along the
    span before applying dihedral. ``aspect_ratio`` follows the conventional
    full-wing definition, using full span and total wing area.
    """

    root_tip_ratio: float
    aspect_ratio: float
    total_length: float
    dihedral_deg: float


@dataclass(frozen=True)
class WingGeometry:
    root_chord: float
    tip_chord: float
    semi_span: float
    projected_span: float
    dihedral_height: float
    half_area: float
    full_area: float
    aspect_ratio: float
    root_tip_ratio: float
    taper_ratio: float
    dihedral_deg: float


def compute_wing_geometry(inputs: WingInputs) -> WingGeometry:
    """Compute trapezoidal one-sided wing dimensions from planform inputs."""
    if inputs.root_tip_ratio <= 0.0:
        raise ValueError("root_tip_ratio must be positive")
    if inputs.aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be positive")
    if inputs.total_length <= 0.0:
        raise ValueError("total_length must be positive")

    semi_span = inputs.total_length
    taper_ratio = 1.0 / inputs.root_tip_ratio
    full_span = 2.0 * semi_span
    full_area = full_span * full_span / inputs.aspect_ratio
    half_area = 0.5 * full_area
    root_chord = 2.0 * half_area / (semi_span * (1.0 + taper_ratio))
    tip_chord = taper_ratio * root_chord

    dihedral_rad = math.radians(inputs.dihedral_deg)
    projected_span = semi_span * math.cos(dihedral_rad)
    dihedral_height = semi_span * math.sin(dihedral_rad)

    return WingGeometry(
        root_chord=root_chord,
        tip_chord=tip_chord,
        semi_span=semi_span,
        projected_span=projected_span,
        dihedral_height=dihedral_height,
        half_area=half_area,
        full_area=full_area,
        aspect_ratio=inputs.aspect_ratio,
        root_tip_ratio=inputs.root_tip_ratio,
        taper_ratio=taper_ratio,
        dihedral_deg=inputs.dihedral_deg,
    )


def load_airfoil(path: Path, section_points: int = 161) -> np.ndarray:
    """Load, normalize and resample an airfoil into a clean closed loop."""
    if section_points < 17:
        raise ValueError("section_points must be at least 17")
    coords = normalize_airfoil(parse_selig_dat(path))
    return clean_airfoil_loop(coords, section_points=section_points)


def clean_airfoil_loop(coords: np.ndarray, section_points: int = 161) -> np.ndarray:
    """Build a CAD-friendly loop by resampling upper/lower surfaces.

    Raw generated airfoils can contain local point spikes.  Splitting into upper
    and lower surfaces and interpolating on a shared cosine grid removes point
    ordering noise while preserving the section shape closely enough for CAD
    lofting.
    """
    x_grid = cosine_x_grid(section_points)
    resampled = resample_surfaces(split_surfaces(coords), x_grid)

    upper = np.column_stack([resampled.x, resampled.y_upper])
    lower = np.column_stack([resampled.x, resampled.y_lower])
    loop = np.vstack([upper[::-1], lower[1:]])
    return closed_airfoil_loop(loop)


def closed_airfoil_loop(coords: np.ndarray) -> np.ndarray:
    """Return a finite, closed coordinate loop suitable for loft sections."""
    arr = np.asarray(coords, dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) < 3:
        raise ValueError("airfoil must contain at least three finite points")
    keep = [0]
    for i in range(1, len(arr)):
        if np.linalg.norm(arr[i] - arr[keep[-1]]) > 1e-10:
            keep.append(i)
    arr = arr[keep]
    if np.linalg.norm(arr[0] - arr[-1]) > 1e-10:
        arr = np.vstack([arr, arr[0]])
    return arr


def make_section(coords: np.ndarray, chord: float, y: float, z_offset: float) -> list[tuple[float, float, float]]:
    """Scale a normalized 2D airfoil into a 3D section.

    Axis convention:
    X is chordwise, Y is spanwise, Z is thickness/vertical.
    """
    if chord <= 0.0:
        raise ValueError("chord must be positive")
    loop = closed_airfoil_loop(coords)
    return [(float(x * chord), float(y), float(z * chord + z_offset)) for x, z in loop]


def _wire_from_points(points: Sequence[tuple[float, float, float]]):
    import cadquery as cq

    vectors = [cq.Vector(*point) for point in points]
    if vectors[0].sub(vectors[-1]).Length > 1e-9:
        vectors.append(vectors[0])
    edges = []
    for start, end in zip(vectors[:-1], vectors[1:]):
        if start.sub(end).Length > 1e-9:
            edges.append(cq.Edge.makeLine(start, end))
    return cq.Wire.assembleEdges(edges)


def build_wing_solid(coords: np.ndarray, geometry: WingGeometry):
    """Build a lofted CadQuery solid for the one-sided wing."""
    import cadquery as cq

    root_section = make_section(coords, geometry.root_chord, 0.0, 0.0)
    tip_section = make_section(coords, geometry.tip_chord, geometry.projected_span, geometry.dihedral_height)
    root_wire = _wire_from_points(root_section)
    tip_wire = _wire_from_points(tip_section)
    return cq.Solid.makeLoft([root_wire, tip_wire], ruled=True)


def export_step(coords: np.ndarray, geometry: WingGeometry, output: Path) -> None:
    """Export the wing as STEP using CadQuery/OpenCascade."""
    import cadquery as cq

    output.parent.mkdir(parents=True, exist_ok=True)
    solid = build_wing_solid(coords, geometry)
    cq.exporters.export(solid, str(output))


def write_metadata(
    path: Path,
    airfoil: Path,
    geometry: WingGeometry,
    step_output: Path | None,
    cadquery_used: bool,
    section_points: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "airfoil": str(airfoil),
        "step_output": str(step_output) if step_output else None,
        "cadquery_used": cadquery_used,
        "section_points": section_points,
        "geometry": asdict(geometry),
        "conventions": {
            "total_length": "one-sided span before dihedral",
            "aspect_ratio": "full-wing AR = full_span^2 / full_area",
            "root_tip_ratio": "root_chord / tip_chord",
            "axes": "X chordwise, Y spanwise, Z vertical/thickness",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_taper(args: argparse.Namespace) -> float:
    if args.root_tip_ratio is not None and args.taper_ratio is not None:
        raise ValueError("Use either --root-tip-ratio or --taper-ratio, not both")
    if args.root_tip_ratio is not None:
        return float(args.root_tip_ratio)
    if args.taper_ratio is not None:
        if args.taper_ratio <= 0.0:
            raise ValueError("--taper-ratio must be positive")
        return 1.0 / float(args.taper_ratio)
    return 2.0


def default_output_for_airfoil(output_dir: Path, airfoil: Path) -> Path:
    return output_dir / f"{airfoil.stem}_wing.step"


def generate_one(airfoil: Path, args: argparse.Namespace) -> WingGeometry:
    root_tip_ratio = resolve_taper(args)
    inputs = WingInputs(
        root_tip_ratio=root_tip_ratio,
        aspect_ratio=args.aspect_ratio,
        total_length=args.total_length,
        dihedral_deg=args.dihedral,
    )
    geometry = compute_wing_geometry(inputs)
    coords = load_airfoil(airfoil, section_points=args.section_points)

    output = args.output if args.output else default_output_for_airfoil(args.output_dir, airfoil)
    metadata = args.metadata if args.metadata else output.with_suffix(".metadata.json")

    cadquery_used = False
    if not args.no_step:
        try:
            export_step(coords, geometry, output)
            cadquery_used = True
        except ModuleNotFoundError as exc:
            if exc.name != "cadquery":
                raise
            raise RuntimeError(
                "CadQuery is required for STEP export. Install cadquery or rerun with --no-step "
                "to write metadata only."
            ) from exc
    write_metadata(metadata, airfoil, geometry, None if args.no_step else output, cadquery_used, args.section_points)
    return geometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one-sided tapered wings from Selig/UIUC airfoil DAT files.")
    parser.add_argument("--airfoil", type=Path, help="Selig/UIUC airfoil .dat file.")
    parser.add_argument(
        "--example-three",
        action="store_true",
        help="Generate the three selected example airfoils from the current pca_unet output directory.",
    )
    parser.add_argument("--root-tip-ratio", type=float, help="Root chord divided by tip chord. Defaults to 2.0.")
    parser.add_argument("--taper-ratio", type=float, help="Tip chord divided by root chord; alternative to --root-tip-ratio.")
    parser.add_argument("--aspect-ratio", type=float, required=True, help="Conventional full-wing aspect ratio.")
    parser.add_argument("--total-length", type=float, required=True, help="One-sided span length before applying dihedral.")
    parser.add_argument("--dihedral", type=float, default=0.0, help="Dihedral angle in degrees.")
    parser.add_argument("--output", type=Path, help="STEP output path for a single --airfoil run.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wing_3d_clean"), help="Output directory for examples.")
    parser.add_argument("--metadata", type=Path, help="Metadata JSON path for a single --airfoil run.")
    parser.add_argument("--section-points", type=int, default=161, help="Cosine-spaced points per surface before lofting.")
    parser.add_argument("--no-step", action="store_true", help="Skip STEP export and write metadata only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.airfoil) == bool(args.example_three):
        parser.error("Specify exactly one of --airfoil or --example-three")
    if args.output and args.example_three:
        parser.error("--output can only be used with --airfoil")
    if args.metadata and args.example_three:
        parser.error("--metadata can only be used with --airfoil")

    airfoils = DEFAULT_EXAMPLE_AIRFOILS if args.example_three else [args.airfoil]
    for airfoil in airfoils:
        if airfoil is None:
            continue
        geometry = generate_one(airfoil, args)
        print(
            f"{airfoil}: root_chord={geometry.root_chord:.6g}, "
            f"tip_chord={geometry.tip_chord:.6g}, projected_span={geometry.projected_span:.6g}, "
            f"dihedral_height={geometry.dihedral_height:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
