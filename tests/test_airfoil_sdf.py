from pathlib import Path

import numpy as np

from scripts.airfoil_geometry import (
    cosine_x_grid,
    normalize_airfoil,
    parse_selig_dat,
    resample_surfaces,
    split_surfaces,
)
from scripts.airfoil_sdf import build_airfoil_polygon, rasterize_airfoil_sdf


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "airfoil"


def _simple_airfoil():
    coords = normalize_airfoil(parse_selig_dat(FIXTURE_DIR / "simple_symmetric.dat"))
    return resample_surfaces(split_surfaces(coords), cosine_x_grid(65))


def test_build_airfoil_polygon_is_closed_and_ordered() -> None:
    resampled = _simple_airfoil()

    polygon = build_airfoil_polygon(resampled)

    assert polygon.shape == (129, 2)
    np.testing.assert_allclose(polygon[0], polygon[-1])
    assert np.isclose(polygon[:, 0].min(), 0.0)
    assert np.isclose(polygon[:, 0].max(), 1.0)


def test_rasterize_airfoil_sdf_outputs_expected_arrays() -> None:
    resampled = _simple_airfoil()

    result = rasterize_airfoil_sdf(resampled, height=128, width=256, sdf_scale=0.08)

    assert result["sdf2d_norm"].shape == (128, 256)
    assert result["semantic_mask"].shape == (128, 256)
    assert result["sdf2d_norm"].dtype == np.float32
    assert result["semantic_mask"].dtype == np.uint8
    assert set(np.unique(result["semantic_mask"])) <= {0, 1}
    assert result["sdf2d_norm"].min() >= -1.0
    assert result["sdf2d_norm"].max() <= 1.0


def test_rasterize_airfoil_sdf_signs_inside_and_outside() -> None:
    resampled = _simple_airfoil()

    result = rasterize_airfoil_sdf(
        resampled,
        height=101,
        width=121,
        bounds=(-0.05, 1.05, -0.25, 0.25),
        sdf_scale=0.08,
    )
    sdf = result["sdf2d_norm"]
    x = result["grid_x"]
    y = result["grid_y"]

    center_idx = np.unravel_index(np.argmin((x - 0.5) ** 2 + y**2), x.shape)
    top_idx = np.unravel_index(np.argmin((x - 0.5) ** 2 + (y - 0.22) ** 2), x.shape)

    assert sdf[center_idx] < 0
    assert result["semantic_mask"][center_idx] == 1
    assert sdf[top_idx] > 0
    assert result["semantic_mask"][top_idx] == 0
