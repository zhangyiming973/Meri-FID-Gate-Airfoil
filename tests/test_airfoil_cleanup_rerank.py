import numpy as np

from scripts.airfoil.evaluate_generated_airfoils import (
    _condition_score,
    _dat_coordinates,
    _sample_sdf_airfoil,
    _write_airfoil_dat,
)
from scripts.airfoil_geometry import ResampledAirfoil


def _airfoil() -> ResampledAirfoil:
    x = np.linspace(0.0, 1.0, 9)
    thickness = 0.12 * np.sin(np.pi * x)
    camber = 0.02 * np.sin(np.pi * x)
    return ResampledAirfoil(x=x, y_upper=camber + thickness / 2, y_lower=camber - thickness / 2)


def test_condition_score_weights_normalized_errors() -> None:
    target = {"t_max": 0.12, "x_tmax": 0.4, "camber_max": 0.02, "x_camber_max": 0.5, "te_gap": 0.0}
    measured_good = {"t_max": 0.121, "x_tmax": 0.41, "camber_max": 0.021, "x_camber_max": 0.51, "te_gap": 0.001}
    measured_bad = {"t_max": 0.16, "x_tmax": 0.7, "camber_max": 0.05, "x_camber_max": 0.9, "te_gap": 0.04}

    assert _condition_score(measured_good, target) < _condition_score(measured_bad, target)


def test_dat_coordinates_start_at_upper_te_and_end_at_lower_te() -> None:
    coords = _dat_coordinates(_airfoil())

    np.testing.assert_allclose(coords[0], [1.0, _airfoil().y_upper[-1]])
    np.testing.assert_allclose(coords[-1], [1.0, _airfoil().y_lower[-1]])
    assert coords.shape == (17, 2)


def test_write_airfoil_dat(tmp_path) -> None:
    out = tmp_path / "candidate.dat"

    _write_airfoil_dat(out, "candidate", _airfoil())

    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "candidate"
    assert len(text) == 18
    assert len(text[1].split()) == 2


def test_sample_sdf_airfoil_ignores_contour_order_spike() -> None:
    x_axis = np.linspace(-0.05, 1.05, 180)
    y_axis = np.linspace(-0.25, 0.25, 120)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    x = np.clip(grid_x, 0.0, 1.0)
    thickness = 0.12 * np.sin(np.pi * x)
    camber = 0.02 * np.sin(np.pi * x)
    y_upper = camber + 0.5 * thickness
    y_lower = camber - 0.5 * thickness
    outside_x = (grid_x < 0.0) | (grid_x > 1.0)
    sdf = np.maximum(grid_y - y_upper, y_lower - grid_y)
    sdf[outside_x] = 0.05

    upper = np.column_stack([np.linspace(1.0, 0.0, 40), np.linspace(0.0, 0.0, 40)])
    lower = np.column_stack([np.linspace(0.0, 1.0, 40), np.linspace(0.0, 0.0, 40)])
    bad_spike = np.array([[0.5, 0.22], [0.51, -0.22]])
    contour = np.vstack([upper[:20], bad_spike, upper[20:], lower])

    sampled = _sample_sdf_airfoil(sdf, grid_x, grid_y, contour, n_points=65)

    assert sampled.thickness.max() > 0.09
    assert sampled.thickness.min() >= -1e-9
    assert np.max(np.abs(np.diff(sampled.y_upper, n=2))) < 0.02
