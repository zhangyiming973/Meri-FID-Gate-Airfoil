from pathlib import Path

import numpy as np

from scripts.airfoil_geometry import (
    compute_airfoil_conditions,
    cosine_x_grid,
    normalize_airfoil,
    parse_selig_dat,
    resample_surfaces,
    split_surfaces,
    validate_airfoil,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "airfoil"


def test_parse_selig_dat_skips_non_numeric_lines() -> None:
    coords = parse_selig_dat(FIXTURE_DIR / "simple_cambered.dat")

    assert coords.shape == (9, 2)
    np.testing.assert_allclose(coords[0], [1.0, 0.01])
    np.testing.assert_allclose(coords[4], [0.0, 0.0])


def test_normalize_airfoil_sets_unit_chord() -> None:
    raw = parse_selig_dat(FIXTURE_DIR / "simple_symmetric.dat")
    shifted_scaled = raw * np.array([2.0, 0.5]) + np.array([3.0, -0.1])

    normalized = normalize_airfoil(shifted_scaled)

    assert np.isclose(normalized[:, 0].min(), 0.0)
    assert np.isclose(normalized[:, 0].max(), 1.0)
    leading_edge = normalized[np.argmin(normalized[:, 0])]
    assert abs(leading_edge[1]) < 1e-6


def test_split_and_resample_surfaces_produce_nonnegative_thickness() -> None:
    coords = normalize_airfoil(parse_selig_dat(FIXTURE_DIR / "simple_symmetric.dat"))
    surfaces = split_surfaces(coords)
    x_grid = cosine_x_grid(33)
    resampled = resample_surfaces(surfaces, x_grid)

    assert resampled.x.shape == (33,)
    assert resampled.y_upper.shape == (33,)
    assert resampled.y_lower.shape == (33,)
    assert np.all(resampled.thickness >= -1e-6)
    assert resampled.thickness.max() > 0.1


def test_compute_conditions_for_cambered_airfoil() -> None:
    coords = normalize_airfoil(parse_selig_dat(FIXTURE_DIR / "simple_cambered.dat"))
    resampled = resample_surfaces(split_surfaces(coords), cosine_x_grid(65))

    conditions = compute_airfoil_conditions(resampled)

    assert set(conditions) == {"t_max", "x_tmax", "camber_max", "x_camber_max", "te_gap"}
    assert 0.09 < conditions["t_max"] < 0.13
    assert 0.35 < conditions["x_tmax"] < 0.65
    assert conditions["camber_max"] > 0.02
    assert 0.35 < conditions["x_camber_max"] < 0.85
    assert 0.0 <= conditions["te_gap"] < 0.02
    assert validate_airfoil(resampled, conditions) == []


def test_validate_airfoil_reports_invalid_thickness() -> None:
    coords = normalize_airfoil(parse_selig_dat(FIXTURE_DIR / "simple_symmetric.dat"))
    resampled = resample_surfaces(split_surfaces(coords), cosine_x_grid(17))
    bad = resampled.__class__(
        x=resampled.x,
        y_upper=resampled.y_lower,
        y_lower=resampled.y_upper,
    )

    reasons = validate_airfoil(bad, compute_airfoil_conditions(bad))

    assert "negative_thickness" in reasons
