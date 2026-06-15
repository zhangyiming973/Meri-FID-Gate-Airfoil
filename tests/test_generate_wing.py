import math

import numpy as np

from scripts.generate_wing import WingInputs, clean_airfoil_loop, closed_airfoil_loop, compute_wing_geometry, make_section


def test_compute_wing_geometry_uses_full_wing_aspect_ratio() -> None:
    geometry = compute_wing_geometry(
        WingInputs(
            root_tip_ratio=2.0,
            aspect_ratio=8.0,
            total_length=5.0,
            dihedral_deg=5.0,
        )
    )

    assert math.isclose(geometry.taper_ratio, 0.5)
    assert math.isclose(geometry.full_area, 12.5)
    assert math.isclose(geometry.half_area, 6.25)
    assert math.isclose(geometry.root_chord, 5.0 / 3.0)
    assert math.isclose(geometry.tip_chord, 5.0 / 6.0)
    assert math.isclose(geometry.projected_span, 5.0 * math.cos(math.radians(5.0)))
    assert math.isclose(geometry.dihedral_height, 5.0 * math.sin(math.radians(5.0)))


def test_closed_airfoil_loop_adds_trailing_closure() -> None:
    coords = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
        ]
    )

    loop = closed_airfoil_loop(coords)

    assert loop.shape == (5, 2)
    np.testing.assert_allclose(loop[0], loop[-1])


def test_make_section_maps_airfoil_to_xyz() -> None:
    coords = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.1],
            [1.0, 0.0],
        ]
    )

    section = make_section(coords, chord=2.0, y=3.0, z_offset=4.0)

    assert section == [(2.0, 3.0, 4.0), (0.0, 3.0, 4.2), (2.0, 3.0, 4.0)]


def test_clean_airfoil_loop_resamples_to_ordered_closed_loop() -> None:
    coords = np.array(
        [
            [1.0, 0.0],
            [0.75, 0.04],
            [0.5, 0.06],
            [0.25, 0.04],
            [0.0, 0.0],
            [0.25, -0.04],
            [0.5, -0.06],
            [0.75, -0.04],
            [1.0, 0.0],
        ]
    )

    loop = clean_airfoil_loop(coords, section_points=33)

    assert loop.shape == (65, 2)
    np.testing.assert_allclose(loop[0], loop[-1])
    assert np.isclose(loop[:, 0].min(), 0.0)
    assert np.isclose(loop[:, 0].max(), 1.0)
