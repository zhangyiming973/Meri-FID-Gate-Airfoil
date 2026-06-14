import json
from pathlib import Path

import numpy as np
import pandas as pd

from schemes.pca_unet.data.dataset import ConditionStats, MeridianSDFDataset
from scripts.split_utils import build_condition_stats, make_train_test_split


AIRFOIL_COLUMNS = ["t_max", "x_tmax", "camber_max", "x_camber_max", "te_gap"]


def _write_airfoil_npz(path: Path, value: float = 0.0) -> None:
    sdf = np.full((128, 256), value, dtype=np.float32)
    semantic = (sdf < 0).astype(np.uint8)
    condition = {
        "t_max": 0.12,
        "x_tmax": 0.35,
        "camber_max": 0.02,
        "x_camber_max": 0.45,
        "te_gap": 0.0,
    }
    np.savez(
        path,
        sdf2d_norm=sdf,
        semantic_mask=semantic,
        condition_json=json.dumps(condition),
        condition_raw=np.array([condition[c] for c in AIRFOIL_COLUMNS], dtype=np.float32),
    )


def test_make_train_test_split_accepts_ratio_for_airfoil_columns(tmp_path: Path) -> None:
    index_path = tmp_path / "airfoil_index.csv"
    df = pd.DataFrame(
        {
            "sample_id": [f"af_{i:03d}" for i in range(20)],
            "passed_quality": [True] * 20,
            "t_max": np.linspace(0.08, 0.18, 20),
            "x_tmax": np.linspace(0.2, 0.4, 20),
            "camber_max": np.linspace(0.0, 0.05, 20),
            "x_camber_max": np.linspace(0.3, 0.6, 20),
            "te_gap": np.zeros(20),
        }
    )
    df.to_csv(index_path, index=False)

    train_df, test_df = make_train_test_split(
        index_path,
        test_size=0.15,
        output_dir=tmp_path,
        condition_columns=AIRFOIL_COLUMNS,
    )

    assert len(train_df) == 17
    assert len(test_df) == 3
    meta = json.loads((tmp_path / "split_meta.json").read_text(encoding="utf-8"))
    assert meta["test_size"] == 3
    assert meta["requested_test_size"] == 0.15


def test_airfoil_dataset_uses_condition_stats_columns_and_single_channel(tmp_path: Path) -> None:
    npz_dir = tmp_path / "airfoil_samples"
    npz_dir.mkdir()
    _write_airfoil_npz(npz_dir / "af_001.npz", value=-0.1)
    df = pd.DataFrame(
        [
            {
                "sample_id": "af_001",
                "npz_file": "af_001.npz",
                "t_max": 0.12,
                "x_tmax": 0.35,
                "camber_max": 0.02,
                "x_camber_max": 0.45,
                "te_gap": 0.0,
            }
        ]
    )
    stats = ConditionStats.from_dataframe(df, AIRFOIL_COLUMNS)

    item = MeridianSDFDataset(df, npz_dir, condition_stats=stats, use_semantic=False)[0]

    assert item["x"].shape == (1, 128, 256)
    assert item["sdf"].shape == (1, 128, 256)
    assert item["semantic"].shape == (1, 128, 256)
    assert item["condition"].shape == (5,)
    np.testing.assert_allclose(item["condition_raw"].numpy(), [0.12, 0.35, 0.02, 0.45, 0.0])
    assert item["physics"] == {}


def test_build_condition_stats_preserves_airfoil_columns() -> None:
    df = pd.DataFrame(
        {
            "t_max": [0.1, 0.2],
            "x_tmax": [0.3, 0.4],
            "camber_max": [0.01, 0.03],
            "x_camber_max": [0.4, 0.5],
            "te_gap": [0.0, 0.01],
        }
    )

    stats = build_condition_stats(df, AIRFOIL_COLUMNS)

    assert stats.columns == AIRFOIL_COLUMNS
    np.testing.assert_allclose(stats.mean, [0.15, 0.35, 0.02, 0.45, 0.005])
