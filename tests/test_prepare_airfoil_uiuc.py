import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_airfoil_uiuc import prepare_airfoil_uiuc


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "airfoil"


def test_prepare_airfoil_uiuc_writes_index_samples_and_report(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "simple_symmetric.dat").write_text(
        (FIXTURE_DIR / "simple_symmetric.dat").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (raw_dir / "simple_cambered.dat").write_text(
        (FIXTURE_DIR / "simple_cambered.dat").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (raw_dir / "bad.dat").write_text("bad fixture\n0 0\n", encoding="utf-8")

    summary = prepare_airfoil_uiuc(raw_dir=raw_dir, output_dir=out_dir, height=64, width=96, force=True)

    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1

    index_path = out_dir / "airfoil_index.csv"
    report_path = out_dir / "filter_report.json"
    assert index_path.exists()
    assert report_path.exists()

    index_df = pd.read_csv(index_path)
    assert len(index_df) == 3
    assert {
        "sample_id",
        "npz_file",
        "source_file",
        "passed_quality",
        "t_max",
        "x_tmax",
        "camber_max",
        "x_camber_max",
        "te_gap",
    } <= set(index_df.columns)

    passed = index_df[index_df["passed_quality"].astype(str).str.lower() == "true"]
    assert len(passed) == 2
    sample_npz = out_dir / "airfoil_samples" / str(passed.iloc[0]["npz_file"])
    assert sample_npz.exists()
    data = np.load(sample_npz, allow_pickle=True)
    assert data["sdf2d_norm"].shape == (64, 96)
    assert data["semantic_mask"].shape == (64, 96)
    assert data["coords_raw"].shape[1] == 2
    assert data["coords_resampled"].shape[0] == 2
    assert data["condition_raw"].shape == (5,)
    condition_json = json.loads(str(data["condition_json"]))
    assert set(condition_json) == {"t_max", "x_tmax", "camber_max", "x_camber_max", "te_gap"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"] == summary
    assert any(item["sample_id"] == "bad" and item["reasons"] for item in report["items"])
