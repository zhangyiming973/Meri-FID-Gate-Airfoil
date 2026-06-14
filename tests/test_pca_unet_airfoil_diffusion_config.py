import json
from pathlib import Path

import numpy as np
import torch

from schemes.pca_unet.data.dataset import ConditionStats
from schemes.pca_unet.models.autoencoder import MeridianAutoEncoder
from schemes.pca_unet.records.body_latent import BodyLatentRecord, save_records
from schemes.pca_unet.train.train_diffusion import (
    build_diffusion_train_loader,
    build_pca_unet_for_conditions,
    load_autoencoder_from_checkpoint,
)


AIRFOIL_COLUMNS = ["t_max", "x_tmax", "camber_max", "x_camber_max", "te_gap"]


def _record(i: int) -> BodyLatentRecord:
    z = np.full((4, 8, 16), i / 10.0, dtype=np.float32)
    cond_raw = np.array([0.12, 0.35, 0.02, 0.45, 0.0], dtype=np.float32)
    cond_norm = cond_raw + i * 0.01
    return BodyLatentRecord(
        sample_id=f"af_{i:03d}",
        z_m=z,
        condition_raw=cond_raw,
        condition_norm=cond_norm,
        recon_l1=0.0,
        recon_l2=0.0,
        zero_band_l1=0.0,
        dim_error=0.0,
        semantic_error=0.0,
    )


def test_load_autoencoder_from_checkpoint_uses_saved_rectangular_geometry(tmp_path: Path) -> None:
    model = MeridianAutoEncoder(1, latent_channels=4, latent_spatial=[8, 16], output_size=[128, 256])
    ckpt_path = tmp_path / "best_autoencoder.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "in_channels": 1,
            "input_size": [128, 256],
            "latent_spatial": [8, 16],
        },
        ckpt_path,
    )

    loaded = load_autoencoder_from_checkpoint(
        ckpt_path,
        {"latent_channels": 4, "latent_spatial": 16, "input_size": [256, 256]},
        torch.device("cpu"),
    )

    assert loaded.output_size == (128, 256)
    assert loaded.latent_spatial == (8, 16)
    z = torch.randn(1, 4, 8, 16)
    assert loaded.decode(z).shape == (1, 1, 128, 256)


def test_build_pca_unet_for_conditions_uses_condition_stats_columns() -> None:
    stats = ConditionStats(
        mean=np.zeros(5, dtype=np.float32),
        std=np.ones(5, dtype=np.float32),
        columns=AIRFOIL_COLUMNS,
    )

    unet, grid = build_pca_unet_for_conditions(pca_dim=32, cond_stats=stats, base_ch=8)

    assert unet.cond_dim == 5
    assert grid.dim == 32
    x = torch.randn(2, grid.channels, grid.height, grid.width)
    t = torch.randint(0, 10, (2,))
    cond = torch.randn(2, 5)
    assert unet(x, t, cond).shape == x.shape


def test_build_diffusion_train_loader_from_fake_latent_records(tmp_path: Path) -> None:
    records = [_record(i) for i in range(4)]
    save_records(records, tmp_path / "body_latent_train.json")
    z_train_np = np.stack([r.z_m for r in records])

    loader = build_diffusion_train_loader(records, z_train_np, pca_dim=3, batch_size=2)
    z_batch, c_batch = next(iter(loader))

    assert z_batch.shape[0] == 2
    assert c_batch.shape == (2, 5)
    assert z_batch.ndim == 4


def test_condition_stats_file_keeps_airfoil_columns(tmp_path: Path) -> None:
    stats = ConditionStats(
        mean=np.zeros(5, dtype=np.float32),
        std=np.ones(5, dtype=np.float32),
        columns=AIRFOIL_COLUMNS,
    )
    path = tmp_path / "condition_stats.json"
    path.write_text(json.dumps(stats.to_dict()), encoding="utf-8")

    loaded = ConditionStats.from_dict(json.loads(path.read_text(encoding="utf-8")))

    assert loaded.columns == AIRFOIL_COLUMNS
