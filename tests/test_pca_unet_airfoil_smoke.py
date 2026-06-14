import torch

from schemes.pca_unet.losses.ae_losses import AELossWeights, compute_ae_losses
from schemes.pca_unet.models.autoencoder import MeridianAutoEncoder
from schemes.pca_unet.pipeline import validate_autoencoder_geometry


def test_validate_autoencoder_geometry_accepts_airfoil_rectangular_config() -> None:
    cfg = {"input_size": [128, 256], "latent_spatial": [8, 16]}

    validate_autoencoder_geometry(cfg)


def test_validate_autoencoder_geometry_rejects_mismatched_config() -> None:
    cfg = {"input_size": [128, 256], "latent_spatial": [16, 16]}

    try:
        validate_autoencoder_geometry(cfg)
    except ValueError as exc:
        assert "input_size / latent_spatial" in str(exc)
    else:
        raise AssertionError("Expected invalid AE geometry to raise ValueError")


def test_airfoil_autoencoder_cpu_single_training_step() -> None:
    torch.manual_seed(0)
    model = MeridianAutoEncoder(1, latent_channels=4, latent_spatial=[8, 16], output_size=[128, 256])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(2, 1, 128, 256)
    target = x.clamp(-1, 1)
    semantic = torch.zeros_like(target)
    weights = AELossWeights(l1=1.0, l2=0.0, zero_level=0.0, dimension=0.0, semantic=0.0)

    _, recon = model(x)
    losses = compute_ae_losses(recon, target, semantic, {}, weights)
    losses["total"].backward()
    opt.step()

    assert recon.shape == target.shape
    assert torch.isfinite(losses["total"])
