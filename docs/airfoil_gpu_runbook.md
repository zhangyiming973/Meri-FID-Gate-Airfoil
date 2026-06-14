# UIUC Airfoil GPU Runbook

This runbook is for the later CUDA machine. The current development machine is CPU-only and should only run unit tests and small smoke checks.

## Environment Checks

Run these before training:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0))
    print("memory_gb", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

Record:

1. CUDA version.
2. PyTorch version.
3. GPU model.
4. GPU memory.
5. Driver version, if available.

## Full Data Preparation

```bash
python run.py prepare-airfoil-uiuc --force
python run.py prepare-splits --dataset airfoil_uiuc_sdf --force
```

Expected outputs:

```text
data/airfoil_uiuc_sdf/processed/airfoil_index.csv
data/airfoil_uiuc_sdf/processed/train_split.csv
data/airfoil_uiuc_sdf/processed/test_split.csv
data/airfoil_uiuc_sdf/processed/condition_stats.json
data/airfoil_uiuc_sdf/processed/filter_report.json
data/airfoil_uiuc_sdf/processed/airfoil_samples/*.npz
```

## AE Training

```bash
python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf --stage ae
```

Record:

1. AE batch size.
2. Peak GPU memory.
3. Epoch time.
4. Final train/validation L1.
5. Whether latent gate passes.

If memory is tight, reduce:

```json
{
  "autoencoder": {
    "batch_size": 8
  }
}
```

## Diffusion Training

Use the AE run directory from the previous step:

```bash
python run.py train \
  --scheme pca_unet \
  --dataset airfoil_uiuc_sdf \
  --stage unet \
  --ae-run-dir outputs/pca_unet/airfoil_uiuc_sdf/<timestamp>/autoencoder
```

Record:

1. UNet batch size.
2. Peak GPU memory.
3. Epoch time.
4. Best `val_gen_l1`.
5. Mean generation L1.

If memory is tight, reduce:

```json
{
  "unet": {
    "batch_size": 16,
    "unet_base_ch": 48
  }
}
```

## Evaluation

```bash
python run.py test --scheme pca_unet --run-dir outputs/pca_unet/airfoil_uiuc_sdf/<timestamp>
```

Inspect:

1. `generation_metrics.json`
2. generation image grid
3. per-sample L1 chart
4. failed or blank generations

## CPU-Only Development Checks

These are safe on the current machine:

```bash
/home/vipuser/miniconda3/bin/python -m pytest tests/test_airfoil_geometry.py
/home/vipuser/miniconda3/bin/python -m pytest tests/test_airfoil_sdf.py
/home/vipuser/miniconda3/bin/python -m pytest tests/test_prepare_airfoil_uiuc.py
/home/vipuser/miniconda3/bin/python -m pytest tests/test_pca_unet_airfoil_dataset.py
/home/vipuser/miniconda3/bin/python -m pytest tests/test_pca_unet_airfoil_smoke.py
/home/vipuser/miniconda3/bin/python -m pytest tests/test_pca_unet_airfoil_diffusion_config.py
```

Do not run full AE or diffusion training on the CPU-only development machine unless using a tiny temporary fixture dataset.
