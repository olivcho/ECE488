# ECE488 Final Project: Lensless Reconstruction Transfer

Tests whether physics-informed reconstruction networks trained on one DiffuserCam generalize to a new camera. See `docs/` for the proposal and `CLAUDE.md` for project conventions.

## Setup
```bash
uv sync
uv run pytest
```

## Train the U-Net baseline
```bash
uv run python scripts/train_unet.py --config configs/unet_overfit.yaml
uv run python scripts/train_unet.py --config configs/unet.yaml
```

## Run the classical ADMM baseline
Scores ADMM with total variation on validation images at several iteration counts. Each setting gets its own run folder.
```bash
uv run python scripts/run_admm.py --config configs/admm_quick.yaml
uv run python scripts/run_admm.py --config configs/admm.yaml
```

## Troubleshooting
If a dataset download stalls with no progress, retry with Hugging Face's Xet downloader turned off:

```bash
HF_HUB_DISABLE_XET=1 uv run python scripts/train_unet.py --config configs/unet_overfit.yaml
```
