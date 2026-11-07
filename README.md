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
Tune the penalties with a grid sweep, one run folder per (τ, μ) point. A rerun skips finished points, and `--set key=value` overrides any config value for a single run:
```bash
uv run python scripts/sweep_admm.py --config configs/admm_sweep.yaml
uv run python scripts/run_admm.py --config configs/admm.yaml --set admm.tau=3e-4 --set data.max_val_examples=null
```
On Colab, use `notebooks/run_admm_colab.ipynb`.

## Train Le-ADMM
Unrolls the same ADMM for 5 iterations with learned τ, μ₁, μ₂, μ₃ per iteration. `le_admm.yaml` adds a U-Net denoiser, `le_admm_plain.yaml` learns only the penalties. The PSF enters through the forward model, not the weights, so a trained network transfers to a new camera by rebuilding the forward model from its PSF.
```bash
uv run python scripts/train_le_admm.py --config configs/le_admm_plain.yaml
```
On Colab, use `notebooks/train_le_admm_colab.ipynb`.

## Troubleshooting
If a dataset download stalls with no progress, retry with Hugging Face's Xet downloader turned off:

```bash
HF_HUB_DISABLE_XET=1 uv run python scripts/train_unet.py --config configs/unet_overfit.yaml
```
