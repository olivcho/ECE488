# Results

Built from every run folder under `experiments/`. One table per split; rows on different splits or image sets are never compared. Best run per config, at the checkpoint `train_unet.py` saves as `best.pt` (lowest `val_loss`) or, for ADMM, the best-scoring `eval_at` iteration. PSNR in dB (higher is better), SSIM higher is better, LPIPS (AlexNet) lower is better. *Whole* is the full 270×480 frame, *screen* is the monitor crop in `lensless.metrics`.

## Checks

- `unet_overfit_check/261106_unet_v1` has no `metrics.csv`: configured for 400 epochs with `eval_every: 50`, it never reached an evaluation, so it stopped early. `run_info.json` records no stop reason; `261106_unet_v2` reruns the same config.
- All four runs have `uncommitted_changes: true` in `run_info.json`; the U-Net runs record commit `33824ad`, which is not in the repository history.
- U-Net overfit rows score their own 4 training images (`val_from_train: true`), so they sit in the train table and are not comparable with the ADMM val row.
- `admm_quick_check` uses a 1-shard split (`train_shards: 1`, `val_size: 100`), not the full-data split (`val_size: 1000`) used by `admm.yaml` and the learned-model configs; its 8 images are not a subset of the full-data val split.
- No same-split comparison of a learned model with ADMM exists yet, so the "learned model below ADMM" check cannot be run.
- No NaN/inf or flat metrics, no rising `val_loss`, no test-split runs.

## Train split

| Model / config | Run | Split | Images | Epochs (best / run) | Whole PSNR | Whole SSIM | Whole LPIPS | Screen PSNR | Screen SSIM | Screen LPIPS | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net 16ch d4, `unet_overfit.yaml` | [261106_unet_v2](../experiments/unet_overfit_check/261106_unet_v2) | train (scored on training images) | 4 (shard 1, `val_size` 100, seed 488) | 400 / 400 | 25.14 | 0.753 | 0.325 | 23.50 | 0.701 | 0.414 | 2026-11-06 |
| U-Net 16ch d4, `unet_overfit_20ep.yaml` | [261106_unet_v3](../experiments/unet_overfit_check/261106_unet_v3) | train (scored on training images) | 4 (shard 1, `val_size` 100, seed 488) | 20 / 20 | 9.87 | 0.280 | 0.936 | 9.08 | 0.366 | 0.945 | 2026-11-06 |

## Val split

| Model / config | Run | Split | Images | Iterations (best / run) | Whole PSNR | Whole SSIM | Whole LPIPS | Screen PSNR | Screen SSIM | Screen LPIPS | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ADMM-TV τ 1e-3, μ 0.1, `admm_quick.yaml` | [261106_admm_v1](../experiments/admm_quick_check/261106_admm_v1) | val, 1-shard split (not the full-data val split) | 8 (shard 1, `val_size` 100, seed 488, first 8) | 100 / 100 | 14.70 | 0.496 | 0.563 | 13.36 | 0.469 | 0.636 | 2026-11-06 |

## Test split

No runs.
