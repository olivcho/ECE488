"""Reconstruct validation images with classical ADMM. All settings come from a YAML config."""
import argparse
import csv
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from lensless.admm import ForwardModel, admm_iterations, objective
from lensless.data import LensedPairDataset, load_train_examples, split_train_val
from lensless.experiment import create_run_dir, seed_everything, write_run_metadata
from lensless.metrics import image_metrics
from lensless.psf import IMAGE_SIZE, load_psf, prepare_psf


def pick_device() -> str:
    # Apple's MPS backend lacks parts of the FFT and complex support the solver needs, so it falls back to the CPU.
    return "cuda" if torch.cuda.is_available() else "cpu"


def synthetic_examples(count: int, seed: int) -> list[dict]:
    """Small fake pairs, so tests can run the whole script without a download."""
    rng = np.random.default_rng(seed)
    return [
        {
            "lensless": rng.integers(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8),
            "lensed": rng.integers(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8),
        }
        for _ in range(count)
    ]


def synthetic_psf(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, *IMAGE_SIZE, generator=generator) ** 8


def build_validation_set(cfg: dict) -> LensedPairDataset:
    if cfg["source"] == "synthetic":
        examples = synthetic_examples(cfg["synthetic_examples"], cfg["split_seed"])
    else:
        examples = load_train_examples(cfg["cache_dir"], cfg["train_shards"])
    _, val_indices = split_train_val(len(examples), cfg["val_size"], cfg["split_seed"])
    if cfg["max_val_examples"]:
        val_indices = val_indices[: cfg["max_val_examples"]]
    return LensedPairDataset(examples, val_indices)


def build_forward_model(cfg: dict, device: str) -> ForwardModel:
    psf = synthetic_psf(cfg["seed"]) if cfg["source"] == "synthetic" else load_psf(cfg["cache_dir"])
    return ForwardModel(prepare_psf(psf).to(device))


def scale_output(recon: torch.Tensor, mode: str) -> torch.Tensor:
    """The dataset scaled every image to full range, so the reconstruction gets the same treatment."""
    if mode == "max":
        return recon / recon.flatten(1).amax(dim=1).clamp_min(1e-8)[:, None, None, None]
    if mode == "none":
        return recon
    raise ValueError(f"unknown output_scaling {mode!r}")


@torch.no_grad()
def evaluate(model: ForwardModel, loader, admm_cfg: dict, perceptual_loss, device: str) -> tuple[dict[int, dict[str, float]], torch.Tensor]:
    """Score the reconstruction at several iteration counts in one pass, since ADMM visits each in turn."""
    tau, mu1, mu2, mu3 = (admm_cfg[key] for key in ("tau", "mu1", "mu2", "mu3"))
    iterations = admm_cfg["iterations"]
    eval_at = sorted(set(admm_cfg["eval_at"]) | {iterations})
    totals: dict[int, dict[str, float]] = {iteration: {} for iteration in eval_at}
    total_images = 0
    first_preview = None
    for lensless, lensed in loader:
        lensless, lensed = lensless.to(device), lensed.to(device)
        start = time.perf_counter()
        for iteration, state in admm_iterations(model, lensless, tau, mu1, mu2, mu3, iterations):
            if iteration not in eval_at:
                continue
            recon = scale_output(model.crop(state.v), admm_cfg["output_scaling"])
            scores = image_metrics(recon, lensed, perceptual_loss)
            scores["objective"] = objective(model, lensless, state.v, tau).mean().item()
            scores["seconds_per_image"] = (time.perf_counter() - start) / len(lensless)
            for key, value in scores.items():
                totals[iteration][key] = totals[iteration].get(key, 0.0) + value * len(lensless)
            if iteration == iterations and first_preview is None:
                first_preview = torch.cat([torch.cat(list(row[:4]), dim=2) for row in (lensless, recon.clamp(0, 1), lensed)], dim=1)
        total_images += len(lensless)
    averaged = {iteration: {key: value / total_images for key, value in scores.items()} for iteration, scores in totals.items()}
    return averaged, first_preview


def save_preview(grid: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path)


def write_metrics(path: Path, rows: dict[int, dict[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = None
        for iteration, scores in rows.items():
            row = {"iteration": iteration, **{f"val_{key}": value for key, value in scores.items()}}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)


def run(config: dict, experiments_root: str | None = None) -> Path:
    seed_everything(config["seed"])
    device = pick_device()
    exp = config["experiment"]
    run_dir = create_run_dir(experiments_root or exp["root"], exp["name"], exp["technique"])
    write_run_metadata(run_dir, config, device)

    val_set = build_validation_set(config["data"])
    loader = DataLoader(val_set, config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
    model = build_forward_model(config["psf"], device)
    perceptual_loss = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
    print(f"run: {run_dir} | device: {device} | val: {len(val_set)}", flush=True)

    scores_by_iteration, preview = evaluate(model, loader, config["admm"], perceptual_loss, device)
    write_metrics(run_dir / "metrics.csv", scores_by_iteration)
    save_preview(preview, run_dir / "previews" / f"iteration_{config['admm']['iterations']:04d}.png")
    for iteration, scores in scores_by_iteration.items():
        print(f"iteration {iteration}: objective={scores['objective']:.4g} screen_psnr={scores['screen_psnr']:.2f} screen_lpips={scores['screen_lpips']:.3f} {scores['seconds_per_image']:.2f}s/image", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiments-root", help="Create run folders here, such as a Google Drive folder")
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.experiments_root)


if __name__ == "__main__":
    main()
