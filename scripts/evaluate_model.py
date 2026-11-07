"""Score a trained U-Net or Le-ADMM checkpoint on one data split, and test transfer to a new camera.

Four conditions, each a row in metrics.csv:
  real                  the dataset's own measurements, the training camera
  simulated_same_psf    measurements simulated from the lensed images with the training PSF, to size the simulation gap
  new_psf_no_swap       measurements simulated with a second PSF, model unchanged: what happens with no transfer mechanism
  new_psf_swapped       the same measurements after handing Le-ADMM the second PSF; the U-Net has nothing to swap
"""
import argparse
import sys
from pathlib import Path

import lpips
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_admm import apply_overrides, build_eval_set, build_forward_model, parse_override, pick_device, save_preview, synthetic_psf  # noqa: E402

from lensless.admm import ForwardModel, simulate_measurement
from lensless.experiment import create_run_dir, seed_everything, write_run_metadata
from lensless.le_admm import LeADMM
from lensless.metrics import image_metrics
from lensless.psf import load_psf_file, prepare_psf, transform_psf
from lensless.unet import UNet


def load_model(checkpoint_path: str, forward_model: ForwardModel, device: str) -> tuple[torch.nn.Module, str]:
    """Rebuild the model from the config saved inside the checkpoint, so the architecture always matches the weights."""
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    technique = saved["config"]["experiment"]["technique"]
    if technique == "unet":
        model = UNet(**saved["config"]["model"])
    elif technique == "le_admm":
        model = LeADMM(forward_model, **saved["config"]["model"])
    else:
        raise ValueError(f"cannot evaluate technique {technique!r}")
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), technique


def build_new_forward_model(cfg: dict, training_model: ForwardModel, device: str) -> ForwardModel:
    """A second camera: another PSF file, or the training PSF rearranged when no second file exists."""
    if cfg.get("path"):
        psf = prepare_psf(load_psf_file(cfg["path"]))
    elif cfg.get("transform"):
        psf = transform_psf(prepare_psf_from_model(training_model), cfg["transform"])
    elif cfg.get("synthetic_seed") is not None:
        psf = prepare_psf(synthetic_psf(cfg["synthetic_seed"]))
    else:
        raise ValueError("new_psf needs a path, a transform or a synthetic_seed")
    return ForwardModel(psf.to(device))


def prepare_psf_from_model(model: ForwardModel) -> torch.Tensor:
    """Recover the prepared PSF from the forward model's spectrum, so transforms act on exactly what was used."""
    full = torch.fft.irfft2(model.psf_spectrum, s=model.full_size)
    return model.crop(torch.fft.fftshift(full, dim=(-2, -1)))


@torch.no_grad()
def score_condition(model, loader, measure, perceptual_loss, device, preview_path: Path) -> dict[str, float]:
    """`measure` maps a (lensless, lensed) batch to the measurement this condition feeds the model."""
    totals: dict[str, float] = {}
    total_images = 0
    for index, (lensless, lensed) in enumerate(loader):
        lensless, lensed = lensless.to(device), lensed.to(device)
        measurement = measure(lensless, lensed)
        prediction = model(measurement)
        for key, value in image_metrics(prediction, lensed, perceptual_loss).items():
            totals[key] = totals.get(key, 0.0) + value * len(lensless)
        total_images += len(lensless)
        if index == 0:
            grid = torch.cat([torch.cat(list(row[:4]), dim=2) for row in (measurement, prediction.clamp(0, 1), lensed)], dim=1)
            save_preview(grid, preview_path)
    return {key: value / total_images for key, value in totals.items()}


def run(config: dict, experiments_root: str | None = None) -> Path:
    seed_everything(config["seed"])
    device = pick_device()
    exp = config["experiment"]
    run_dir = create_run_dir(experiments_root or exp["root"], exp["name"], exp["technique"])
    write_run_metadata(run_dir, config, device)

    split = config["data"]["split"]
    loader = DataLoader(build_eval_set(config["data"]), config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
    training_camera = build_forward_model(config["psf"], device)
    model, technique = load_model(config["checkpoint"], training_camera, device)
    perceptual_loss = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
    print(f"run: {run_dir} | device: {device} | {technique} on {split}: {len(loader.dataset)}", flush=True)

    swap = config["psf_swap"]
    new_camera = build_new_forward_model(swap["new_psf"], training_camera, device)
    noise = swap["noise_std"]

    def simulated(camera):
        # One generator per condition keeps the noise identical for the swapped and unswapped rows.
        generator = torch.Generator(device=device).manual_seed(config["seed"])
        return lambda lensless, lensed: simulate_measurement(camera, lensed, noise, generator)

    conditions = [
        ("real", lambda lensless, lensed: lensless, None),
        ("simulated_same_psf", simulated(training_camera), None),
        ("new_psf_no_swap", simulated(new_camera), None),
    ]
    if technique == "le_admm":
        conditions.append(("new_psf_swapped", simulated(new_camera), new_camera))

    rows = []
    for name, measure, camera in conditions:
        if camera is not None:
            model.set_forward_model(camera)
        scores = score_condition(model, loader, measure, perceptual_loss, device, run_dir / "previews" / f"{name}.png")
        rows.append({"condition": name, **{f"{split}_{key}": value for key, value in scores.items()}})
        print(f"{name}: screen_psnr={scores['screen_psnr']:.2f} screen_lpips={scores['screen_lpips']:.3f}", flush=True)
    if technique == "le_admm":
        model.set_forward_model(training_camera)
    write_rows(run_dir / "metrics.csv", rows)
    return run_dir


def write_rows(path: Path, rows: list[dict]) -> None:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="best.pt or last.pt from a training run; overrides the config's checkpoint")
    parser.add_argument("--experiments-root", help="Create run folders here, such as a Google Drive folder")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config value, such as data.split=test")
    args = parser.parse_args()
    overrides = dict(map(parse_override, args.set))
    if args.checkpoint:
        overrides["checkpoint"] = args.checkpoint
    run(apply_overrides(yaml.safe_load(Path(args.config).read_text()), overrides), args.experiments_root)


if __name__ == "__main__":
    main()
