"""Plot metrics from a finished run, adding a score for its untrained starting model."""
import argparse
import csv
import sys
from pathlib import Path

import lpips
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_unet import build_datasets, evaluate, pick_device

from lensless.experiment import seed_everything
from lensless.unet import UNet

METRICS = [("psnr", "PSNR (dB), higher is better"), ("ssim", "SSIM, higher is better"), ("lpips", "LPIPS, lower is better")]
REGIONS = [("screen", "Screen", "#0B7FB5", "o"), ("whole", "Whole frame", "#C2570C", "s")]


def score_untrained_model(config: dict) -> dict[str, float]:
    # Same seed and construction order as training, so this is the run's starting model.
    seed_everything(config["train"]["seed"])
    device = pick_device()
    _, val_set = build_datasets(config["data"])
    model = UNet(**config["model"]).to(device)
    perceptual_loss = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
    loader = DataLoader(val_set, config["train"]["batch_size"])
    return evaluate(model, loader, perceptual_loss, config["train"]["lpips_weight"], device)


def plot(rows: list[dict], output: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor="white")
    for axis, (metric, label) in zip(axes, METRICS):
        for region, name, color, marker in REGIONS:
            epochs = [row["epoch"] for row in rows]
            values = [row[f"{region}_{metric}"] for row in rows]
            axis.plot(epochs, values, color=color, lw=2, marker=marker, ms=6, label=name)
            axis.annotate(f"{values[-1]:.3g}", (epochs[-1], values[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
        axis.set_title(label, fontsize=11, loc="left")
        axis.set_xlabel("Epoch")
        axis.set_xlim(-10, epochs[-1] + 45)
        axis.grid(True, color="#d3dbe1", lw=1)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    figure.suptitle(title, x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    figure.savefig(output, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    rows = [{"epoch": 0, **score_untrained_model(config)}]
    with (args.run / "metrics.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append({"epoch": int(row["epoch"]), **{k.removeprefix("val_"): float(v) for k, v in row.items() if k.startswith("val_")}})
    for row in rows:
        print(row["epoch"], *(f"{k}={row[k]:.3f}" for k in ("screen_psnr", "screen_ssim", "screen_lpips", "whole_psnr", "whole_ssim", "whole_lpips")))
    plot(rows, args.output, "U-Net overfit check on 4 training images, scored on those same images")


if __name__ == "__main__":
    main()
