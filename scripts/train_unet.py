"""Train the pure U-Net baseline. All settings come from a YAML config."""
import argparse
import csv
import json
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from lensless.data import load_train_examples, select_split
from lensless.experiment import create_run_dir, git_state, seed_everything, write_run_metadata
from lensless.metrics import image_metrics
from lensless.unet import UNet


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def synthetic_examples(count: int, seed: int) -> list[dict]:
    """Small fake pairs, so tests can run the whole script without a download."""
    rng = np.random.default_rng(seed)
    return [
        {
            "lensless": rng.integers(0, 256, (270, 480, 3), dtype=np.uint8),
            "lensed": rng.integers(0, 256, (270, 480, 3), dtype=np.uint8),
        }
        for _ in range(count)
    ]


def build_datasets(cfg: dict):
    """Both splits come from select_split, the same helper the ADMM and evaluation scripts use."""
    if cfg["source"] == "synthetic":
        examples = synthetic_examples(cfg["synthetic_examples"], cfg["split_seed"])
    else:
        examples = load_train_examples(cfg["cache_dir"], cfg["train_shards"])
    train_set = select_split(examples, "train", cfg["val_size"], cfg["split_seed"], cfg["max_train_examples"])
    # A deliberate overfit check scores the training images, to test memorization.
    if cfg["val_from_train"]:
        return train_set, train_set
    return train_set, select_split(examples, "val", cfg["val_size"], cfg["split_seed"], cfg["max_val_examples"])


def compute_loss(prediction, target, perceptual_loss, lpips_weight: float) -> torch.Tensor:
    loss = F.mse_loss(prediction, target)
    if lpips_weight > 0:
        loss = loss + lpips_weight * perceptual_loss(prediction.clamp(0, 1), target, normalize=True).mean()
    return loss


def train_one_epoch(model, loader, optimizer, perceptual_loss, lpips_weight, device) -> float:
    model.train()
    total_loss, total_images = 0.0, 0
    for lensless, lensed in loader:
        lensless, lensed = lensless.to(device), lensed.to(device)
        loss = compute_loss(model(lensless), lensed, perceptual_loss, lpips_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(lensless)
        total_images += len(lensless)
    return total_loss / total_images


@torch.no_grad()
def evaluate(model, loader, perceptual_loss, lpips_weight, device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_images = 0
    for lensless, lensed in loader:
        lensless, lensed = lensless.to(device), lensed.to(device)
        prediction = model(lensless)
        scores = image_metrics(prediction, lensed, perceptual_loss)
        scores["loss"] = compute_loss(prediction, lensed, perceptual_loss, lpips_weight).item()
        for key, value in scores.items():
            totals[key] = totals.get(key, 0.0) + value * len(lensless)
        total_images += len(lensless)
    return {key: value / total_images for key, value in totals.items()}


@torch.no_grad()
def save_preview(model, dataset, path: Path, device, count: int = 4) -> None:
    """Save input, output and target rows for the same images every time."""
    model.eval()
    pairs = [dataset[i] for i in range(min(count, len(dataset)))]
    lensless = torch.stack([pair[0] for pair in pairs]).to(device)
    lensed = torch.stack([pair[1] for pair in pairs]).to(device)
    rows = [torch.cat(list(batch), dim=2) for batch in (lensless, model(lensless).clamp(0, 1), lensed)]
    grid = (torch.cat(rows, dim=1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(grid).save(path)


def append_metrics(path: Path, row: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, state: dict) -> None:
    """Write then rename, so a disconnect cannot leave a half-written checkpoint."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def warn_if_code_changed(run_dir: Path) -> None:
    recorded = json.loads((run_dir / "run_info.json").read_text())["git"]["commit"]
    current = git_state()["commit"]
    if recorded != current:
        print(f"WARNING: run started at commit {recorded}, resuming at {current}", flush=True)


def build_unet(config: dict, device: str) -> torch.nn.Module:
    return UNet(**config["model"]).to(device)


def run(config: dict, checkpoint_root: str | None = None, experiments_root: str | None = None, resume_run: str | None = None,
        build_model=build_unet, device: str | None = None) -> Path:
    """Train any model that maps a lensless image to a lensed one; Le-ADMM reuses this loop with its own builder."""
    seed_everything(config["train"]["seed"])
    device = device or pick_device()
    exp, train_cfg = config["experiment"], config["train"]
    if resume_run:
        run_dir = Path(resume_run)
        warn_if_code_changed(run_dir)
    else:
        run_dir = create_run_dir(experiments_root or exp["root"], exp["name"], exp["technique"])
        write_run_metadata(run_dir, config, device)
    checkpoint_dir = Path(checkpoint_root) / exp["name"] / run_dir.name if checkpoint_root else run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = build_datasets(config["data"])
    workers = config["data"]["num_workers"]
    train_loader = DataLoader(train_set, train_cfg["batch_size"], shuffle=True, num_workers=workers, drop_last=False)
    val_loader = DataLoader(val_set, train_cfg["batch_size"], shuffle=False, num_workers=workers)
    print(f"run: {run_dir} | device: {device} | train: {len(train_set)} | val: {len(val_set)}", flush=True)

    model = build_model(config, device)
    perceptual_loss = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    best_val_loss = float("inf")
    start_epoch = 1
    if resume_run:
        # The shuffle order differs after resuming, which should not matter across thousands of updates.
        saved = torch.load(checkpoint_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        best_val_loss, start_epoch = saved["best_val_loss"], saved["epoch"] + 1
        print(f"resumed {run_dir} after epoch {saved['epoch']}", flush=True)

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, perceptual_loss, train_cfg["lpips_weight"], device)
        train_seconds = time.perf_counter() - epoch_start
        if epoch % train_cfg["eval_every"] == 0 or epoch == train_cfg["epochs"]:
            val_scores = evaluate(model, val_loader, perceptual_loss, train_cfg["lpips_weight"], device)
            append_metrics(run_dir / "metrics.csv", {"epoch": epoch, "train_loss": train_loss, "train_seconds": train_seconds, **{f"val_{k}": v for k, v in val_scores.items()}})
            save_preview(model, val_set, run_dir / "previews" / f"epoch_{epoch:04d}.png", device)
            print(f"epoch {epoch}: {train_seconds:.0f}s train_loss={train_loss:.4f} val_loss={val_scores['loss']:.4f} screen_psnr={val_scores['screen_psnr']:.2f}", flush=True)
            if val_scores["loss"] < best_val_loss:
                best_val_loss = val_scores["loss"]
                save_checkpoint(checkpoint_dir / "best.pt", {"model": model.state_dict(), "epoch": epoch, "config": config})
        save_checkpoint(checkpoint_dir / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_val_loss": best_val_loss, "config": config})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Start a new run from this config")
    source.add_argument("--resume", help="Continue this run folder, using its saved config")
    parser.add_argument("--experiments-root", help="Create run folders here, such as a Google Drive folder")
    parser.add_argument("--checkpoint-root", help="Store checkpoints here instead of inside the run folder")
    args = parser.parse_args()
    config_path = Path(args.resume) / "config.yaml" if args.resume else Path(args.config)
    run(yaml.safe_load(config_path.read_text()), args.checkpoint_root, args.experiments_root, args.resume)


if __name__ == "__main__":
    main()
