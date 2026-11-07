"""Train Le-ADMM, the unrolled ADMM with learned penalties. Shares the training loop with train_unet.py."""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_unet  # noqa: E402
from run_admm import build_forward_model, pick_device  # noqa: E402

from lensless.le_admm import LeADMM


def build_le_admm(config: dict, device: str) -> torch.nn.Module:
    forward_model = build_forward_model(config["psf"], device)
    return LeADMM(forward_model, **config["model"]).to(device)


def run(config: dict, checkpoint_root: str | None = None, experiments_root: str | None = None, resume_run: str | None = None) -> Path:
    # pick_device from run_admm skips MPS, which lacks the FFT support the solver needs.
    return train_unet.run(config, checkpoint_root, experiments_root, resume_run, build_model=build_le_admm, device=pick_device())


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
