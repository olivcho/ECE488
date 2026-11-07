"""Train the conditional flow-matching model. Shares the training loop with train_unet.py.

Each training step regresses the velocity target − noise at a random t, so
`train_loss` in metrics.csv is a velocity MSE. Validation runs the Euler
sampler and scores the samples like every other model, so `val_*` columns
are directly comparable to the U-Net and Le-ADMM rows.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_unet  # noqa: E402

from lensless.flow_unet import FlowMatching


def build_flow(config: dict, device: str) -> torch.nn.Module:
    return FlowMatching(**config["model"]).to(device)


def flow_train_loss(model, lensless, lensed, perceptual_loss, lpips_weight: float) -> torch.Tensor:
    # lpips_weight is ignored here: a perceptual loss on a velocity field has no meaning.
    # It still shapes the validation loss that picks best.pt, which scores the sampled images.
    return model.loss(lensless, lensed)


def run(config: dict, checkpoint_root: str | None = None, experiments_root: str | None = None, resume_run: str | None = None) -> Path:
    return train_unet.run(config, checkpoint_root, experiments_root, resume_run, build_model=build_flow, train_loss=flow_train_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
