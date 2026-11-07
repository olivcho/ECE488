"""Load DiffuserCam image pairs as float tensors."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ID = "bezzam/DiffuserCam-Lensless-Mirflickr-Dataset-NORM"
NUM_TRAIN_SHARDS = 15


def train_shard_paths(num_shards: int) -> list[str]:
    """Name the first shards, so debugging runs avoid the full 7.6 GB download."""
    if not 1 <= num_shards <= NUM_TRAIN_SHARDS:
        raise ValueError(f"num_shards must be 1 to {NUM_TRAIN_SHARDS}, got {num_shards}")
    return [f"data/train-{i:05d}-of-{NUM_TRAIN_SHARDS:05d}.parquet" for i in range(num_shards)]


def load_train_examples(cache_dir: str | Path, num_shards: int | None = None):
    """Load training pairs from their parquet files.

    Reading the files directly has two benefits. A subset no longer fails the checks
    against the dataset card, which expects every split at full size. The test split
    is also never downloaded, so it cannot leak into training.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    shard_paths = train_shard_paths(NUM_TRAIN_SHARDS if num_shards is None else num_shards)
    local_files = [
        hf_hub_download(REPO_ID, shard, repo_type="dataset", cache_dir=str(cache_dir))
        for shard in shard_paths
    ]
    return load_dataset("parquet", data_files={"train": local_files}, split="train", cache_dir=str(cache_dir))


def split_train_val(num_examples: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Carve a validation set from train, because the dataset only has train and test."""
    if not 0 < val_size < num_examples:
        raise ValueError(f"val_size must be between 1 and {num_examples - 1}, got {val_size}")
    order = np.random.default_rng(seed).permutation(num_examples).tolist()
    return sorted(order[val_size:]), sorted(order[:val_size])


def image_to_tensor(image) -> torch.Tensor:
    """Convert an 8-bit H×W×C image to a float C×H×W tensor in [0, 1]."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected H×W×3 image, got shape {array.shape}")
    return torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1)


class LensedPairDataset(Dataset):
    """Return (lensless, lensed) tensors for selected rows."""

    def __init__(self, examples, indices: list[int] | None = None):
        self.examples = examples
        self.indices = list(range(len(examples))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[self.indices[position]]
        return image_to_tensor(example["lensless"]), image_to_tensor(example["lensed"])
