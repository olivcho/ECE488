"""Load the DiffuserCam point spread function and match it to the dataset images."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lensless.data import REPO_ID

IMAGE_SIZE = (270, 480)


def load_psf(cache_dir: str | Path) -> torch.Tensor:
    """Download psf.tiff and return it as a float C×H×W tensor at the sensor's full resolution."""
    import tifffile
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, "psf.tiff", repo_type="dataset", cache_dir=str(cache_dir))
    psf = np.asarray(tifffile.imread(path), dtype=np.float32)
    if psf.ndim == 2:
        psf = psf[..., None]
    return torch.from_numpy(psf).permute(2, 0, 1)


def prepare_psf(psf: torch.Tensor, size: tuple[int, int] = IMAGE_SIZE) -> torch.Tensor:
    """Resize, remove the background and scale, so that h * v is a local average of v.

    The dataset images are the raw sensor downsampled by 4, so the PSF must be box-averaged the same
    way or the convolution would run at the wrong scale. The median estimates the sensor's dark level:
    the PSF covers a small fraction of the frame, so most pixels only see background. A unit sum per
    channel keeps |H(f)| at most 1, which keeps the measurement in the same range as the scene.
    """
    if psf.ndim != 3:
        raise ValueError(f"expected C×H×W PSF, got shape {tuple(psf.shape)}")
    resized = F.interpolate(psf[None], size=size, mode="area")[0] if tuple(psf.shape[-2:]) != tuple(size) else psf
    background = resized.flatten(1).median(dim=1).values[:, None, None]
    cleaned = (resized - background).clamp_min(0.0)
    total = cleaned.sum(dim=(-2, -1), keepdim=True)
    if (total <= 0).any():
        raise ValueError("PSF has no signal above the background in at least one channel")
    return cleaned / total
