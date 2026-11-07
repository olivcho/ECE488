"""Load the DiffuserCam point spread function and match it to the dataset images."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lensless.data import REPO_ID

IMAGE_SIZE = (270, 480)


PSF_TRANSFORMS = ("rotate180", "flip_horizontal", "flip_vertical")


def load_psf(cache_dir: str | Path) -> torch.Tensor:
    """Download psf.tiff and return it as a float C×H×W tensor at the sensor's full resolution."""
    from huggingface_hub import hf_hub_download

    return load_psf_file(hf_hub_download(REPO_ID, "psf.tiff", repo_type="dataset", cache_dir=str(cache_dir)))


def load_psf_file(path: str | Path) -> torch.Tensor:
    """Read a PSF image from disk as a float C×H×W tensor, for a second camera's PSF."""
    import tifffile

    psf = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
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


def transform_psf(psf: torch.Tensor, name: str) -> torch.Tensor:
    """Stand in for a second camera when only one real PSF exists.

    Flipping or rotating keeps the diffuser's statistics (speckle size, spectrum, support) and changes
    only where each speckle sits, which is the part of the PSF that a network could have memorized.
    """
    if name == "rotate180":
        return psf.flip(dims=(-2, -1))
    if name == "flip_horizontal":
        return psf.flip(dims=(-1,))
    if name == "flip_vertical":
        return psf.flip(dims=(-2,))
    raise ValueError(f"unknown PSF transform {name!r}, expected one of {PSF_TRANSFORMS}")
