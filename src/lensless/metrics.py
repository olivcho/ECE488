"""Image quality metrics, reported for the whole frame and the monitor screen."""
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

# Measured from the mean lensed test image. Whole-frame scores include a dark
# surround that is about a third of the frame, so reporting both shows its effect.
SCREEN_ROWS = (0, 213)
SCREEN_COLS = (59, 468)


def crop_to_screen(images: torch.Tensor) -> torch.Tensor:
    """Keep the monitor region of an N×C×270×480 batch."""
    if images.shape[-2:] != (270, 480):
        raise ValueError(f"expected 270×480 images, got {tuple(images.shape[-2:])}")
    return images[..., SCREEN_ROWS[0]:SCREEN_ROWS[1], SCREEN_COLS[0]:SCREEN_COLS[1]]


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-image PSNR in dB for images in [0, 1]."""
    per_image_mse = (prediction - target).pow(2).flatten(1).mean(dim=1)
    return (-10.0 * torch.log10(per_image_mse.clamp_min(1e-12))).mean()


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean SSIM for images in [0, 1]."""
    return structural_similarity_index_measure(prediction, target, data_range=1.0)


@torch.no_grad()
def image_metrics(prediction: torch.Tensor, target: torch.Tensor, perceptual_loss) -> dict[str, float]:
    """Score one batch. Predictions are clipped because real images stay in [0, 1]."""
    prediction = prediction.clamp(0.0, 1.0)
    regions = {
        "whole": (prediction, target),
        "screen": (crop_to_screen(prediction), crop_to_screen(target)),
    }
    scores = {}
    for region, (region_prediction, region_target) in regions.items():
        scores[f"{region}_psnr"] = psnr(region_prediction, region_target).item()
        scores[f"{region}_ssim"] = ssim(region_prediction, region_target).item()
        scores[f"{region}_lpips"] = perceptual_loss(
            region_prediction, region_target, normalize=True
        ).mean().item()
    return scores
