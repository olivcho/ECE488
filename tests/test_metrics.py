import pytest
import torch

from lensless.metrics import crop_to_screen, image_metrics, psnr, ssim


def test_psnr_matches_known_value():
    target = torch.zeros(2, 3, 8, 8)
    assert psnr(target + 0.1, target).item() == pytest.approx(20.0, abs=1e-4)


def test_psnr_averages_per_image():
    target = torch.zeros(2, 1, 4, 4)
    prediction = target.clone()
    prediction[0] = 0.1
    prediction[1] = 0.01
    assert psnr(prediction, target).item() == pytest.approx(30.0, abs=1e-3)


def test_ssim_is_one_for_identical_images():
    image = torch.rand(2, 3, 64, 64)
    assert ssim(image, image).item() == pytest.approx(1.0, abs=1e-5)
    assert ssim(torch.rand(2, 3, 64, 64), image).item() < 0.5


def test_crop_to_screen_uses_measured_region():
    assert crop_to_screen(torch.zeros(1, 3, 270, 480)).shape == (1, 3, 213, 409)
    with pytest.raises(ValueError):
        crop_to_screen(torch.zeros(1, 3, 272, 480))


def test_image_metrics_reports_both_regions_and_clips():
    target = torch.zeros(1, 3, 270, 480)
    scores = image_metrics(target - 5.0, target, lambda a, b, normalize: (a - b).abs().mean().unsqueeze(0))
    assert set(scores) == {f"{r}_{m}" for r in ("whole", "screen") for m in ("psnr", "ssim", "lpips")}
    assert scores["whole_lpips"] == 0.0 and scores["screen_psnr"] > 100
