"""ADMM reconstruction for lensless images, following docs/admm_updates.md.

Each update is a plain function of the current state, so the same step can later be unrolled with
learned penalties (Le-ADMM) without rewriting the solver. Tensors are batched as N×C×H×W, and each
color channel is solved with its own PSF channel.
"""
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.nn import functional as F

Scalar = float | torch.Tensor


class ForwardModel:
    """The camera: b = C M v.

    M is circular convolution with the PSF on a plane twice the sensor size, and C crops the center
    back to the sensor. Doubling the plane keeps the circular wrap-around away from the sensor, so the
    cheap Fourier-domain inverse in the v update stays faithful to the physical, non-circular blur.
    """

    def __init__(self, psf: torch.Tensor):
        if psf.ndim != 3:
            raise ValueError(f"expected C×H×W PSF, got shape {tuple(psf.shape)}")
        self.channels, self.height, self.width = psf.shape
        self.full_size = (2 * self.height, 2 * self.width)
        self.top, self.left = self.height // 2, self.width // 2
        self.device = psf.device
        # ifftshift moves the PSF's center to index (0, 0), so convolving does not also translate the image.
        self.psf_spectrum = torch.fft.rfft2(torch.fft.ifftshift(self.pad(psf), dim=(-2, -1)))
        self.sensor_mask = self.pad(torch.ones(1, self.height, self.width, device=psf.device))
        self.laplacian_spectrum = _laplacian_spectrum(self.full_size, psf.device)

    def crop(self, full: torch.Tensor) -> torch.Tensor:
        """C: keep the sensor region."""
        return full[..., self.top : self.top + self.height, self.left : self.left + self.width]

    def pad(self, sensor: torch.Tensor) -> torch.Tensor:
        """Cᴴ: zero-pad the sensor onto the full plane, centered."""
        bottom = self.full_size[0] - self.height - self.top
        right = self.full_size[1] - self.width - self.left
        return F.pad(sensor, (self.left, right, self.top, bottom))

    def convolve(self, full: torch.Tensor) -> torch.Tensor:
        """M: blur with the PSF."""
        return torch.fft.irfft2(torch.fft.rfft2(full) * self.psf_spectrum, s=self.full_size)

    def convolve_adjoint(self, full: torch.Tensor) -> torch.Tensor:
        """Mᴴ: correlate with the PSF, which is convolution with the flipped PSF."""
        return torch.fft.irfft2(torch.fft.rfft2(full) * self.psf_spectrum.conj(), s=self.full_size)

    def measure(self, full: torch.Tensor) -> torch.Tensor:
        """b = C M v."""
        return self.crop(self.convolve(full))


def gradient(image: torch.Tensor) -> torch.Tensor:
    """Ψ: horizontal and vertical forward differences with circular boundaries, stacked as N×C×2×H×W."""
    horizontal = image - torch.roll(image, shifts=1, dims=-1)
    vertical = image - torch.roll(image, shifts=1, dims=-2)
    return torch.stack([horizontal, vertical], dim=-3)


def gradient_adjoint(differences: torch.Tensor) -> torch.Tensor:
    """Ψᴴ: the negative divergence that pairs with `gradient` under the plain dot product."""
    horizontal, vertical = differences.unbind(dim=-3)
    return (horizontal - torch.roll(horizontal, shifts=-1, dims=-1)) + (vertical - torch.roll(vertical, shifts=-1, dims=-2))


def _laplacian_spectrum(size: tuple[int, int], device) -> torch.Tensor:
    """|Ψ̃|²: the transfer function of ΨᴴΨ, read off from its response to an impulse."""
    impulse = torch.zeros(1, *size, device=device)
    impulse[0, 0, 0] = 1.0
    return torch.fft.rfft2(gradient_adjoint(gradient(impulse))).real


def soft_threshold(values: torch.Tensor, threshold: Scalar) -> torch.Tensor:
    """𝒯ₜ(z) = sign(z) max(|z| − t, 0), the proximal map of t‖·‖₁."""
    return torch.sign(values) * (values.abs() - threshold).clamp_min(0.0)


@dataclass
class ADMMState:
    """All variables from the doc's "change every iteration" table. Shapes are N×C×2H×2W unless noted."""

    v: torch.Tensor
    x: torch.Tensor
    u: torch.Tensor  # N×C×2×2H×2W: one horizontal and one vertical channel
    w: torch.Tensor
    xi: torch.Tensor
    eta: torch.Tensor  # same shape as u
    rho: torch.Tensor

    @classmethod
    def zeros(cls, model: ForwardModel, batch_size: int) -> "ADMMState":
        full = torch.zeros(batch_size, model.channels, *model.full_size, device=model.device)
        pair = torch.zeros(batch_size, model.channels, 2, *model.full_size, device=model.device)
        return cls(v=full, x=full.clone(), u=pair, w=full.clone(), xi=full.clone(), eta=pair.clone(), rho=full.clone())


def update_u(v, eta, tau: Scalar, mu2: Scalar) -> torch.Tensor:
    """Step 1: u ← 𝒯_{τ/μ₂}(Ψv + η/μ₂)."""
    return soft_threshold(gradient(v) + eta / mu2, tau / mu2)


def update_x(model: ForwardModel, v, b, xi, mu1: Scalar) -> torch.Tensor:
    """Step 2: x ← (CᴴC + μ₁I)⁻¹(ξ + μ₁Mv + Cᴴb), a per-pixel division."""
    return (xi + mu1 * model.convolve(v) + model.pad(b)) / (model.sensor_mask + mu1)


def update_w(v, rho, mu3: Scalar) -> torch.Tensor:
    """Step 3: w ← max(v + ρ/μ₃, 0)."""
    return (v + rho / mu3).clamp_min(0.0)


def update_v(model: ForwardModel, x, u, w, xi, eta, rho, mu1: Scalar, mu2: Scalar, mu3: Scalar) -> torch.Tensor:
    """Step 4: v ← (μ₁MᴴM + μ₂ΨᴴΨ + μ₃I)⁻¹ r, solved by element-wise division in Fourier space."""
    r = (mu3 * w - rho) + gradient_adjoint(mu2 * u - eta) + model.convolve_adjoint(mu1 * x - xi)
    denominator = mu1 * model.psf_spectrum.abs().square() + mu2 * model.laplacian_spectrum + mu3
    return torch.fft.irfft2(torch.fft.rfft2(r) / denominator, s=model.full_size)


def admm_step(model: ForwardModel, state: ADMMState, b, tau: Scalar, mu1: Scalar, mu2: Scalar, mu3: Scalar) -> ADMMState:
    """One full iteration in the doc's order: u, x, w, v, then the dual variables."""
    u = update_u(state.v, state.eta, tau, mu2)
    x = update_x(model, state.v, b, state.xi, mu1)
    w = update_w(state.v, state.rho, mu3)
    v = update_v(model, x, u, w, state.xi, state.eta, state.rho, mu1, mu2, mu3)
    # Step 5: each dual variable accumulates the disagreement its constraint still has.
    xi = state.xi + mu1 * (model.convolve(v) - x)
    eta = state.eta + mu2 * (gradient(v) - u)
    rho = state.rho + mu3 * (v - w)
    return ADMMState(v=v, x=x, u=u, w=w, xi=xi, eta=eta, rho=rho)


def objective(model: ForwardModel, b, v, tau: Scalar) -> torch.Tensor:
    """Per-image value of ½‖b − CMv‖² + τ‖Ψv‖₁, the original problem without the splitting."""
    data_fit = 0.5 * (b - model.measure(v)).square().flatten(1).sum(dim=1)
    return data_fit + tau * gradient(v).abs().flatten(1).sum(dim=1)


def admm_iterations(model: ForwardModel, b, tau: Scalar, mu1: Scalar, mu2: Scalar, mu3: Scalar, iterations: int) -> Iterator[tuple[int, ADMMState]]:
    """Yield (iteration, state) after each step, so callers can score intermediate iteration counts."""
    if b.shape[-3:] != (model.channels, model.height, model.width):
        raise ValueError(f"expected measurements of shape N×{model.channels}×{model.height}×{model.width}, got {tuple(b.shape)}")
    state = ADMMState.zeros(model, b.shape[0])
    for iteration in range(1, iterations + 1):
        state = admm_step(model, state, b, tau, mu1, mu2, mu3)
        yield iteration, state


def admm_reconstruct(model: ForwardModel, b, tau: Scalar, mu1: Scalar, mu2: Scalar, mu3: Scalar, iterations: int) -> torch.Tensor:
    """Run ADMM and return Cv, the estimate cropped to the sensor."""
    state = ADMMState.zeros(model, b.shape[0])
    for _, state in admm_iterations(model, b, tau, mu1, mu2, mu3, iterations):
        pass
    return model.crop(state.v)
