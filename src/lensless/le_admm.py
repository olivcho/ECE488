"""Le-ADMM: the ADMM solver unrolled for a fixed number of iterations with learned penalties.

Following Monakhova et al. (2019), each unrolled iteration owns its own τ, μ₁, μ₂ and μ₃, and an
optional U-Net denoises the cropped estimate at the end (Le-ADMM-U). The camera enters only through
the ForwardModel, which holds no learned weights, so a trained network is applied to a new camera
by handing it a ForwardModel built from the new PSF.
"""
import torch
from torch import nn

from lensless.admm import ADMMState, ForwardModel, admm_step
from lensless.unet import UNet


class LeADMM(nn.Module):
    def __init__(self, forward_model: ForwardModel, iterations: int = 5, tau: float = 1e-3, mu: float = 0.03, denoiser: dict | None = None):
        super().__init__()
        if iterations < 1:
            raise ValueError("Le-ADMM needs at least one unrolled iteration")
        self.forward_model = forward_model
        self.iterations = iterations
        # Penalties are stored as logarithms so gradient steps can never push them to zero or negative,
        # where the soft threshold and the Fourier division stop making sense.
        self.log_tau = nn.Parameter(torch.full((iterations,), float(torch.log(torch.tensor(tau)))))
        self.log_mu = nn.Parameter(torch.full((iterations, 3), float(torch.log(torch.tensor(mu)))))
        # The classical output's brightness depends on the PSF normalization, so a single learned gain
        # maps it onto the target's 0 to 1 range when there is no denoiser to do it.
        self.log_gain = nn.Parameter(torch.zeros(()))
        self.denoiser = UNet(**denoiser) if denoiser else None

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    @property
    def mu(self) -> torch.Tensor:
        """Shape iterations×3, columns μ₁, μ₂, μ₃."""
        return self.log_mu.exp()

    def set_forward_model(self, forward_model: ForwardModel) -> None:
        """Swap the camera. This is the whole transfer experiment: nothing learned changes."""
        if forward_model.full_size != self.forward_model.full_size or forward_model.channels != self.forward_model.channels:
            raise ValueError("the new PSF must have the same size and channel count as the one trained on")
        self.forward_model = forward_model

    def unroll(self, b: torch.Tensor) -> torch.Tensor:
        """The learned ADMM estimate Cv before any denoising."""
        tau, mu = self.tau, self.mu
        state = ADMMState.zeros(self.forward_model, b.shape[0])
        for k in range(self.iterations):
            state = admm_step(self.forward_model, state, b, tau[k], mu[k, 0], mu[k, 1], mu[k, 2])
        return self.forward_model.crop(state.v) * self.log_gain.exp()

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        estimate = self.unroll(b)
        return self.denoiser(estimate) if self.denoiser is not None else estimate
