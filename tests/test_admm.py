"""Each ADMM update is checked against the equation it solves, so a bad reconstruction means tuning, not a bug."""
import pytest
import torch

from lensless.admm import (
    ADMMState,
    ForwardModel,
    admm_iterations,
    admm_reconstruct,
    admm_step,
    gradient,
    gradient_adjoint,
    objective,
    soft_threshold,
    update_u,
    update_v,
    update_w,
    update_x,
)
from lensless.psf import prepare_psf


def small_model(seed: int = 0, channels: int = 2, height: int = 12, width: int = 20) -> ForwardModel:
    generator = torch.Generator().manual_seed(seed)
    psf = torch.rand(channels, height, width, generator=generator) ** 8  # A few bright speckles, like a diffuser
    return ForwardModel(psf / psf.sum(dim=(-2, -1), keepdim=True))


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum()


def test_pad_then_crop_is_identity_and_crop_is_centered():
    model = small_model()
    sensor = torch.rand(3, model.channels, model.height, model.width)
    padded = model.pad(sensor)
    assert padded.shape == (3, model.channels, 24, 40)
    assert torch.equal(model.crop(padded), sensor)
    assert padded[..., : model.top, :].abs().sum() == 0 and padded[..., :, : model.left].abs().sum() == 0


def test_measuring_a_point_source_returns_the_psf():
    psf = torch.rand(1, 12, 20)
    psf = psf / psf.sum()
    model = ForwardModel(psf)
    point = torch.zeros(1, 1, *model.full_size)
    point[0, 0, model.top + 6, model.left + 10] = 1.0
    measured = model.measure(point)[0]
    # The PSF is stored with its center at the sensor center, so a centered point reproduces it exactly.
    assert torch.allclose(measured, psf, atol=1e-6)


def test_convolution_adjoint_satisfies_dot_product_test():
    model = small_model()
    a = torch.randn(2, model.channels, *model.full_size)
    b = torch.randn(2, model.channels, *model.full_size)
    assert dot(model.convolve(a), b).item() == pytest.approx(dot(a, model.convolve_adjoint(b)).item(), rel=1e-5)


def test_gradient_adjoint_satisfies_dot_product_test():
    image = torch.randn(2, 3, 7, 9)
    differences = torch.randn(2, 3, 2, 7, 9)
    assert dot(gradient(image), differences).item() == pytest.approx(dot(image, gradient_adjoint(differences)).item(), rel=1e-5)


def test_gradient_is_zero_on_flat_images_and_sees_an_edge():
    image = torch.zeros(1, 1, 4, 6)
    assert gradient(image).abs().sum() == 0
    image[..., :, 3:] = 1.0
    horizontal, vertical = gradient(image)[0, 0]
    assert vertical.abs().sum() == 0 and horizontal.abs().sum() == 2 * 4  # the edge and the wrap-around


def test_laplacian_spectrum_matches_direct_operator():
    model = small_model()
    image = torch.randn(1, 1, *model.full_size)
    via_fourier = torch.fft.irfft2(torch.fft.rfft2(image) * model.laplacian_spectrum, s=model.full_size)
    assert torch.allclose(via_fourier, gradient_adjoint(gradient(image)), atol=1e-5)


def test_soft_threshold_shrinks_toward_zero():
    values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert torch.allclose(soft_threshold(values, 1.0), torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0]))


def test_update_u_is_optimal_for_its_subproblem():
    """A proximal step must beat nearby points on the objective it minimizes."""
    tau, mu2 = 0.3, 2.0
    v, eta = torch.randn(1, 1, 4, 6), torch.randn(1, 1, 2, 4, 6)
    u = update_u(v, eta, tau, mu2)

    def cost(candidate):
        return tau * candidate.abs().sum() + mu2 / 2 * (gradient(v) - candidate).square().sum() + (eta * (gradient(v) - candidate)).sum()

    for _ in range(20):
        assert cost(u) <= cost(u + 0.05 * torch.randn_like(u)) + 1e-6


def test_update_x_solves_its_normal_equation():
    model = small_model()
    mu1 = 0.7
    v = torch.randn(2, model.channels, *model.full_size)
    xi = torch.randn_like(v)
    b = torch.randn(2, model.channels, model.height, model.width)
    x = update_x(model, v, b, xi, mu1)
    left = model.pad(model.crop(x)) + mu1 * x  # (CᴴC + μ₁I)x
    right = xi + mu1 * model.convolve(v) + model.pad(b)
    assert torch.allclose(left, right, atol=1e-5)


def test_update_w_is_nonnegative_projection():
    v, rho, mu3 = torch.tensor([[-1.0, 0.5]]), torch.tensor([[0.5, -2.0]]), 1.0
    assert torch.equal(update_w(v, rho, mu3), torch.tensor([[0.0, 0.0]]))
    assert torch.equal(update_w(torch.tensor([[2.0]]), torch.tensor([[1.0]]), 2.0), torch.tensor([[2.5]]))


def test_update_v_solves_its_normal_equation():
    model = small_model()
    mu1, mu2, mu3 = 0.5, 0.2, 0.9
    shape = (2, model.channels, *model.full_size)
    x, w, xi, rho = (torch.randn(shape) for _ in range(4))
    u, eta = torch.randn(2, model.channels, 2, *model.full_size), torch.randn(2, model.channels, 2, *model.full_size)
    v = update_v(model, x, u, w, xi, eta, rho, mu1, mu2, mu3)
    left = mu1 * model.convolve_adjoint(model.convolve(v)) + mu2 * gradient_adjoint(gradient(v)) + mu3 * v
    right = (mu3 * w - rho) + gradient_adjoint(mu2 * u - eta) + model.convolve_adjoint(mu1 * x - xi)
    assert torch.allclose(left, right, atol=1e-4)


def test_step_keeps_shapes_and_zero_state_starts_zero():
    model = small_model()
    b = torch.rand(3, model.channels, model.height, model.width)
    state = ADMMState.zeros(model, 3)
    assert state.v.abs().sum() == 0 and state.u.shape == (3, model.channels, 2, *model.full_size)
    new = admm_step(model, state, b, 1e-3, 1.0, 1.0, 1.0)
    assert new.v.shape == state.v.shape and new.eta.shape == state.eta.shape


def test_admm_recovers_a_piecewise_constant_scene():
    """Noise-free simulated data: the objective must fall and the scene must come back."""
    model = small_model(seed=1)
    scene = torch.zeros(1, model.channels, *model.full_size)
    scene[..., model.top + 3 : model.top + 8, model.left + 5 : model.left + 14] = 0.8
    scene[..., model.top + 9 : model.top + 11, model.left + 2 : model.left + 18] = 0.4
    b = model.measure(scene)

    # Penalties near the data term's curvature converge in hundreds of iterations; mu=1 needs thousands.
    tau, mu = 1e-4, 1e-2
    objectives = [objective(model, b, state.v, tau).item() for _, state in admm_iterations(model, b, tau, mu, mu, mu, 300)]
    assert objectives[-1] < 0.05 * objectives[0]

    recon = admm_reconstruct(model, b, tau, mu, mu, mu, 300)
    assert recon.shape == (1, model.channels, model.height, model.width)
    error = (recon - model.crop(scene)).abs().mean().item()
    assert error < 0.01, error


def test_admm_rejects_wrong_measurement_shape():
    model = small_model()
    with pytest.raises(ValueError):
        next(admm_iterations(model, torch.zeros(1, model.channels, 5, 5), 1e-3, 1.0, 1.0, 1.0, 1))


def test_prepare_psf_downsamples_removes_background_and_sums_to_one():
    psf = torch.full((3, 1080, 1920), 10.0)
    psf[:, 500:520, 900:940] = 200.0
    prepared = prepare_psf(psf)
    assert prepared.shape == (3, 270, 480)
    assert torch.allclose(prepared.sum(dim=(-2, -1)), torch.ones(3))
    assert prepared.min() == 0 and (prepared > 0).sum() == 3 * 5 * 10
    with pytest.raises(ValueError):
        prepare_psf(torch.zeros(270, 480))
    with pytest.raises(ValueError):
        prepare_psf(torch.ones(1, 270, 480))
