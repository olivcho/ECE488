import pytest
import torch

from lensless.admm import ForwardModel, admm_reconstruct
from lensless.le_admm import LeADMM


def make_model(seed: int = 0, size=(12, 16)) -> ForwardModel:
    generator = torch.Generator().manual_seed(seed)
    psf = torch.rand(3, *size, generator=generator) ** 4
    return ForwardModel(psf / psf.sum(dim=(-2, -1), keepdim=True))


def measurement(model: ForwardModel, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(2, 3, model.height, model.width, generator=generator)


def test_untrained_le_admm_matches_classical_admm():
    model = make_model()
    b = measurement(model)
    network = LeADMM(model, iterations=4, tau=1e-3, mu=0.05)
    with torch.no_grad():
        unrolled = network(b)
    classical = admm_reconstruct(model, b, 1e-3, 0.05, 0.05, 0.05, iterations=4)
    assert torch.allclose(unrolled, classical, atol=1e-6)


def test_every_penalty_receives_a_gradient():
    model = make_model()
    network = LeADMM(model, iterations=3)
    target = torch.rand_like(measurement(model))
    loss = torch.nn.functional.mse_loss(network(measurement(model)), target)
    loss.backward()
    # The first τ cannot matter: v and η start at zero, so the first soft threshold sees only zeros.
    assert network.log_tau.grad is not None and network.log_tau.grad[0] == 0 and (network.log_tau.grad[1:] != 0).all()
    assert network.log_mu.grad is not None and (network.log_mu.grad != 0).all()
    assert network.log_gain.grad is not None and network.log_gain.grad != 0
    assert network.tau.shape == (3,) and network.mu.shape == (3, 3)


def test_penalties_stay_positive_after_large_steps():
    network = LeADMM(make_model(), iterations=2)
    with torch.no_grad():
        network.log_tau -= 50.0
        network.log_mu -= 50.0
    assert (network.tau > 0).all() and (network.mu > 0).all()
    assert torch.isfinite(network(measurement(network.forward_model))).all()


def test_swapping_the_psf_changes_the_output_but_not_the_weights():
    network = LeADMM(make_model(seed=0), iterations=2)
    b = measurement(network.forward_model)
    before = {name: value.clone() for name, value in network.state_dict().items()}
    with torch.no_grad():
        first = network(b)
        network.set_forward_model(make_model(seed=7))
        second = network(b)
    assert not torch.allclose(first, second)
    assert all(torch.equal(before[name], value) for name, value in network.state_dict().items())
    assert "forward_model" not in " ".join(before)
    with pytest.raises(ValueError):
        network.set_forward_model(make_model(size=(8, 8)))


def test_denoiser_variant_keeps_the_sensor_shape():
    model = make_model(size=(16, 16))
    network = LeADMM(model, iterations=2, denoiser={"base_channels": 4, "depth": 2})
    b = measurement(model)
    assert network(b).shape == b.shape
    assert sum(p.numel() for p in network.parameters()) > 2 * 4 + 1
    with pytest.raises(ValueError):
        LeADMM(model, iterations=0)
