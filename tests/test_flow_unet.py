import pytest
import torch
from torch import nn

from lensless.flow_unet import FlowMatching, FlowUNet, euler_sample, flow_matching_target, sinusoidal_embedding


def small_flow_unet() -> FlowUNet:
    return FlowUNet(base_channels=4, depth=2, time_dim=16, attention_heads=2)


@pytest.mark.parametrize("height,width", [(270, 480), (272, 480), (33, 47)])
def test_velocity_matches_image_size(height, width):
    model = small_flow_unet()
    state, measurement, t = torch.zeros(2, 3, height, width), torch.zeros(2, 3, height, width), torch.rand(2)
    assert model(state, measurement, t).shape == (2, 3, height, width)


def test_time_embedding_is_bounded_distinct_and_repeatable():
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    embedding = sinusoidal_embedding(t, 32)
    assert embedding.shape == (4, 32)
    assert embedding.abs().max() <= 1.0
    assert torch.equal(embedding, sinusoidal_embedding(t, 32))
    pairwise = torch.cdist(embedding, embedding)
    assert (pairwise + torch.eye(4) > 1e-3).all(), "every time must get its own embedding"
    with pytest.raises(ValueError):
        sinusoidal_embedding(t, 7)


def test_time_changes_the_predicted_velocity():
    torch.manual_seed(0)
    model = small_flow_unet()
    nn.init.normal_(model.output.weight)  # The zero-initialised output would hide any time dependence.
    state, measurement = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    early, late = model(state, measurement, torch.tensor([0.1])), model(state, measurement, torch.tensor([0.9]))
    assert not torch.allclose(early, late)


def test_flow_matching_target_interpolates_and_gives_straight_velocity():
    noise, target = torch.rand(3, 3, 8, 8), torch.rand(3, 3, 8, 8)
    t = torch.tensor([0.0, 0.5, 1.0])
    state, velocity = flow_matching_target(noise, target, t)
    assert state.shape == velocity.shape == target.shape
    assert torch.equal(state[0], noise[0])
    assert torch.allclose(state[1], 0.5 * (noise[1] + target[1]))
    assert torch.equal(state[2], target[2])
    assert torch.equal(velocity, target - noise)


class ConstantVelocity(nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.value = value
        self.times = []

    def forward(self, state, measurement, t):
        self.times.append(t[0].item())
        return self.value.expand_as(state)


def test_euler_sampler_visits_each_step_once_and_integrates_exactly():
    noise, measurement = torch.rand(2, 3, 8, 8), torch.zeros(2, 3, 8, 8)
    velocity = torch.rand(1, 3, 8, 8)
    model = ConstantVelocity(velocity)
    sample = euler_sample(model, measurement, noise, steps=4)
    assert model.times == [0.0, 0.25, 0.5, 0.75]
    # A constant field integrates to noise + velocity whatever the step count.
    assert torch.allclose(sample, noise + velocity, atol=1e-6)
    with pytest.raises(ValueError):
        euler_sample(model, measurement, noise, steps=0)


def test_forward_samples_repeatably_with_image_shape():
    torch.manual_seed(0)
    model = FlowMatching(base_channels=4, depth=2, time_dim=16, attention_heads=2, sample_steps=3).eval()
    measurement = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        first, second = model(measurement), model(measurement)
    assert first.shape == measurement.shape
    assert torch.equal(first, second), "the fixed sampling seed must make validation scores repeatable"


def test_every_parameter_receives_a_gradient():
    model = FlowMatching(base_channels=4, depth=2, time_dim=16, attention_heads=2)
    model.loss(torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_small_batch_velocity_can_be_learned():
    torch.manual_seed(0)
    model = FlowMatching(base_channels=8, depth=2, time_dim=16, attention_heads=2)
    measurement, target = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first_loss = None
    for _ in range(60):
        loss = model.loss(measurement, target)
        first_loss = first_loss or loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * first_loss
