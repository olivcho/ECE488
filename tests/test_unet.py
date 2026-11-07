import pytest
import torch

from lensless.unet import UNet


@pytest.mark.parametrize("height,width", [(270, 480), (272, 480), (33, 47)])
def test_output_matches_input_size(height, width):
    model = UNet(base_channels=4, depth=4)
    assert model(torch.zeros(2, 3, height, width)).shape == (2, 3, height, width)


def test_skip_connections_carry_gradients_to_first_layer():
    model = UNet(base_channels=4, depth=2)
    model(torch.rand(2, 3, 32, 32)).mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_small_batch_can_be_memorized():
    torch.manual_seed(0)
    inputs, targets = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    model = UNet(base_channels=8, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first_loss = None
    for _ in range(60):
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        first_loss = first_loss or loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.25 * first_loss
