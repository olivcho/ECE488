"""Conditional flow matching for lensless reconstruction.

The velocity network is a time-conditioned U-Net with self-attention at the
bottleneck. It receives the noisy state x_t concatenated with the lensless
measurement b on the channel axis, plus a sinusoidal embedding of t that is
added into every block. Training regresses the straight-line velocity
target − noise; sampling integrates that field from t = 0 to t = 1 with a
fixed-step Euler integrator.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Map times in [0, 1] to an N×dim vector of sines and cosines.

    Times are scaled by 1,000 first, so the lowest frequencies still move
    across [0, 1]; without the scale their arguments would stay near zero.
    """
    if dim % 2:
        raise ValueError(f"embedding dim must be even, got {dim}")
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angles = 1_000.0 * t.float()[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def group_norm(channels: int) -> nn.GroupNorm:
    # GroupNorm rather than BatchNorm: one batch mixes every noise level, so batch statistics are not meaningful.
    return nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)


class TimeBlock(nn.Module):
    """Two 3×3 convolutions with the time embedding added as a per-channel shift between them."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), group_norm(out_channels), nn.SiLU())
        self.time = nn.Linear(time_dim, out_channels)
        self.second = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), group_norm(out_channels), nn.SiLU())
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, features: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.first(features) + self.time(time_embedding)[:, :, None, None]
        return self.second(hidden) + self.skip(features)


class SelfAttention(nn.Module):
    """Multi-head self-attention over all spatial positions, with a residual connection.

    Used only at the bottleneck, where the grid is small enough for full
    attention. This supplies the global mixing a lensless inverse needs,
    since every scene point spreads over the whole sensor.
    """

    def __init__(self, channels: int, heads: int):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels {channels} must be divisible by heads {heads}")
        self.heads = heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)
        # Zero init makes the block an identity at the start, so early training is the plain U-Net.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        qkv = self.qkv(self.norm(features)).view(batch, 3, self.heads, channels // self.heads, height * width)
        query, key, value = qkv.transpose(-1, -2).unbind(dim=1)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(-1, -2).reshape(batch, channels, height, width)
        return features + self.project(attended)


class FlowUNet(nn.Module):
    """Velocity network v(x_t, b, t). Same encoder, decoder and padding scheme as lensless.unet.UNet."""

    def __init__(self, image_channels: int = 3, base_channels: int = 16, depth: int = 3, time_dim: int = 128, attention_heads: int = 4):
        super().__init__()
        self.size_multiple = 2**depth
        self.time_dim = time_dim
        widths = [base_channels * 2**level for level in range(depth + 1)]

        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.encoders = nn.ModuleList()
        previous = 2 * image_channels  # x_t and b stacked on channels
        for width in widths[:-1]:
            self.encoders.append(TimeBlock(previous, width, time_dim))
            previous = width
        self.bottleneck_in = TimeBlock(widths[-2], widths[-1], time_dim)
        self.attention = SelfAttention(widths[-1], attention_heads)
        self.bottleneck_out = TimeBlock(widths[-1], widths[-1], time_dim)
        self.decoders = nn.ModuleList(
            TimeBlock(widths[level + 1] + widths[level], widths[level], time_dim) for level in reversed(range(depth))
        )
        self.output = nn.Conv2d(widths[0], image_channels, 1)
        # A zero initial velocity keeps the first samples at the noise, which is the stable starting point.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, measurement: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        height, width = state.shape[-2:]
        time_embedding = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        # 270 cannot be halved three times, so pad, then crop the output back.
        features = F.pad(
            torch.cat([state, measurement], dim=1),
            (0, -width % self.size_multiple, 0, -height % self.size_multiple),
            mode="replicate",
        )

        skip_connections = []
        for encoder in self.encoders:
            features = encoder(features, time_embedding)
            skip_connections.append(features)
            features = F.max_pool2d(features, 2)
        features = self.bottleneck_in(features, time_embedding)
        features = self.attention(features)
        features = self.bottleneck_out(features, time_embedding)

        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = decoder(torch.cat([features, skip], dim=1), time_embedding)

        return self.output(features)[..., :height, :width]


def flow_matching_target(noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate x_t = (1 − t)·noise + t·target and return it with the velocity target − noise."""
    t = t.view(-1, *([1] * (target.dim() - 1)))
    return (1.0 - t) * noise + t * target, target - noise


def flow_matching_loss(velocity_model: nn.Module, measurement: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One training step's loss: MSE between the predicted and the straight-line velocity at a random t."""
    t = torch.rand(target.shape[0], device=target.device)
    noise = torch.randn_like(target)
    state, velocity = flow_matching_target(noise, target, t)
    return F.mse_loss(velocity_model(state, measurement, t), velocity)


def euler_sample(velocity_model: nn.Module, measurement: torch.Tensor, noise: torch.Tensor, steps: int) -> torch.Tensor:
    """Integrate dx/dt = v(x, b, t) from t = 0 to 1 in equal steps, starting at the noise."""
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    state = noise
    for step in range(steps):
        t = torch.full((state.shape[0],), step / steps, device=state.device)
        state = state + velocity_model(state, measurement, t) / steps
    return state


class FlowMatching(nn.Module):
    """Reconstruction model whose forward pass samples an image from a measurement.

    Exposing sampling as `forward(measurement)` lets the evaluation, preview and
    PSF-swap code treat it exactly like the U-Net and Le-ADMM.
    """

    def __init__(self, base_channels: int = 16, depth: int = 3, time_dim: int = 128, attention_heads: int = 4,
                 sample_steps: int = 20, sample_seed: int = 0):
        super().__init__()
        self.velocity = FlowUNet(3, base_channels, depth, time_dim, attention_heads)
        self.sample_steps = sample_steps
        self.sample_seed = sample_seed

    def loss(self, measurement: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return flow_matching_loss(self.velocity, measurement, target)

    def forward(self, measurement: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        # Fixed-seed noise, drawn on the CPU, so a validation score is repeatable on any device.
        generator = torch.Generator().manual_seed(self.sample_seed)
        noise = torch.randn(measurement.shape, generator=generator).to(measurement.device)
        return euler_sample(self.velocity, measurement, noise, steps or self.sample_steps)
