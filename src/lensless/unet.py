"""Pure U-Net baseline. It receives no PSF, so all camera knowledge is in its weights."""
import torch
from torch import nn
from torch.nn import functional as F


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.size_multiple = 2**depth
        widths = [base_channels * 2**level for level in range(depth + 1)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths[:-1]:
            self.encoders.append(double_conv(previous, width))
            previous = width
        self.bottleneck = double_conv(widths[-2], widths[-1])

        self.decoders = nn.ModuleList(
            double_conv(widths[level + 1] + widths[level], widths[level])
            for level in reversed(range(depth))
        )
        self.output = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        # 270 cannot be halved four times, so pad, then crop the output back.
        features = F.pad(
            images,
            (0, -width % self.size_multiple, 0, -height % self.size_multiple),
            mode="replicate",
        )

        skip_connections = []
        for encoder in self.encoders:
            features = encoder(features)
            skip_connections.append(features)
            features = F.max_pool2d(features, 2)
        features = self.bottleneck(features)

        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            # Bilinear upsampling avoids the checkerboard artifacts of transposed convolutions.
            features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = decoder(torch.cat([features, skip], dim=1))

        return self.output(features)[..., :height, :width]
