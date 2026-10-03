from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class FCSiamDiff(nn.Module):
    """Fully convolutional Siamese difference baseline with shared encoder weights."""

    def __init__(self, input_channels: int = 2, widths: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        source = input_channels
        for width in widths:
            self.encoder_blocks.append(DoubleConv(source, width))
            source = width
        self.pool = nn.MaxPool2d(2)
        self.decoder_blocks = nn.ModuleList()
        top = widths[-1]
        for skip in reversed(widths[:-1]):
            self.decoder_blocks.append(DoubleConv(top + skip, skip))
            top = skip
        self.head = nn.Conv2d(widths[0], 1, 1)

    def _encode(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = []
        for index, block in enumerate(self.encoder_blocks):
            tensor = block(tensor)
            features.append(tensor)
            if index < len(self.encoder_blocks) - 1:
                tensor = self.pool(tensor)
        return tuple(features)

    def forward(
        self, pre: torch.Tensor, post: torch.Tensor, coherence: torch.Tensor | None = None
    ) -> ChangeDetectionOutput:
        del coherence
        pre_features, post_features = self._encode(pre), self._encode(post)
        differences = [torch.abs(after - before) for before, after in zip(pre_features, post_features)]
        tensor = differences[-1]
        for block, skip in zip(self.decoder_blocks, reversed(differences[:-1]), strict=True):
            tensor = F.interpolate(tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            tensor = block(torch.cat([tensor, skip], dim=1))
        logits = self.head(tensor)
        return ChangeDetectionOutput(logits, None, logits)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

