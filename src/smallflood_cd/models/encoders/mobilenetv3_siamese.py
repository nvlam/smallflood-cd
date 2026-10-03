from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3SiameseEncoder(nn.Module):
    """Shared MobileNetV3-Small feature encoder at strides 4, 8, 16, and 32."""

    output_channels = (16, 24, 48, 576)
    capture_indices = (1, 3, 8, 12)

    def __init__(self, input_channels: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        network = mobilenet_v3_small(weights=weights)
        if input_channels != 3:
            first = network.features[0][0]
            replacement = nn.Conv2d(
                input_channels,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    mean_weight = first.weight.mean(dim=1, keepdim=True)
                    replacement.weight.copy_(mean_weight.repeat(1, input_channels, 1, 1))
            network.features[0][0] = replacement
        self.features = network.features

    def forward_once(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        for index, block in enumerate(self.features):
            tensor = block(tensor)
            if index in self.capture_indices:
                outputs.append(tensor)
        return tuple(outputs)

    def forward(
        self, pre: torch.Tensor, post: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        return self.forward_once(pre), self.forward_once(post)

