from __future__ import annotations

import torch.nn as nn

from smallflood_cd.models.blocks import DepthwiseSeparableConv


class ChangeHead(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(DepthwiseSeparableConv(channels, channels), nn.Conv2d(channels, 1, 1))
