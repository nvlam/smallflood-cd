from __future__ import annotations

from typing import Any

from torch import nn

from smallflood_cd.models import SmallFloodCDNet
from smallflood_cd.models.baselines.bit import BITBaseline
from smallflood_cd.models.baselines.fc_siam_diff import FCSiamDiff


def build_model(config: dict[str, Any]) -> nn.Module:
    name = config.get("name")
    if name == "bit_sar_v2":
        from smallflood_cd.models.baselines.bit_sar_v2 import BITSARV2

        return BITSARV2(
            input_channels=int(config.get("input_channels", 2)),
            pretrained=bool(config.get("pretrained", False)),
        )
    if name == "fc_siam_diff":
        return FCSiamDiff(
            input_channels=int(config.get("input_channels", 2)),
            widths=tuple(config.get("widths", [16, 32, 64, 128])),
        )
    if name == "bit":
        return BITBaseline(
            input_channels=int(config.get("input_channels", 2)),
            embedding_dim=int(config.get("embedding_dim", 64)),
            token_count=int(config.get("token_count", 4)),
            transformer_layers=int(config.get("transformer_layers", 2)),
            attention_heads=int(config.get("attention_heads", 4)),
            pretrained=bool(config.get("pretrained", False)),
        )
    if name != "smallflood_cdnet":
        raise ValueError(f"Unsupported model: {name}")
    projection = tuple(config.get("projection_channels", [24, 32, 64, 96]))
    model = SmallFloodCDNet(
        input_channels=int(config.get("input_channels", 2)),
        projection_channels=projection,
        pretrained=bool(config.get("pretrained", False)),
        coherence_channels=(1 if config.get("include_coherence", False) else 0),
        use_boundary_head=bool(config.get("boundary_head", True)),
        boundary_residual_init=float(config.get("boundary_residual_init", 0.1)),
    )
    model.assert_parameter_budget(int(config.get("max_parameters", 5_000_000)))
    return model
