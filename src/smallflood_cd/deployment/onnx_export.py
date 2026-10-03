from __future__ import annotations

from pathlib import Path

import torch


class ChangeLogitExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return self.model(pre, post).change_logits


def export_onnx(
    model: torch.nn.Module,
    destination: str | Path,
    input_channels: int = 2,
    patch_size: int = 256,
    opset: int = 17,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ChangeLogitExportWrapper(model.eval())
    example = torch.zeros(1, input_channels, patch_size, patch_size)
    torch.onnx.export(
        wrapper,
        (example, example.clone()),
        destination,
        input_names=["pre", "post"],
        output_names=["change_logits"],
        opset_version=opset,
        dynamo=False,
    )
    return destination
