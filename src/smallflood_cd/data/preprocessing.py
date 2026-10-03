from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_array(path: str | Path) -> torch.Tensor:
    """Load `.npy` directly or a GeoTIFF through the optional rasterio extra."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        array = np.load(source, allow_pickle=False)
    elif source.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as error:
            raise RuntimeError("GeoTIFF loading requires the `geo` optional dependency") from error
        with rasterio.open(source) as dataset:
            array = dataset.read()
    else:
        raise ValueError(f"Unsupported array format: {source.suffix}")
    tensor = torch.from_numpy(np.asarray(array)).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


class ChannelNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean.reshape(-1, 1, 1).float()
        self.std = std.reshape(-1, 1, 1).float().clamp_min(1e-6)

    @classmethod
    def from_json(cls, path: str | Path) -> "ChannelNormalizer":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(torch.tensor(values["mean"]), torch.tensor(values["std"]))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-3] != self.mean.shape[0]:
            raise ValueError("Input channel count does not match normalization statistics")
        return (tensor - self.mean) / self.std
