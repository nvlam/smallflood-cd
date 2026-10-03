from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from smallflood_cd.data.component_weights import component_weight_map
from smallflood_cd.data.connected_components import ConnectedComponentCache


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    event_id: str
    pre_path: str
    post_path: str
    label_path: str
    valid_mask_path: str
    coherence_path: str | None
    uncertain_mask_path: str | None
    split: str


def read_manifest(path: str | Path, split: str) -> list[PatchRecord]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "patch_id",
        "event_id",
        "pre_path",
        "post_path",
        "label_path",
        "valid_mask_path",
        "split",
    }
    if rows and not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0]))
        raise ValueError(f"Manifest is missing columns: {missing}")
    return [
        PatchRecord(
            patch_id=row["patch_id"],
            event_id=row["event_id"],
            pre_path=row["pre_path"],
            post_path=row["post_path"],
            label_path=row["label_path"],
            valid_mask_path=row["valid_mask_path"],
            coherence_path=row.get("coherence_path") or None,
            uncertain_mask_path=row.get("uncertain_mask_path") or None,
            split=row["split"],
        )
        for row in rows
        if row["split"] == split
    ]


class ManifestDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Backend-neutral manifest dataset.

    A loader callback is injected so GeoTIFF, NumPy, and test fixture backends
    can share the same event-isolation contract.
    """

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        loader: Callable[[str], torch.Tensor],
        component_cache: ConnectedComponentCache | None = None,
        reference_component_area: float | None = None,
        component_alpha: float = 1.0,
        component_gamma: float = 0.5,
        component_weight_cap: float = 4.0,
        use_uncertain_mask: bool = True,
    ) -> None:
        self.records = read_manifest(manifest, split)
        self.loader = loader
        self.component_cache = component_cache
        self.reference_component_area = reference_component_area
        self.component_alpha = component_alpha
        self.component_gamma = component_gamma
        self.component_weight_cap = component_weight_cap
        self.use_uncertain_mask = use_uncertain_mask

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        target = self.loader(record.label_path)
        item: dict[str, torch.Tensor | str] = {
            "pre": self.loader(record.pre_path),
            "post": self.loader(record.post_path),
            "target": target,
            "valid_mask": self.loader(record.valid_mask_path),
            "patch_id": record.patch_id,
            "event_id": record.event_id,
        }
        if record.coherence_path:
            item["coherence"] = self.loader(record.coherence_path)
        if self.use_uncertain_mask and record.uncertain_mask_path:
            item["uncertain_mask"] = self.loader(record.uncertain_mask_path)
        if self.component_cache is not None:
            if self.reference_component_area is None:
                raise ValueError("A reference component area is required with component caching")
            mask = target.squeeze(0).detach().cpu().numpy() > 0.5
            cached = self.component_cache.get_or_create(mask)
            labels = torch.from_numpy(cached.labels).unsqueeze(0)
            item["component_labels"] = labels
            item["component_weights"] = component_weight_map(
                labels.unsqueeze(0),
                self.reference_component_area,
                self.component_alpha,
                self.component_gamma,
                self.component_weight_cap,
            ).squeeze(0)
        return item
