from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def label_connected_components(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Label foreground components in a 2-D binary mask without external dependencies."""
    if mask.ndim != 2:
        raise ValueError("Connected-component input must be a 2-D array")
    if connectivity not in {4, 8}:
        raise ValueError("Connectivity must be 4 or 8")
    foreground = np.asarray(mask, dtype=bool)
    labels = np.zeros(foreground.shape, dtype=np.int32)
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    height, width = foreground.shape
    next_label = 0
    for row, column in np.argwhere(foreground):
        if labels[row, column] != 0:
            continue
        next_label += 1
        labels[row, column] = next_label
        queue: deque[tuple[int, int]] = deque([(int(row), int(column))])
        while queue:
            current_row, current_column = queue.popleft()
            for row_offset, column_offset in offsets:
                neighbor_row = current_row + row_offset
                neighbor_column = current_column + column_offset
                if not (0 <= neighbor_row < height and 0 <= neighbor_column < width):
                    continue
                if not foreground[neighbor_row, neighbor_column]:
                    continue
                if labels[neighbor_row, neighbor_column] != 0:
                    continue
                labels[neighbor_row, neighbor_column] = next_label
                queue.append((neighbor_row, neighbor_column))
    return labels


def component_areas(labels: np.ndarray) -> np.ndarray:
    """Return foreground component areas ordered by component ID."""
    if labels.ndim != 2:
        raise ValueError("Component labels must be a 2-D array")
    counts = np.bincount(labels.reshape(-1).astype(np.int64))
    return counts[1:].astype(np.int64, copy=False)


def reference_component_area(label_maps: list[np.ndarray]) -> float:
    """Compute the median foreground-component area from training maps only."""
    areas = [component_areas(labels) for labels in label_maps]
    nonempty = [values for values in areas if values.size]
    if not nonempty:
        raise ValueError("At least one foreground component is required")
    return float(np.median(np.concatenate(nonempty)))


@dataclass(frozen=True)
class ComponentCacheRecord:
    labels: np.ndarray
    areas: np.ndarray
    cache_path: Path
    cache_hit: bool


class ConnectedComponentCache:
    """Content-addressed cache for deterministic connected-component targets."""

    cache_version = 1

    def __init__(self, root: str | Path, connectivity: int = 8) -> None:
        if connectivity not in {4, 8}:
            raise ValueError("Connectivity must be 4 or 8")
        self.root = Path(root)
        self.connectivity = connectivity

    def _key(self, mask: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
        digest.update(str(mask.shape).encode())
        digest.update(f"v{self.cache_version}:c{self.connectivity}".encode())
        return digest.hexdigest()

    def get_or_create(self, mask: np.ndarray) -> ComponentCacheRecord:
        binary = np.asarray(mask, dtype=np.uint8)
        key = self._key(binary)
        destination = self.root / key[:2] / f"{key}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as cached:
                return ComponentCacheRecord(
                    cached["labels"], cached["areas"], destination, True
                )
        labels = label_connected_components(binary, self.connectivity)
        areas = component_areas(labels)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.npz")
        metadata = json.dumps(
            {"version": self.cache_version, "connectivity": self.connectivity}
        )
        np.savez_compressed(temporary, labels=labels, areas=areas, metadata=metadata)
        temporary.replace(destination)
        return ComponentCacheRecord(labels, areas, destination, False)

