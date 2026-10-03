"""Dataset-specific preparation helpers. No statistics are fitted on held-out data."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


def event_group(name: str) -> str:
    """Keep separate tracks from one flood in the same partition."""
    event = Path(name).stem.split('_ID_')[0]
    for suffix in ('_SAR', '_GT'):
        event = event.removesuffix(suffix)
    if event.startswith('20230805_Hebei'):
        return '20230805_Hebei'
    if event.startswith('20231201_Jubba'):
        return '20231201_Jubba'
    return event


def assign_events(events: list[str], test_events: list[str], seed: int = 42) -> dict[str, str]:
    """Deterministic 20% event holdout, selected without examining labels."""
    tests = {event_group(event) for event in test_events}
    available = sorted({event_group(event) for event in events} - tests)
    if len(available) < 2:
        raise ValueError('At least two non-test flood events are required')
    ordered = sorted(available, key=lambda e: hashlib.sha256(f'{seed}:{e}'.encode()).hexdigest())
    validation = set(ordered[:max(1, math.ceil(len(ordered) * 0.2))])
    return {e: 'test' if e in tests else 'validation' if e in validation else 'train'
            for e in sorted(set(available) | tests)}


def binary_labels(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """NF=0, FO=1, FU=2; preserve the original three-class labels separately."""
    if not np.isin(labels[valid], [0, 1, 2]).all():
        raise ValueError(f'Unexpected valid label values: {np.unique(labels[valid])}')
    return ((labels == 1) | (labels == 2)).astype(np.uint8) * valid.astype(np.uint8)


class ChannelMoments:
    """Streaming population moments, shared across pre/post for each polarization."""

    def __init__(self, channels: int = 2):
        self.count = 0
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)

    def update(self, image: np.ndarray, valid: np.ndarray):
        values = image[:, valid].astype(np.float64)
        count = values.shape[1]
        if not count:
            return
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite values in valid training input')
        mean = values.mean(axis=1)
        m2 = ((values - mean[:, None]) ** 2).sum(axis=1)
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta**2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def as_dict(self):
        if not self.count:
            raise ValueError('No valid training pixels')
        return {'mean': self.mean.tolist(),
                'std': np.sqrt(self.m2 / self.count).clip(1e-6).tolist(),
                'count_per_channel': self.count,
                'fitted_split': 'train', 'shared_pre_post': True, 'input_units': 'dB'}
