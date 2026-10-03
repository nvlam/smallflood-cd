from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

from smallflood_cd.metrics.pixel_metrics import BinaryMetrics


def macro_average(metrics: list[BinaryMetrics]) -> dict[str, float]:
    if not metrics:
        raise ValueError("Cannot aggregate an empty metric list")
    values: dict[str, list[float]] = defaultdict(list)
    for item in metrics:
        for key, value in asdict(item).items():
            values[key].append(value)
    return {key: sum(items) / len(items) for key, items in values.items()}
