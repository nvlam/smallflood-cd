from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from smallflood_cd.metrics.boundary_metrics import BoundaryMetrics, boundary_metrics
from smallflood_cd.metrics.component_metrics import ComponentMetrics, component_metrics
from smallflood_cd.metrics.pixel_metrics import BinaryMetrics, binary_metrics


@dataclass(frozen=True)
class EvaluationReport:
    pixel: BinaryMetrics
    component: ComponentMetrics
    boundary: BoundaryMetrics

    def to_flat_dict(self) -> dict[str, float | int]:
        flattened: dict[str, float | int] = {}
        for prefix, result in (
            ("pixel", self.pixel),
            ("component", self.component),
            ("boundary", self.boundary),
        ):
            flattened.update({f"{prefix}_{key}": value for key, value in asdict(result).items()})
        return flattened


def evaluate_change_detection(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    small_area_threshold: int,
    probability_threshold: float = 0.5,
    component_match_iou: float = 0.1,
    boundary_radius: int = 1,
    boundary_tolerance: int = 2,
    connectivity: int = 8,
) -> EvaluationReport:
    return EvaluationReport(
        pixel=binary_metrics(logits, target, valid_mask, probability_threshold),
        component=component_metrics(
            logits,
            target,
            valid_mask,
            small_area_threshold,
            probability_threshold,
            component_match_iou,
            connectivity,
        ),
        boundary=boundary_metrics(
            logits,
            target,
            valid_mask,
            probability_threshold,
            boundary_radius,
            boundary_tolerance,
        ),
    )

