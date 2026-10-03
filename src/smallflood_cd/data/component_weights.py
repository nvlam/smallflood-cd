from __future__ import annotations

import torch


def component_weight_map(
    component_labels: torch.Tensor,
    reference_area: float,
    alpha: float = 1.0,
    gamma: float = 0.5,
    weight_cap: float = 4.0,
) -> torch.Tensor:
    """Build per-pixel weights from precomputed connected-component IDs.

    Component ID zero denotes unchanged pixels. IDs must be unique within each
    batch item; weights are calculated independently per item.
    """
    if component_labels.ndim == 3:
        component_labels = component_labels.unsqueeze(1)
    weights = torch.ones_like(component_labels, dtype=torch.float32)
    for batch_index in range(component_labels.shape[0]):
        labels = component_labels[batch_index]
        identifiers, counts = torch.unique(labels, return_counts=True)
        for identifier, area in zip(identifiers.tolist(), counts.tolist(), strict=True):
            if identifier == 0:
                continue
            factor = 1.0 + alpha * min(
                weight_cap, (reference_area / (float(area) + 1e-6)) ** gamma
            )
            weights[batch_index][labels == identifier] = factor
    return weights


def area_weight_lookup(
    component_areas: torch.Tensor,
    reference_area: float,
    alpha: float = 1.0,
    gamma: float = 0.5,
    weight_cap: float = 4.0,
) -> torch.Tensor:
    """Compute one foreground weight per component area."""
    if reference_area <= 0:
        raise ValueError("Reference area must be positive")
    if gamma < 0 or alpha < 0 or weight_cap < 0:
        raise ValueError("Weight parameters must be non-negative")
    areas = component_areas.float().clamp_min(1.0)
    enhancement = (reference_area / areas).pow(gamma).clamp(max=weight_cap)
    return 1.0 + alpha * enhancement
