"""Candidate R: train-only, component-normalized local supervision.

Not wired into any existing runner. Eligibility matches interior-small GT in
object_boundary_patch_v2. CPU SciPy operates only on detached ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F

from smallflood_cd.losses.composite import SmallFloodLoss
from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput


@dataclass(frozen=True)
class LocalSupervision:
    positive_weights: torch.Tensor
    negative_weights: torch.Tensor
    components_per_image: tuple[int, ...]
    empty_rings: int

    @property
    def components(self) -> int:
        return sum(self.components_per_image)


def build_local_supervision(target: torch.Tensor, valid: torch.Tensor) -> LocalSupervision:
    """Fixed threshold35/radius3; average eligible components across whole batch.

    Dense weight maps are a linear-time loss reduction of per-component means;
    overlapping background rings add their coefficients, rather than deduplicate.
    Returned CPU float64 maps are label-derived constants, not model parameters.
    """
    if target.ndim != 4 or target.shape[1] != 1 or target.shape != valid.shape:
        raise ValueError('Expected equal N,1,H,W target and valid masks')
    if any(n < 1 for n in target.shape):
        raise ValueError('Empty tensor dimensions')
    if not torch.all((target == 0) | (target == 1)) or not torch.all((valid == 0) | (valid == 1)):
        raise ValueError('Target and valid must be finite binary masks')
    t = target.detach().cpu().numpy().astype(bool)
    v = valid.detach().cpu().numpy().astype(bool)
    pw, nw = np.zeros(t.shape, np.float64), np.zeros(t.shape, np.float64)
    structure = np.ones((3, 3), bool)
    counts, empty = [], 0
    for b in range(len(t)):
        foreground = t[b, 0] & v[b, 0]
        labels, number = ndi.label(foreground, structure=structure)
        sizes = np.bincount(labels.ravel(), minlength=number + 1)
        safe = ndi.binary_erosion(v[b, 0], structure=structure, border_value=0)
        excluded = set(np.unique(labels[v[b, 0] & ~safe]).tolist())
        count = 0
        for identifier, bounds in enumerate(ndi.find_objects(labels), 1):
            if bounds is None or sizes[identifier] > 35 or identifier in excluded:
                continue
            count += 1
            # Crop including complete radius3 ring; reduces CPU preparation cost.
            ys, xs = bounds
            region = (slice(max(0, ys.start - 3), min(labels.shape[0], ys.stop + 3)),
                      slice(max(0, xs.start - 3), min(labels.shape[1], xs.stop + 3)))
            component = labels[region] == identifier
            ring = (ndi.binary_dilation(component, structure=structure, iterations=3)
                    & v[b, 0][region] & ~t[b, 0][region])
            pw[b, 0][region] += component.astype(np.float64) * (.5 / sizes[identifier])
            if ring.any():
                nw[b, 0][region] += ring.astype(np.float64) * (.5 / int(ring.sum()))
            else:
                empty += 1
        counts.append(count)
    total = sum(counts)
    if total:
        pw /= total
        nw /= total
    return LocalSupervision(torch.from_numpy(pw), torch.from_numpy(nw), tuple(counts), empty)


def local_loss(logits: torch.Tensor, supervision: LocalSupervision) -> torch.Tensor:
    if logits.shape != supervision.positive_weights.shape or logits.shape != supervision.negative_weights.shape:
        raise ValueError('Logits/supervision shape mismatch')
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError('Logits must be finite floating point')
    # No eligible components -> connected differentiable zero, without overflow.
    if not supervision.components:
        return (logits * 0).sum()
    positive = supervision.positive_weights.to(device=logits.device, dtype=logits.dtype)
    negative = supervision.negative_weights.to(device=logits.device, dtype=logits.dtype)
    return (F.softplus(-logits) * positive + F.softplus(logits) * negative).sum()


class LocalComponentLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
        supervision = build_local_supervision(target, valid)
        return local_loss(logits, supervision), supervision


@dataclass
class CandidateRLossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    tversky: torch.Tensor
    local: torch.Tensor
    supervision: LocalSupervision


class CandidateRLoss(nn.Module):
    """Frozen .4 BCE + .4 Tversky + .1 local; requires head-off output."""
    def __init__(self):
        super().__init__()
        self.base = SmallFloodLoss(.4, .4, 0., .3, .7)
        self.auxiliary = LocalComponentLoss()

    def forward(self, output: ChangeDetectionOutput, target: torch.Tensor,
                valid: torch.Tensor) -> CandidateRLossOutput:
        if output.boundary_logits is not None:
            raise ValueError('Candidate R requires boundary head OFF')
        local, supervision = self.auxiliary(output.change_logits, target, valid)
        base = self.base(output, target, torch.ones_like(target), valid,
                         torch.zeros_like(target), valid)
        return CandidateRLossOutput(base.total + .1 * local, base.weighted_bce,
                                    base.weighted_tversky, local, supervision)
