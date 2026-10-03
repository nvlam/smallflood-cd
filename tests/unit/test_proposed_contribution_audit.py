"""Read-only diagnostic forward/backward checks; no optimizer or real data."""
import torch

from smallflood_cd.data.boundary_supervision import build_boundary_supervision
from smallflood_cd.data.component_weights import component_weight_map
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.models.registry import build_model
from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput


def test_weights_change_both_losses_and_logit_gradients():
    labels = torch.zeros(1, 1, 32, 32, dtype=torch.long)
    labels[:, :, 4:6, 4:6] = 1
    labels[:, :, 16:24, 16:24] = 2
    target = (labels > 0).float()
    weights = component_weight_map(labels, 64, 1, .5, 4)
    assert weights[0, 0, 4, 4] > weights[0, 0, 16, 16] > 1
    criterion = SmallFloodLoss()
    gradients = []
    for current in (torch.ones_like(target), weights):
        logits = torch.full_like(target, -1., requires_grad=True)
        out = ChangeDetectionOutput(logits, None, logits)
        losses = criterion(out, target, current, torch.ones_like(target), target, target)
        pair = []
        for loss in (losses.weighted_bce, losses.weighted_tversky):
            pair.append(torch.autograd.grad(loss, logits, retain_graph=True)[0])
        gradients.append(pair)
    for plain, weighted in zip(*gradients):
        assert not torch.equal(plain, weighted)
        assert weighted[0, 0, 4, 4].abs() > weighted[0, 0, 16, 16].abs()


def test_boundary_and_segmentation_paths_receive_gradients():
    torch.manual_seed(42)
    model = build_model({'name': 'smallflood_cdnet', 'input_channels': 2,
                         'boundary_head': True, 'pretrained': False}).eval()
    target = torch.zeros(2, 1, 256, 256)
    target[:, :, 40:48, 40:48] = 1
    target[:, :, 128:192, 128:192] = 1
    valid = torch.ones_like(target)
    supervision = build_boundary_supervision(target, valid)
    out = model(torch.randn(2, 2, 256, 256), torch.randn(2, 2, 256, 256))
    loss = SmallFloodLoss()(out, target, 2 * valid, valid,
                           supervision.target, supervision.valid_mask)
    groups = {'boundary': model.boundary_head.boundary.weight,
              'features': model.boundary_head.features.parameters().__next__(),
              'residual': model.boundary_head.residual.weight,
              'scale': model.boundary_head.residual_scale}
    for term, keys in [(loss.boundary, ['boundary', 'features']),
                       (loss.weighted_bce + loss.weighted_tversky,
                        ['boundary', 'features', 'residual', 'scale'])]:
        grads = torch.autograd.grad(term, [groups[k] for k in keys], retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert not torch.equal(out.change_logits, out.preliminary_logits)


def test_diagnostic_scale_can_become_clamp_saturated():
    from smallflood_cd.models.heads.boundary_head import BoundaryRefinementHead
    head = BoundaryRefinementHead(4, residual_init=-.01).eval()
    _, residual = head(torch.randn(2, 4, 8, 8))
    residual.sum().backward()
    assert torch.count_nonzero(residual) == 0
    assert head.residual_scale.grad.item() == 0


def test_diagnostic_boundary_does_not_erode_invalid_neighborhood():
    target = torch.zeros(1, 1, 9, 9)
    target[:, :, 2:7, 2:5] = 1
    valid = torch.ones_like(target)
    valid[:, :, :, 5:] = 0
    boundary = build_boundary_supervision(target, valid)
    # This may represent a masked cut, not a physical foreground boundary.
    assert boundary.target[0, 0, 4, 4] == 1
    assert boundary.valid_mask[0, 0, 4, 4] == 1
