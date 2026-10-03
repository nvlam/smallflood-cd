from copy import deepcopy
from pathlib import Path
import pytest
import torch
from smallflood_cd.utils.config import load_experiment_config
from smallflood_cd.models.registry import build_model

ROOT = Path(__file__).parents[2]


def configs():
    return [load_experiment_config(ROOT / f'configs/experiment/proposed_factorial_v1/{c}.yaml')
            for c in 'abcd']


def test_only_factorial_switches_differ():
    values = configs()
    for c, (size, boundary) in zip(values, [(False, False), (True, False), (False, True), (True, True)]):
        assert c['loss']['use_size_aware'] == size
        assert c['model']['boundary_head'] == boundary
        assert c['loss']['boundary_weight'] == (0.2 if boundary else 0.)
        assert c['loss']['bce_weight'] == c['loss']['tversky_weight'] == .4
        assert c['training']['epochs'] == 15 and c['training']['batch_size'] == 8
        assert not c['base']['amp'] and c['base']['seed'] == 42
        assert not c['protocol']['test_allowed'] and not c['protocol']['execution_approved']
        del c['loss']['use_size_aware'], c['loss']['boundary_weight']
        del c['model']['boundary_head'], c['protocol']['cell']
    assert all(c == values[0] for c in values)


def test_actual_models_and_common_initialization():
    states = []
    counts = []
    for c in configs():
        torch.manual_seed(42)
        model = build_model(c['model'])
        assert (model.boundary_head is not None) == c['model']['boundary_head']
        counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        states.append({k:v.clone() for k,v in model.state_dict().items()})
    assert counts[0] == counts[1] < counts[2] == counts[3] <= 5_000_000
    for k,v in states[0].items():
        assert all(torch.equal(v, s[k]) for s in states[1:]), k
    for k in states[2]:
        assert torch.equal(states[2][k], states[3][k]), k


def test_full_cell_preserves_imported_pilot_recipe():
    c = configs()[3]
    assert c['loss'] == dict(bce_weight=.4, tversky_weight=.4, boundary_weight=.2,
        use_size_aware=True, tversky_fp=.3, tversky_fn=.7, component_alpha=1.,
        component_gamma=.5, component_weight_cap=4.)
    assert c['model']['boundary_residual_init'] == .1
