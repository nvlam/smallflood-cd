from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from smallflood_cd.engine.validator import checkpoint_score, metrics_from_counts, summarize_events, validate
from smallflood_cd.engine.trainer import fit
from smallflood_cd.losses import SmallFloodLoss
from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput


def test_event_pooling_not_patch_average_and_global_differs():
    result = summarize_events({'a': [1, 0, 1, 98], 'b': [8, 2, 0, 0]})
    assert result['event_macro_f1'] == pytest.approx((2 / 3 + 16 / 18) / 2)
    assert result['global_pixel_f1'] == pytest.approx(18 / 21)
    assert result['selection_score'] == result['event_macro_f1']
    assert metrics_from_counts([0, 0, 0, 99])['f1'] is None
    assert metrics_from_counts([0, 1, 0, 99])['f1'] == 0
    r = summarize_events({'a': [1, 0, 0, 0], 'empty': [0, 0, 0, 99]})
    assert r['event_macro_f1_defined_count'] == 1
    assert r['event_macro_f1'] == 1
    with pytest.raises(ValueError):
        summarize_events({'invalid': [0, 0, 0, 0]})
    with pytest.raises(ValueError):
        checkpoint_score({'f1': 1, 'selection_score': 1})


def test_validator_mask_batch_invariance():
    class Echo(nn.Module):
        def forward(self, pre, post, coherence=None):
            return SimpleNamespace(change_logits=pre)
    records = []
    for event, logits, labels, valid in [
        ('a', [10, 10], [1, 0], [1, 0]),
        ('a', [-10, -10], [1, 0], [1, 1]),
        ('b', [10, -10], [1, 0], [1, 1]),
    ]:
        shape = lambda x: torch.tensor(x, dtype=torch.float32).reshape(1, 1, 2)
        records.append({'pre': shape(logits), 'post': shape(logits),
                        'target': shape(labels), 'valid_mask': shape(valid), 'event_id': event})
    a = validate(Echo(), DataLoader(records, batch_size=1), torch.device('cpu'))
    b = validate(Echo(), DataLoader(records, batch_size=3), torch.device('cpu'))
    assert a == b
    assert a['global_fp'] == 0
    assert a['event_macro_f1'] == pytest.approx((2 / 3 + 1) / 2)


def test_trainer_selects_event_f1_not_global_or_legacy_score(tmp_path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
        def forward(self, pre, post, coherence=None):
            logits = self.conv(pre)
            return ChangeDetectionOutput(logits, None, logits)
    model = Tiny()
    batch = {'pre': torch.ones(1, 4, 4), 'post': torch.ones(1, 4, 4),
             'target': torch.ones(1, 4, 4), 'valid_mask': torch.ones(1, 4, 4), 'event_id': 'a'}
    loader = DataLoader([batch], batch_size=1)
    reports = iter([
        {'event_macro_f1': 0.8, 'global_pixel_f1': 0.3, 'selection_score': 0.1},
        {'event_macro_f1': 0.6, 'global_pixel_f1': 0.9, 'selection_score': 1.0},
    ])
    config = {'selection': {'macro_f1_weight': 0.5}}
    fit(model, loader, loader, SmallFloodLoss(), torch.optim.SGD(model.parameters(), lr=0.01),
        None, torch.device('cpu'), 2, tmp_path, config, validation_callback=lambda: next(reports))
    best = torch.load(tmp_path / 'checkpoints/best_composite.ckpt', weights_only=False)
    assert best['epoch'] == 0
    assert best['best_score'] == 0.8
    assert best['config']['selection']['version'] == 'event_pixel_v2'
    assert config['selection'] == {'macro_f1_weight': 0.5}
