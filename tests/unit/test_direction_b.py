import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[2]


def load(name, filename, monkeypatch, seed='42'):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    monkeypatch.syspath_prepend(str(ROOT/'src'))
    monkeypatch.setenv('PYTHONHASHSEED', seed)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def factorial(monkeypatch):
    return load('direction_b_factorial_test', 'direction_b_factorial.py', monkeypatch)


@pytest.fixture
def original(monkeypatch):
    return load('proposed_factorial_test_b', 'proposed_factorial.py', monkeypatch)


class Toy(torch.utils.data.Dataset):
    def __init__(self, weighted):
        self.weighted = weighted
        self.records = [SimpleNamespace(patch_id=f'p{i}') for i in range(16)]

    def __len__(self):
        return 16

    def __getitem__(self, i):
        y = torch.zeros(1, 32, 32); y[:, 8:12, 8:12] = 1
        result = {'pre': torch.full((2, 32, 32), i/16), 'post': torch.ones(2, 32, 32),
                  'target': y, 'valid_mask': torch.ones_like(y), 'patch_id': f'p{i}',
                  'event_id': 'synthetic'}
        if self.weighted:
            result['component_weights'] = 1 + y*2
        return result


def test_seed_42_matches_original_runner_exactly(factorial, original, tmp_path):
    for cell, c in factorial.pf.configs().items():
        weighted = c['loss']['use_size_aware']
        old = original.run_cell(cell, c, Toy(weighted), Toy(False), tmp_path/f'old{cell}', torch.device('cpu'), 1)
        new = factorial.run_cell(cell, c, Toy(weighted), Toy(False), tmp_path/f'new{cell}', torch.device('cpu'), 1, 42)
        for key in ('initial_sha256', 'common_initial_sha256', 'batch_order_sha256', 'optimizer_steps', 'best_score'):
            assert old[key] == new[key], key
        a = torch.load(tmp_path/f'old{cell}/checkpoints/last.ckpt', weights_only=False)
        b = torch.load(tmp_path/f'new{cell}/checkpoints/last.ckpt', weights_only=False)
        assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])


def test_regression_helpers_reproduce_run_cell(factorial, tmp_path):
    c = factorial.pf.configs()['C']
    summary = factorial.run_cell('C', c, Toy(False), Toy(False), tmp_path/'c', torch.device('cpu'), 1, 42)
    assert factorial.initial_hashes(c, 42) == (summary['initial_sha256'], summary['common_initial_sha256'])
    assert factorial.order_hashes(factorial.PatchIds(Toy(False).records), 42, 1) == summary['batch_order_sha256']


def test_new_seed_changes_state_keeps_pairing(factorial, monkeypatch, tmp_path):
    monkeypatch.setenv('PYTHONHASHSEED', '1337')
    summaries = []
    for cell, c in factorial.pf.configs().items():
        s = factorial.run_cell(cell, c, Toy(c['loss']['use_size_aware']), Toy(False),
                               tmp_path/cell, torch.device('cpu'), 1, 1337)
        assert s['seed'] == 1337 and not s['test_used']
        assert 'seed: 1337' in (tmp_path/cell/'config_resolved.yaml').read_text()
        summaries.append(s)
    factorial.pf.check_pairing(summaries)
    assert summaries[0]['initial_sha256'] != factorial.ANCHOR_INITIAL['A']


def test_seed_guards(factorial, tmp_path):
    with pytest.raises(ValueError):
        factorial.check_seed(7)
    with pytest.raises(RuntimeError, match='PYTHONHASHSEED'):
        factorial.initial_hashes(factorial.pf.configs()['A'], 1337)
    gate = {'stage': 'preflight', 'fingerprint': {'v': 1}, 'test_used': False, 'seed': 2026,
            'protocol': factorial.PROTOCOL_ID,
            'summaries': [{'cell': x, 'epochs': 1, 'optimizer_steps': 2, 'status': 'COMPLETE',
                           'checkpoint_reload_exact': True, 'seed': 2026,
                           'common_initial_sha256': 'h', 'batch_order_sha256': ['o'],
                           'initial_sha256': 'i'} for x in 'ABCD']}
    (tmp_path/'COMPLETE.json').write_text(json.dumps(gate))
    factorial.verify_gate(tmp_path, {'v': 1}, 2026)
    with pytest.raises(RuntimeError, match='different seed'):
        factorial.verify_gate(tmp_path, {'v': 1}, 1337)


def test_bit_pilot_arguments_equal_original_except_seed(monkeypatch, tmp_path):
    pilot = load('direction_b_pilot_test', 'direction_b_pilot.py', monkeypatch)
    orig = load('pilot_bit_sar_v2_test_b', 'pilot_bit_sar_v2.py', monkeypatch)
    import next_steps
    captured = []
    monkeypatch.setattr(next_steps, 'pilot', lambda ns: captured.append(ns))
    monkeypatch.setattr(orig, 'validate_gate', lambda p: None)
    orig.run_pilot(tmp_path/'gate', tmp_path/'out')
    import pilot_bit_sar_v2
    monkeypatch.setattr(pilot_bit_sar_v2, 'validate_gate', lambda p: None)
    pilot.run_bit(42, tmp_path/'gate', tmp_path/'out')
    pilot.run_fc(1337, tmp_path/'out')
    assert vars(captured[0]) == vars(captured[1])
    assert vars(captured[2]) == {**vars(captured[0]), 'model': 'fc_siam_diff', 'seed': 1337}


def test_reviewer_denominator_and_seed_guards(monkeypatch, tmp_path):
    review = load('review_direction_b_test', 'review_direction_b.py', monkeypatch)
    base = tmp_path/'out'/'run'/'last'
    base.mkdir(parents=True)
    report = {'split': 'validation', 'test_used': False,
              'global_pixel': {'valid_pixels': 295762355, 'tp': 3000000, 'fn': 182289}}
    counts = dict(review.DENOMINATORS)
    (base/'report.json').write_text(json.dumps(report))
    (base/'object_boundary_report.json').write_text(json.dumps({'counts': counts}))
    review.check_denominators(tmp_path/'out', Path('run'), 'last.ckpt')
    counts['small_target_interior'] = 369
    (base/'object_boundary_report.json').write_text(json.dumps({'counts': counts}))
    with pytest.raises(RuntimeError, match='small_target_interior'):
        review.check_denominators(tmp_path/'out', Path('run'), 'last.ckpt')
    root = tmp_path/'factorial'; root.mkdir()
    (root/'COMPLETE.json').write_text(json.dumps({'stage': 'run', 'test_used': False, 'seed': 1337,
                                                  'protocol': review.PROTOCOL_ID, 'fingerprint': {}}))
    (root/'provenance.json').write_text(json.dumps({'seed': 1337, 'protocol': review.PROTOCOL_ID}))
    with pytest.raises(RuntimeError, match='wrong seed'):
        review.factorial_jobs(root, 2026)
