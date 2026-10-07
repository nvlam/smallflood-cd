import csv
import importlib.util
import json
from pathlib import Path
import sys
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
def rc(monkeypatch):
    return load('recipe_confirmation_factorial_test', 'recipe_confirmation_factorial.py', monkeypatch)


@pytest.fixture
def review(monkeypatch):
    return load('review_recipe_confirmation_test', 'review_recipe_confirmation.py', monkeypatch)


class Toy(torch.utils.data.Dataset):
    def __init__(self, weighted=False):
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


def weights(path):
    return torch.load(path/'checkpoints/last.ckpt', weights_only=False)['model']


def test_historical_rate_equals_direction_b_runner(rc, tmp_path):
    """Seed 42 at 3e-4 with 0 workers reproduces direction_b_factorial.run_cell for every cell."""
    for cell, c in rc.pf.configs().items():
        weighted = c['loss']['use_size_aware']
        old = rc.dbf.run_cell(cell, c, Toy(weighted), Toy(), tmp_path/f'old{cell}', torch.device('cpu'), 1, 42)
        new = rc.run_cell(cell, c, Toy(weighted), Toy(), tmp_path/f'new{cell}', torch.device('cpu'), 1, 42,
                          rc.HISTORICAL_LR, 0)
        for key in ('initial_sha256', 'common_initial_sha256', 'batch_order_sha256', 'optimizer_steps', 'best_score'):
            assert old[key] == new[key], key
        a, b = weights(tmp_path/f'old{cell}'), weights(tmp_path/f'new{cell}')
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_r1_equals_recipe_stability_runner(rc, tmp_path):
    """The phase-2 R1 path equals the phase-1 code that selected R1."""
    c = rc.pf.configs()['A']
    mine = rc.run_cell('A', c, Toy(), Toy(), tmp_path/'new', torch.device('cpu'), 1, 42, rc.LR, 0)
    phase1 = rc.rs.run_recipe('R1', c, Toy(), Toy(), tmp_path/'p1', torch.device('cpu'), 1, 42,
                              *rc.rs.RECIPES['R1'], 0)
    assert rc.same_run(mine, phase1)
    assert mine['learning_rate'] == 1e-4 and mine['schedule'] == 'constant'
    resolved = (tmp_path/'new/config_resolved.yaml').read_text()
    assert 'learning_rate: 0.0001' in resolved and 'recipe: R1' in resolved
    rows = [json.loads(x) for x in (tmp_path/'new/metrics.jsonl').read_text().splitlines()]
    assert rows[0]['learning_rate'] == 1e-4


def test_rate_changes_weights_not_initial_state(rc, tmp_path):
    c = rc.pf.configs()['A']
    r0 = rc.run_cell('A', c, Toy(), Toy(), tmp_path/'r0', torch.device('cpu'), 1, 42, rc.HISTORICAL_LR, 0)
    r1 = rc.run_cell('A', c, Toy(), Toy(), tmp_path/'r1', torch.device('cpu'), 1, 42, rc.LR, 0)
    assert r0['initial_sha256'] == r1['initial_sha256']  # anchors are server-stack specific
    assert r0['final_sha256'] != r1['final_sha256']


@pytest.mark.skipif(sys.platform == 'darwin', reason='fork-based workers are verified on the Linux server')
def test_workers_equivalent_for_a_and_c(rc, tmp_path):
    for cell in rc.WORKER_CELLS:
        c = rc.pf.configs()[cell]
        runs = [rc.run_cell(cell, c, Toy(), Toy(), tmp_path/f'{cell}{w}', torch.device('cpu'), 1, 42, rc.LR, w)
                for w in (0, rc.WORKERS)]
        assert rc.same_run(*runs)


def test_new_seed_keeps_pairing(rc, monkeypatch, tmp_path):
    monkeypatch.setenv('PYTHONHASHSEED', '201')
    summaries = []
    for cell, c in rc.pf.configs().items():
        s = rc.run_cell(cell, c, Toy(c['loss']['use_size_aware']), Toy(), tmp_path/cell,
                        torch.device('cpu'), 1, 201, rc.LR, 0)
        assert s['seed'] == 201 and not s['test_used']
        assert 'seed: 201' in (tmp_path/cell/'config_resolved.yaml').read_text()
        summaries.append(s)
    rc.pf.check_pairing(summaries)
    assert summaries[0]['initial_sha256'] != rc.dbf.ANCHOR_INITIAL['A']


def test_seed_rate_and_worker_guards(rc, tmp_path):
    c = rc.pf.configs()['A']
    for seed, lr in ((201, rc.HISTORICAL_LR), (1337, rc.LR), (101, rc.LR), (42, 2e-4)):
        with pytest.raises(ValueError):
            rc.check_seed_lr(seed, lr)
    with pytest.raises(ValueError, match='num_workers'):
        rc.run_cell('A', c, Toy(), Toy(), tmp_path/'w', torch.device('cpu'), 1, 42, rc.LR, 2)


def gate(seed, workers, lr=1e-4):
    return {'stage': 'preflight', 'fingerprint': {'v': 1}, 'test_used': False, 'seed': seed,
            'protocol': 'recipe_confirmation_v1', 'learning_rate': lr, 'num_workers': workers,
            'summaries': [{'cell': x, 'epochs': 1, 'optimizer_steps': 2, 'status': 'COMPLETE',
                           'checkpoint_reload_exact': True, 'seed': seed, 'learning_rate': lr,
                           'num_workers': workers, 'common_initial_sha256': 'h',
                           'batch_order_sha256': ['o'], 'initial_sha256': 'i'} for x in 'ABCD']}


def test_gate_and_regression_records(rc, tmp_path):
    (tmp_path/'COMPLETE.json').write_text(json.dumps(gate(202, 4)))
    rc.verify_gate(tmp_path, {'v': 1}, 202, 4)
    with pytest.raises(RuntimeError, match='different seed'):
        rc.verify_gate(tmp_path, {'v': 1}, 203, 4)
    with pytest.raises(RuntimeError, match='workers'):
        rc.verify_gate(tmp_path, {'v': 1}, 202, 0)
    record = {'protocol': rc.PROTOCOL_ID, 'fingerprint': {'v': 1}, 'r1_matches_recipe_stability': True,
              'test_used': False, 'workers_equivalent': {'A': True, 'C': False}, 'num_workers': 0}
    (tmp_path/'REGRESSION_PASS.json').write_text(json.dumps(record))
    assert rc.read_regression(tmp_path, {'v': 1}) == 0
    (tmp_path/'REGRESSION_PASS.json').write_text(json.dumps({**record, 'num_workers': 4}))
    with pytest.raises(RuntimeError, match='inconsistent'):
        rc.read_regression(tmp_path, {'v': 1})
    with pytest.raises(RuntimeError, match='stale'):
        rc.read_regression(tmp_path, {'v': 2})


def test_fc_config_differs_only_in_learning_rate(monkeypatch, tmp_path):
    pilot = load('recipe_confirmation_pilot_test', 'recipe_confirmation_pilot.py', monkeypatch)
    configs = pilot.check_configs()
    assert set(configs) == {'fc_config_sha256', 'fc_original_sha256', 'bit_config_sha256'}
    import next_steps
    import smoke_train
    captured = []
    monkeypatch.setattr(next_steps, 'pilot',
                        lambda ns: captured.append((vars(ns).copy(), smoke_train.CONFIGS['fc_siam_diff'])))
    pilot.run_fc(201, tmp_path/'out', 15)
    pilot.run_fc(202, tmp_path/'out', 1)
    dbp = load('direction_b_pilot_test_rc', 'direction_b_pilot.py', monkeypatch)
    expected = vars(dbp.namespace('fc_siam_diff', 201, tmp_path/'out'))
    assert captured[0] == (expected, pilot.FC_CONFIG)
    assert captured[1] == ({**expected, 'seed': 202, 'epochs': 1}, pilot.FC_CONFIG)
    assert smoke_train.CONFIGS['fc_siam_diff'] == pilot.FC_ORIGINAL


def test_fc_preflight_record_guard(monkeypatch, tmp_path):
    pilot = load('recipe_confirmation_pilot_test2', 'recipe_confirmation_pilot.py', monkeypatch)
    configs = {'fc_config_sha256': 'a', 'fc_original_sha256': 'b', 'bit_config_sha256': 'c'}
    d = tmp_path/'x_fc_siam_diff'; d.mkdir()
    record = {'protocol': pilot.PROTOCOL_ID, 'stage': 'preflight', 'seed': 201, **configs}
    (d/'recipe_confirmation.json').write_text(json.dumps(record))
    pilot.verify_fc_preflight(tmp_path, 201, configs)
    with pytest.raises(RuntimeError, match='stale'):
        pilot.verify_fc_preflight(tmp_path, 202, configs)
    with pytest.raises(RuntimeError, match='stale'):
        pilot.verify_fc_preflight(tmp_path, 201, {**configs, 'fc_config_sha256': 'z'})


def values(macro=None, lumberton=None, interior=None, inner=None):
    """Default: all arms stable, size weighting and boundary <= 0, FC best."""
    base = {'A': (80, 60, 30, 30), 'B': (79, 58, 28, 29), 'C': (78, 59, 29, 29),
            'D': (79, 57, 27, 28), 'FC': (82, 62, 31, 40), 'BIT': (81, 61, 26, 32)}
    out = {}
    for arm, (m, lum, i, b) in base.items():
        for k, seed in enumerate((201, 202, 203)):
            out[(arm, seed)] = {'macro': m + k, 'lumberton': lum, 'interior': i, 'inner_band': b}
    for override, key in ((macro, 'macro'), (lumberton, 'lumberton'), (interior, 'interior'), (inner, 'inner_band')):
        for (arm, seed), v in (override or {}).items():
            out[(arm, seed)][key] = v
    return out


def test_decision_rules(review):
    d = review.decide(values())
    assert d['H0_recipe_transfer_A'] == 'stable' and d['H0_by_arm']['A']['range_macro'] == 2
    assert d['H1_size_weighting']['outcome'] == 'supported'
    assert d['H1_boundary_package']['outcome'] == 'supported'
    assert d['H3_fc_not_worse']['outcome'] == 'supported' and d['H3_fc_not_worse']['h0_label'] is None
    # Size weighting helps in every seed for both pairs -> contradicted.
    up = {(arm, s): 40 for arm in ('B', 'D') for s in (201, 202, 203)}
    assert review.decide(values(interior=up))['H1_size_weighting']['outcome'] == 'contradicted'
    # Helps in one seed only -> inconsistent.
    one = {('B', 202): 40, ('D', 202): 40}
    assert review.decide(values(interior=one))['H1_size_weighting']['outcome'] == 'inconsistent'
    # FC below BIT in every seed -> contradicted; below A in one seed only -> inconsistent.
    low = {('FC', s): 70 for s in (201, 202, 203)}
    assert review.decide(values(macro=low))['H3_fc_not_worse']['outcome'] == 'contradicted'
    assert review.decide(values(macro={('FC', 203): 81}))['H3_fc_not_worse']['outcome'] == 'inconsistent'


def test_h0_lumberton_floor_and_range(review):
    collapse = review.decide(values(lumberton={('A', 202): 29}))  # median 60, floor 30
    assert collapse['H0_recipe_transfer_A'] == 'not stable'
    assert collapse['H1_size_weighting']['h0_label'] == 'not stable'
    assert review.decide(values(lumberton={('A', 202): 30}))['H0_recipe_transfer_A'] == 'stable'
    wide = review.decide(values(macro={('A', 203): 85.01}))  # range 5.01 pp
    assert wide['H0_recipe_transfer_A'] == 'not stable'
    assert review.decide(values(macro={('A', 203): 85}))['H0_recipe_transfer_A'] == 'stable'


def write_seed(base, seed, rows, lumberton='0.6'):
    base.mkdir(parents=True)
    (base/'COMPLETE.json').write_text(json.dumps({
        'protocol': 'recipe_confirmation_v1', 'seed': seed, 'evaluations': 12, 'test_used': False,
        'pixel_counts_match_training': True, 'denominators_match_frozen': True}))
    fields = ['seed', 'model', 'arm', 'run', 'checkpoint', 'epoch', 'event_macro_f1',
              'object_global_small_component_recall_interior', 'object_global_boundary_inner_band_iou']
    with (base/'summary.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=fields); w.writeheader()
        for arm, run in rows:
            for ckpt, epoch in (('best_composite.ckpt', 9), ('last.ckpt', 15)):
                w.writerow({'seed': seed, 'model': 'm', 'arm': arm, 'run': run, 'checkpoint': ckpt,
                            'epoch': epoch, 'event_macro_f1': '0.8',
                            'object_global_small_component_recall_interior': '0.3',
                            'object_global_boundary_inner_band_iou': '0.3'})
            events = base/run/'last'; events.mkdir(parents=True)
            (events/'events.csv').write_text(f'event,f1\n20161011_Lumberton,{lumberton}\n20230805_Hebei,0.8\n')


def test_last_values_reads_reviews(review, tmp_path):
    rows = [(a, f'{a}_proposed') for a in 'ABCD'] + [('FC', 't_fc_siam_diff'), ('BIT', 't_bit_sar_v2')]
    for seed in (201, 202, 203):
        write_seed(tmp_path/f's{seed}', seed, rows)
    v = review.last_values(tmp_path)
    assert len(v) == 18 and v[('FC', 202)] == {'macro': 80.0, 'lumberton': 60.0, 'interior': 30.0,
                                               'inner_band': 30.0}
    (tmp_path/'s203/A_proposed/last/events.csv').write_text('event,f1\n20161011_Lumberton,\n')
    with pytest.raises(RuntimeError, match='Undefined metric'):
        review.last_values(tmp_path)


def test_reviewer_seed_and_protocol_guards(review, tmp_path):
    root = tmp_path/'factorial'; root.mkdir()
    (root/'COMPLETE.json').write_text(json.dumps({'stage': 'run', 'test_used': False, 'seed': 201,
                                                  'protocol': review.PROTOCOL_ID, 'learning_rate': 1e-4,
                                                  'fingerprint': {}}))
    (root/'provenance.json').write_text(json.dumps({'seed': 201, 'protocol': review.PROTOCOL_ID}))
    with pytest.raises(RuntimeError, match='wrong seed'):
        review.factorial_jobs(root, 202)
    pilots = tmp_path/'pilots'; d = pilots/'t_fc_siam_diff'; d.mkdir(parents=True)
    (d/'summary.json').write_text(json.dumps({'status': 'COMPLETE', 'test_used': False}))
    (d/'recipe_confirmation.json').write_text(json.dumps({'protocol': review.PROTOCOL_ID,
                                                          'stage': 'preflight', 'seed': 201}))
    with pytest.raises(RuntimeError, match='incomplete or wrong'):
        review.pilot_dir(pilots, 'fc_siam_diff', 201)
    assert review.arm_of('C_proposed', 'smallflood_cdnet') == 'C'
    assert review.arm_of('x_bit_sar_v2', 'bit_sar_v2') == 'BIT'
