import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[2]
FRACTIONS = {'20160419_Houston': .0233, '20170830_Houston': .0339, '20180508_Somalia': .0043,
             '20190320_Beira': .1586, '20190329_Iran': .0353, '20190502_Canada': .0308,
             '20191012_Japan': .0053, '20210319_PortMacquarie': .0335, '20210324_Sydney': .0043,
             '20220705_Sydney': .0080, '20221013_Niger': .0107, '20231114_Beledweyne': .0045}


def load(name, filename, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    monkeypatch.syspath_prepend(str(ROOT/'src'))
    monkeypatch.setenv('PYTHONHASHSEED', '42')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rs(monkeypatch):
    return load('recipe_stability_test', 'recipe_stability.py', monkeypatch)


class Toy(torch.utils.data.Dataset):
    def __init__(self, event='synthetic'):
        self.event = event

    def __len__(self):
        return 16

    def __getitem__(self, i):
        y = torch.zeros(1, 32, 32); y[:, 8:12, 8:12] = 1
        return {'pre': torch.full((2, 32, 32), i/16), 'post': torch.ones(2, 32, 32), 'target': y,
                'valid_mask': torch.ones_like(y), 'patch_id': f'p{i}', 'event_id': self.event}


def test_dev_split_reproduces_protocol(rs):
    assert rs.dev_split(FRACTIONS) == rs.EXPECTED_DEV == ('20190502_Canada', '20191012_Japan')
    with pytest.raises(RuntimeError, match='Location groups'):
        rs.dev_split({k: v for k, v in FRACTIONS.items() if k != '20160419_Houston'})
    no_low_first = {**FRACTIONS, '20191012_Japan': .02}
    # Japan no longer low: the first all-low group in hash order (Sydney) replaces Houston.
    assert rs.dev_split(no_low_first) == ('20190502_Canada', '20210324_Sydney', '20220705_Sydney')


def test_schedule_values(rs):
    assert rs.lr_lambda('constant', 10, 150) is None
    f = rs.lr_lambda('warmup_cosine', 10, 150)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0) and f(10) == pytest.approx(1.0)
    values = [f(s) for s in range(10, 150)]
    assert all(a >= b for a, b in zip(values, values[1:])) and values[-1] < 1e-3
    with pytest.raises(ValueError):
        rs.lr_lambda('step', 10, 150)


def test_r0_workers0_equals_direction_b_runner(rs, tmp_path):
    import direction_b_factorial as db
    c = db.pf.configs()['A']
    old = db.run_cell('A', c, Toy(), Toy(), tmp_path/'old', torch.device('cpu'), 1, 42)
    new = rs.run_recipe('R0', c, Toy(), Toy(), tmp_path/'new', torch.device('cpu'), 1, 42, 3e-4, 'constant', 0)
    for key in ('initial_sha256', 'batch_order_sha256', 'optimizer_steps', 'best_score'):
        assert old[key] == new[key], key
    a = torch.load(tmp_path/'old/checkpoints/last.ckpt', weights_only=False)['model']
    b = torch.load(tmp_path/'new/checkpoints/last.ckpt', weights_only=False)['model']
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not new['validation_events_loaded'] and not new['test_used']


@pytest.mark.skipif(sys.platform == 'darwin', reason='fork-based workers are verified on the Linux server')
def test_workers_equivalent(rs, tmp_path):
    c = rs.pf.configs()['A']
    r = [rs.run_recipe('R0', c, Toy(), Toy(), tmp_path/f'w{w}', torch.device('cpu'), 1, 42, 3e-4, 'constant', w)
         for w in (0, 2)]
    for key in ('final_sha256', 'batch_order_sha256', 'best_score'):
        assert r[0][key] == r[1][key]


def test_cosine_recipe_changes_lr(rs, tmp_path):
    c = rs.pf.configs()['A']
    s = rs.run_recipe('R2', c, Toy(), Toy(), tmp_path/'r2', torch.device('cpu'), 2, 42, 3e-4, 'warmup_cosine', 0)
    rows = [json.loads(x) for x in (tmp_path/'r2/metrics.jsonl').read_text().splitlines()]
    assert rows[0]['lr_mean'] == pytest.approx(3e-4 * (0.5 + 1.0) / 2)
    assert rows[1]['lr_mean'] < 3e-4 and s['schedule'] == 'warmup_cosine'


def test_binned_ap(rs):
    pos = torch.zeros(10, dtype=torch.int64); neg = torch.zeros(10, dtype=torch.int64)
    pos[9] = 3; neg[0] = 5
    assert rs.binned_ap(pos, neg) == pytest.approx(1.0)
    pos = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]); neg = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    # Descending: TP (p=1), FP, TP -> AP = 0.5*1 + 0.5*(2/3)
    assert rs.binned_ap(pos, neg) == pytest.approx(0.5 + 1/3)
    assert rs.binned_ap(torch.zeros(10, dtype=torch.int64), neg) is None


def test_datasets_refuse_validation_events(rs, monkeypatch):
    class Fake:
        def __init__(self):
            self.records = [SimpleNamespace(patch_id=f'p{i}', event_id='20161011_Lumberton', split='train',
                                            coherence_path=None, uncertain_mask_path=None)
                            for i in range(19166)]

        def __len__(self):
            return len(self.records)

    def fake_dataset(config, split, loss):
        assert split == 'train', split
        return Fake(), 35
    monkeypatch.setattr(rs, '_dataset', fake_dataset)
    with pytest.raises(RuntimeError, match='Validation/test'):
        rs.datasets(rs.pf.configs()['A'], False)


def write_run(root, name, seed, macro, japan):
    d = root/f'{name}_s{seed}'; d.mkdir(parents=True)
    (d/'summary.json').write_text(json.dumps({'status': 'COMPLETE', 'epochs': 15, 'validation_events_loaded': False}))
    row = {'dev_event_macro_f1': macro/100, 'dev_event/20191012_Japan/f1': japan/100,
           'dev_event/20190502_Canada/f1': 0.8}
    (d/'metrics.jsonl').write_text(json.dumps(row) + '\n')


def test_decision_rules(rs, tmp_path):
    values = {'R0': [(70, 60), (80, 70), (75, 65)],   # range 10 -> unstable
              'R1': [(78, 70), (80, 72), (79, 71)],   # stable, mean 79
              'R2': [(79, 70), (81, 20), (80, 70)],   # Japan collapse -> unstable
              'R3': [(79.5, 70), (79.5, 70), (79.5, 70)]}  # stable, mean 79.5, tie with R1
    for name, vals in values.items():
        for seed, (m, j) in zip(rs.SEEDS, vals):
            write_run(tmp_path, name, seed, m, j)
    d = rs.decide(tmp_path)
    assert d['stable_recipes'] == ['R1', 'R3'] and d['selected_recipe'] == 'R1'
    for name in ('R1', 'R3'):
        (tmp_path/f'{name}_s101/metrics.jsonl').write_text(json.dumps(
            {'dev_event_macro_f1': .5, 'dev_event/20191012_Japan/f1': .7, 'dev_event/20190502_Canada/f1': .8}) + '\n')
    d = rs.decide(tmp_path)
    assert d['selected_recipe'] is None and 'stop' in d['outcome']
    assert math.isclose(d['recipes']['R1']['range_macro'], 30.0)
