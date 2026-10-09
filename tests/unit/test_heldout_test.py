import csv
import difflib
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smallflood_cd.models.registry import build_model

ROOT = Path(__file__).parents[2]


@pytest.fixture
def ht(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    monkeypatch.syspath_prepend(str(ROOT/'src'))
    monkeypatch.setenv('PYTHONHASHSEED', '42')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    spec = importlib.util.spec_from_file_location('heldout_test_t', ROOT/'scripts/heldout_test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_the_declared_lines_differ_from_evaluate(ht):
    source = inspect.getsource(ht.rpv.evaluate).splitlines()
    edited = '\n'.join(source)
    for old, new, _ in ht.SPLIT_EDITS:
        edited = edited.replace(old, new)
    changed = [line for line in difflib.ndiff(source, edited.splitlines()) if line[:2] in ('- ', '+ ')]
    assert len(changed) == 2 * sum(count for *_, count in ht.SPLIT_EDITS)
    with pytest.raises(ValueError):
        ht.evaluate_split(Path('x'), 'last.ckpt', Path('y'), None, 'train')


def synthetic_run(tmp_path):
    """Validation and test rows with real arrays; train rows point to missing files."""
    rows = []
    for split, event in (('validation', 'val_event'), ('test', 'test_event')):
        for i in range(2):
            row = {'patch_id': f'{split}{i}', 'event_id': event, 'split': split}
            target = np.zeros((1, 32, 32), dtype='float32')
            target[:, 10:14, 10:14] = i
            for field, array in {'pre': np.full((2, 32, 32), i, dtype='float32'),
                                 'post': np.ones((2, 32, 32), dtype='float32'),
                                 'label': target, 'valid_mask': np.ones_like(target)}.items():
                path = tmp_path/f'{split}{i}_{field}.npy'
                np.save(path, array)
                row[field + '_path'] = str(path)
            rows.append(row)
    rows.append({**rows[0], 'split': 'train', 'patch_id': 'train',
                 **{k: '/not-present.npy' for k in rows[0] if k.endswith('_path')}})
    manifest = tmp_path/'manifest.csv'
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats = tmp_path/'stats.json'
    stats.write_text(json.dumps({'fitted_split': 'train', 'small_area_threshold': 35}))
    config = {'base': {'seed': 42}, 'data': {'manifest': str(manifest), 'component_stats': str(stats)},
              'model': {'name': 'fc_siam_diff', 'input_channels': 2, 'widths': [2, 4, 8, 16]}}
    run = tmp_path/'run_fc_siam_diff'
    (run/'checkpoints').mkdir(parents=True)
    torch.manual_seed(0)
    torch.save({'config': config, 'model': build_model(config['model']).state_dict(), 'epoch': 14},
               run/'checkpoints'/'last.ckpt')
    (run/'environment.json').write_text(json.dumps({
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'validation_patches': 2}))
    return run


def test_validation_mode_equals_original_and_test_mode_reads_test(ht, tmp_path):
    run = synthetic_run(tmp_path)
    args = SimpleNamespace(device='cpu', batch_size=8, object_boundary=True)
    ht.rpv.evaluate(run, 'last.ckpt', tmp_path/'orig', args)
    ht.evaluate_split(run, 'last.ckpt', tmp_path/'new', args, 'validation')
    a = json.loads((tmp_path/'orig'/run.name/'last/report.json').read_text())
    b = json.loads((tmp_path/'new'/run.name/'last/report.json').read_text())
    for key in ht.COMPARED_REPORT:
        assert a[key] == b[key], key
    assert b['split'] == 'validation' and b['test_used'] is False
    for name in ('events.csv', 'patches.csv'):
        assert (tmp_path/'orig'/run.name/'last'/name).read_text() == (tmp_path/'new'/run.name/'last'/name).read_text()
    da = json.loads((tmp_path/'orig'/run.name/'last/object_boundary_report.json').read_text())
    db = json.loads((tmp_path/'new'/run.name/'last/object_boundary_report.json').read_text())
    assert all(da[k] == db[k] for k in ht.COMPARED_DETAIL)
    ht.evaluate_split(run, 'last.ckpt', tmp_path/'test', args, 'test')
    t = json.loads((tmp_path/'test'/run.name/'last/report.json').read_text())
    assert t['split'] == 'test' and t['test_used'] is True and t['training_performed'] is False
    events = (tmp_path/'test'/run.name/'last/events.csv').read_text()
    assert 'test_event' in events and 'val_event' not in events
    den = ht.denominators(tmp_path/'test'/run.name/'last')
    assert den['patches'] == 2 and set(den['events']) == {'test_event'}


def test_denominator_consistency_and_single_use(ht, tmp_path, monkeypatch):
    root = tmp_path/'heldout'/'20261010T000000Z'
    root.mkdir(parents=True)
    current = {'patches': 5, 'events': {'e': {'valid_pixels': 10, 'positives': 2}}}
    ht.check_denominators(root, current)
    ht.check_denominators(root, current)
    with pytest.raises(RuntimeError, match='denominators changed'):
        ht.check_denominators(root, {**current, 'patches': 6})
    monkeypatch.setattr(ht, 'OUTPUT_PARENT', tmp_path/'heldout')
    ht.single_use(root)
    (tmp_path/'heldout'/'20261011T000000Z').mkdir()
    with pytest.raises(RuntimeError, match='single use'):
        ht.single_use(root)


def test_resume_skips_recorded_and_moves_partial(ht, tmp_path):
    seed_dir = tmp_path/'s42'
    (seed_dir/'run_a'/'last').mkdir(parents=True)
    (seed_dir/'run_a'/'best_composite').mkdir(parents=True)
    (seed_dir/'progress.jsonl').write_text(json.dumps({'run': 'run_a', 'checkpoint': 'last.ckpt'}) + '\n')
    assert ht.prepare_resume(seed_dir, Path('x/run_a'), 'last.ckpt') is True
    assert ht.prepare_resume(seed_dir, Path('x/run_a'), 'best_composite.ckpt') is False
    assert not (seed_dir/'run_a'/'best_composite').exists()
    assert (seed_dir/'run_a'/'best_composite.incomplete_0').exists()


def test_validated_sha_guard(ht, tmp_path):
    run = {'directory': tmp_path/'run', 'review': tmp_path/'review'}
    (run['directory']/'checkpoints').mkdir(parents=True)
    (run['review']/'last').mkdir(parents=True)
    ckpt = run['directory']/'checkpoints'/'last.ckpt'
    ckpt.write_bytes(b'weights')
    digest = hashlib.sha256(b'weights').hexdigest()
    (run['review']/'last'/'report.json').write_text(json.dumps({'split': 'validation', 'checkpoint_sha256': digest}))
    assert ht.validated_sha(run, 'last.ckpt')[0] == digest
    ckpt.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='validation record'):
        ht.validated_sha(run, 'last.ckpt')


def test_run_list_is_fixed(ht, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for seed, (root, review) in ht.CONFIRMATION.items():
        for model in ('fc_siam_diff', 'bit_sar_v2'):
            (tmp_path/root/f'pilots_s{seed}'/f'x_{model}').mkdir(parents=True)
    runs = ht.runs()
    assert len(runs) == 36
    assert {(r['recipe'], r['seed']) for r in runs} == {('old', s) for s in (42, 1337, 2026)} | {('R1', s) for s in (201, 202, 203)}
    a201 = next(r for r in runs if r['seed'] == 201 and r['arm'] == 'A')
    assert str(a201['directory']).endswith('recipe_confirmation_v1/20261007T234436Z/factorial_s201/A_proposed')
    assert str(a201['review']).endswith('recipe_confirmation_review/20261007T234436Z/s201/A_proposed')
    (tmp_path/ht.CONFIRMATION[202][0]/'pilots_s202'/'y_fc_siam_diff').mkdir()
    with pytest.raises(RuntimeError, match='Expected one'):
        ht.runs()


def fake_values(ht, override=None):
    """Synthetic last/best values in percent: default FC best, no size/boundary gains."""
    base = {'A': 60, 'B': 59, 'C': 58, 'D': 57, 'FC': 70, 'BIT': 65}
    values = {}
    for recipe, seeds in ht.RECIPES.items():
        for k, seed in enumerate(seeds):
            for arm, m in base.items():
                for ckpt in ht.CHECKPOINTS:
                    values[(recipe, arm, seed, ckpt)] = {
                        'event_macro_f1': m + k, 'pixel_f1': m, 'component_f1': 40, 'interior': 30 - (arm in 'BD'),
                        'boundary_f1': 50, 'inner_band': 30 - (arm in 'CD'),
                        'event_f1': {'20210727_Weihui': 80, '20230609_NovaKakhovka': 70,
                                     '20231201_Jubba': 40 + 10*k},
                        'event_pred_positive': {e: 1 for e in ht.TEST_EVENTS},
                        'event_gt_positive': {e: 1 for e in ht.TEST_EVENTS}}
    for key, update in (override or {}).items():
        values[key].update(update)
    return values


def test_t1_rules_and_not_evaluable(ht):
    v = fake_values(ht)
    r = ht.t1(v, 'R1')
    assert r['size_weighting']['outcome'] == 'supported'
    assert r['boundary_package']['outcome'] == 'supported'
    assert r['fc_not_worse']['outcome'] == 'supported'
    v = fake_values(ht, {('old', 'B', 1337, 'last.ckpt'): {'interior': None}})
    assert ht.t1(v, 'old')['size_weighting']['outcome'] == 'not evaluable'
    v = fake_values(ht, {('R1', 'FC', s, 'last.ckpt'): {'event_macro_f1': 10} for s in (201, 202, 203)})
    assert ht.t1(v, 'R1')['fc_not_worse']['outcome'] == 'contradicted'


def test_t2_rank_stability(ht):
    t = ht.t2(fake_values(ht), 'R1')
    assert t['pairs'] == 15 and t['stable_pairs'] == 15
    assert t['by_arm']['A']['event_macro_f1']['range'] == 2
    v = fake_values(ht, {('R1', 'A', 202, 'last.ckpt'): {'event_macro_f1': 99}})
    assert ht.t2(v, 'R1')['stable_pairs'] < 15


def test_t3_cut_points(ht):
    gt = {'20210727_Weihui': 5.0, '20230609_NovaKakhovka': 3.0, '20231201_Jubba': 0.5}
    r = ht.t3(fake_values(ht), gt)
    assert r['low_fraction_event'] == '20231201_Jubba'
    assert r['groups_with_low_event_largest'] == 12 and r['outcome'] == 'generalizes'
    gt_low_weihui = {**gt, '20210727_Weihui': 0.1}
    assert ht.t3(fake_values(ht), gt_low_weihui)['outcome'] == 'does not generalize'
    # Seven groups where Jubba is largest -> generalizes; five -> partial.
    def with_hits(n):
        v = fake_values(ht)
        groups = [(r, a) for r in ht.RECIPES for a in ht.ARMS]
        for recipe, arm in groups[n:]:
            for k, s in enumerate(ht.RECIPES[recipe]):
                v[(recipe, arm, s, 'last.ckpt')]['event_f1'] = {
                    '20210727_Weihui': 50 + 30*k, '20230609_NovaKakhovka': 70, '20231201_Jubba': 40}
        return ht.t3(v, gt)
    assert with_hits(7)['outcome'] == 'generalizes'
    assert with_hits(5)['outcome'] == 'partial'
    assert with_hits(4)['outcome'] == 'does not generalize'
