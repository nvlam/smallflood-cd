import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[2]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    monkeypatch.setenv('PYTHONHASHSEED', '42')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    spec = importlib.util.spec_from_file_location('candidate_runner_test', ROOT / 'scripts/candidate_r_runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield module
    torch.set_num_threads(threads)


class Toy(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        y = torch.zeros(1, 32, 32)
        # One active patch guarantees one active and one empty shuffled batch.
        if i == 0:
            y[:, 8:12, 8:12] = 1
        return {'pre': torch.full((2, 32, 32), i / 16), 'post': torch.ones(2, 32, 32),
                'target': y, 'valid_mask': torch.ones_like(y),
                'patch_id': f'p{i}', 'event_id': 'synthetic'}


def checkpoint(path):
    return torch.load(path / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)


def test_configs_exact_factors_and_reject_drift(runner, monkeypatch):
    cs = runner.configs()
    a, r = cs.values()
    assert a['local_supervision']['coefficient'] == 0
    assert r['local_supervision']['coefficient'] == .1
    for key in ('base', 'model', 'loss', 'training', 'selection', 'data'):
        assert a[key] == r[key]
    original = runner.load_experiment_config
    def changed(path):
        value = original(path)
        value['training']['learning_rate'] *= 2
        return value
    monkeypatch.setattr(runner, 'load_experiment_config', changed)
    with pytest.raises(RuntimeError, match='configuration changed'):
        runner.configs()


def test_paired_cpu_smokes_repeat_and_historical_a_equivalence(runner, tmp_path):
    cs = runner.configs()
    rows = []
    for directory, cell in [('A_control', 'A_control'), ('R', 'R'), ('R_repeat', 'R')]:
        rows.append(runner.run_cell(cell, cs[cell], Toy(), Toy(), tmp_path / directory,
                                   torch.device('cpu'), 1,
                                   eligible={f'p{i}': int(i == 0) for i in range(16)}))
        metrics = json.loads((tmp_path / directory / 'metrics.jsonl').read_text())
        assert metrics['eligibility']['empty_batches'] == 1
        assert metrics['eligibility']['eligible_total'] == 1
        assert metrics['scaled_loss_terms_mean']['local'] == (
            0 if cell == 'A_control' else pytest.approx(.1 * metrics['loss_terms_mean']['local']))
        assert rows[-1]['parameter_count'] == 1079865
    runner.check_pairing(rows)
    runner.check_repeat(rows[1], rows[2])
    runner.check_repeat_checkpoints(tmp_path)
    assert rows[0]['final_sha256'] != rows[1]['final_sha256']
    runner.exact_cpu(checkpoint(tmp_path / 'R')['optimizer'], checkpoint(tmp_path / 'R_repeat')['optimizer'])
    old = runner.factorial.run_cell('A', runner.factorial.configs()['A'], Toy(), Toy(),
                                     tmp_path / 'historical_code', torch.device('cpu'), 1)
    assert old['initial_sha256'] == rows[0]['initial_sha256']
    assert old['batch_order_sha256'] == rows[0]['batch_order_sha256']
    for key in ('model', 'optimizer', 'torch_rng_state'):
        runner.exact_cpu(checkpoint(tmp_path / 'historical_code')[key],
                         checkpoint(tmp_path / 'A_control')[key])
    with pytest.raises(FileExistsError):
        runner.run_cell('R', cs['R'], Toy(), Toy(), tmp_path / 'R', torch.device('cpu'), 1)
    rows[2]['final_sha256'] = 'corrupted'
    with pytest.raises(RuntimeError, match='repeat mismatch'):
        runner.check_repeat(rows[1], rows[2])


def test_diagnostics_no_side_effects_and_empty_zero(runner):
    from smallflood_cd.models.smallflood_cdnet import ChangeDetectionOutput
    p = torch.nn.Parameter(torch.zeros(1, 1, 32, 32))
    logits = p * 2
    target = torch.zeros_like(logits)
    losses = runner.CandidateRLoss()(ChangeDetectionOutput(logits, None, logits), target, torch.ones_like(target))
    base = .4 * losses.bce + .4 * losses.tversky
    d = runner.gradient_diagnostics(base, losses.local, logits, .1)
    assert p.grad is None
    assert d['local_logit_grad_l2'] == d['applied_aux_logit_grad_l2'] == 0
    expected = torch.autograd.grad(base, p, retain_graph=True)[0]
    losses.total.backward()
    assert torch.equal(p.grad, expected)


def test_actual_eligibility_mismatch_fails(runner, tmp_path):
    with pytest.raises(RuntimeError, match='eligibility differs'):
        runner.run_cell('R', runner.configs()['R'], Toy(), Toy(), tmp_path / 'bad',
                        torch.device('cpu'), 1, eligible={f'p{i}': 999 for i in range(16)})
    assert not (tmp_path / 'bad/summary.json').exists()


def test_frozen_schedule_mismatch_fails(runner, tmp_path):
    with pytest.raises(RuntimeError, match='batch schedule'):
        runner.run_cell('R', runner.configs()['R'], Toy(), Toy(), tmp_path / 'bad',
                        torch.device('cpu'), 1,
                        expected_schedule=[dict(epoch=999, batches=2, empty_batches=1,
                                                eligible_total=1, batch_order_sha256='bad')])
    assert not (tmp_path / 'bad/summary.json').exists()


def test_checkpoint_tie_keeps_earlier(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'validate', lambda *a: {'event_macro_f1': .5})
    runner.run_cell('R', runner.configs()['R'], Toy(), Toy(), tmp_path / 'tie', torch.device('cpu'), 2)
    best = torch.load(tmp_path / 'tie/checkpoints/best_composite.ckpt', map_location='cpu', weights_only=False)
    assert best['epoch'] == 0
    assert checkpoint(tmp_path / 'tie')['epoch'] == 1


def test_datasets_never_request_test_and_enrich_preflight(runner, monkeypatch):
    calls = []
    class Fake:
        def __init__(self, split):
            self.records = [SimpleNamespace(event_id=split, patch_id=f'{split}{i:05}',
                                           coherence_path=None, uncertain_mask_path=None)
                            for i in range(19166 if split == 'train' else 4767)]
        def __len__(self):
            return len(self.records)
    def factory(config, split, loss):
        assert split in ('train', 'validation')
        calls.append(split)
        return Fake(split), 35
    monkeypatch.setattr(runner.factorial, '_dataset', factory)
    monkeypatch.setattr(runner, 'select_indices', lambda *a: list(range(8)))
    details = [dict(event_id='train', patch_id=f'train{i:05}', eligible=int(i < 8)) for i in range(19166)]
    train, val = runner.datasets(runner.configs()['A_control'], details, True)
    assert calls == ['train', 'validation']
    assert len(train) == 16 and len(val) == 8
    assert train.indices == list(range(16))
    details[0]['patch_id'] = 'wrong'
    with pytest.raises(RuntimeError, match='Train records'):
        runner.datasets(runner.configs()['A_control'], details, False)


@pytest.fixture
def audit_files(runner, monkeypatch, tmp_path):
    # Metadata fixture with the approved aggregates; no dataset arrays are read.
    details = []
    for i in range(19166):
        n = 3 if i < 681 else 2 if i < 1420 else 0
        details.append(dict(event_id='train', patch_id=f'p{i:05}', eligible=n,
                            small_clipped=n + (9976 if i == 19165 else 0), empty_rings=0,
                            eligible_pixels=n * 27 + int(i < 527)))
    schedule = [{'metadata_fixture': True}]
    monkeypatch.setattr(runner, 'batch_schedule', lambda d: schedule)
    report = dict(status='COMPLETE', train_patches=19166, test_used=False,
                  validation_arrays_used=False, training_performed=False, model_loaded=False,
                  metadata_sha256={'manifest': runner.factorial.DATA_HASHES['manifest'],
                                   'stats': runner.factorial.DATA_HASHES['component_stats']},
                  loss_source_sha256=runner.sha(ROOT / 'src/smallflood_cd/losses/local_component.py'),
                  script_sha256=runner.sha(ROOT / 'scripts/audit_candidate_r_eligibility.py'),
                  prior_A_batch_order_matches=True, epochs=schedule,
                  counts=dict(patches=19166, patches_with_eligible=1420, eligible=3521,
                              small_clipped=13497, empty_rings=0, eligible_pixels=95594,
                              foreground_pixels=23436067))
    runner.dump(tmp_path / 'report.json', report)
    runner.dump(tmp_path / 'patch_counts.json', details)
    return tmp_path, report, details


def test_audit_accepts_approved_metadata(runner, audit_files):
    path, report, details = audit_files
    assert runner.load_audit(path) == (report, details)


@pytest.mark.parametrize('key,value', [
    ('test_used', True), ('loss_source_sha256', 'drift'),
    ('prior_A_batch_order_matches', False), ('epochs', []),
])
def test_audit_rejects_scope_or_drift(runner, audit_files, key, value):
    path, report, _ = audit_files
    report[key] = value
    runner.dump(path / 'report.json', report)
    with pytest.raises(RuntimeError):
        runner.load_audit(path)


def test_gate_verifies_checkpoint_and_stale_provenance(runner, tmp_path):
    summaries = []
    for directory, cell in [('A_control', 'A_control'), ('R', 'R'), ('R_repeat', 'R')]:
        cp = tmp_path / directory / 'checkpoints'
        cp.mkdir(parents=True)
        for f in ('last.ckpt', 'best_composite.ckpt'):
            (cp / f).write_bytes(b'fixture')
        s = dict(status='COMPLETE', cell=cell, epochs=1, optimizer_steps=2, active_batches=1,
                 checkpoint_reload_exact=True, test_used=False, initial_sha256='same',
                 final_sha256=cell, diagnostics_sha256=cell, batch_order_sha256=['same'], parameter_count=1,
                 checkpoint_sha256={f: runner.sha(cp / f) for f in ('last.ckpt', 'best_composite.ckpt')})
        runner.dump(tmp_path / directory / 'summary.json', s)
        summaries.append(s)
    gate = dict(stage='preflight', fingerprint={'version': 1}, test_used=False, repeat_exact=True,
                summaries=summaries)
    runner.dump(tmp_path / 'COMPLETE.json', gate)
    runner.verify_gate(tmp_path, {'version': 1})
    with pytest.raises(RuntimeError, match='stale'):
        runner.verify_gate(tmp_path, {'version': 2})
    (tmp_path / 'R/checkpoints/last.ckpt').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='checkpoint changed'):
        runner.verify_gate(tmp_path, {'version': 1})


def test_pairing_rejects_order(runner):
    a = dict(initial_sha256='x', batch_order_sha256=['a'], parameter_count=1)
    with pytest.raises(RuntimeError, match='batch_order'):
        runner.check_pairing([a, {**a, 'batch_order_sha256': ['b']}])


@pytest.mark.parametrize('args', [[], ['--approve-training'], ['--preflight', 'missing']])
def test_full_run_needs_both_gates(runner, monkeypatch, tmp_path, args):
    monkeypatch.setattr(runner, 'pin_source', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['candidate_r_runner.py', 'run', '--output', str(tmp_path / 'new'),
                                    '--audit', str(tmp_path / 'audit'), *args])
    with pytest.raises(SystemExit):
        runner.main()
    assert not (tmp_path / 'new').exists()


def test_no_cpu_fallback(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'pin_source', lambda: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['candidate_r_runner.py', 'preflight', '--output', str(tmp_path / 'new'),
                                    '--audit', str(tmp_path / 'audit')])
    with pytest.raises(RuntimeError, match='CUDA required'):
        runner.main()
    assert not (tmp_path / 'new').exists()


def test_exact_checkpoint_dtype_and_value_checks(runner):
    runner.exact_cpu({'x': [torch.ones(2)]}, {'x': [torch.ones(2)]})
    with pytest.raises(RuntimeError):
        runner.exact_cpu(torch.ones(2), torch.zeros(2))
    with pytest.raises(RuntimeError):
        runner.exact_cpu(torch.ones(2), torch.ones(2, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_exact_cpu_gpu_checkpoint_comparison(runner):
    runner.exact_cpu({'step': torch.tensor(8.), 'state': torch.ones(3, device='cuda')},
                     {'step': torch.tensor(8.), 'state': torch.ones(3)})
