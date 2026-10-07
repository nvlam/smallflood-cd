"""Validation-only review and decision for recipe_confirmation_v1.

docs/recipe_confirmation_protocol_v1_20261008.md. `review` evaluates best and last of
factorial A/B/C/D, FC-Siam-Diff and BIT-SAR v2 for one seed (12 checkpoints) with the
unchanged review_pilot_validation.evaluate(); one process per seed, because evaluate()
requires PYTHONHASHSEED to equal the run seed. `decide` applies the section 8 rules to the
three reviewed seeds. Never trains, never reads test.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
import yaml
from bit_sar_v2_entry import pin_source
from review_pilot_validation import evaluate, sha256, write_csv
from review_proposed_factorial import verify_counts
import review_direction_b as rdb

PROTOCOL_ID = 'recipe_confirmation_v1'
SEEDS = (201, 202, 203)
LR = 1e-4
FULL_STEPS = 35940
EVENTS = {'lumberton': '20161011_Lumberton', 'coraki': '20220302_Coraki_Australia',
          'hebei': '20230805_Hebei'}
ARMS = ('A', 'B', 'C', 'D', 'FC', 'BIT')
# Section 8 H0 criteria, copied from recipe_stability_v1 section 5.
MAX_RANGE_PP = 5.0
LUMBERTON_FLOOR = 0.5


def factorial_jobs(root, seed):
    complete = json.loads((root/'COMPLETE.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    if (complete.get('stage') != 'run' or complete.get('test_used') is not False
            or complete.get('seed') != seed or complete.get('protocol') != PROTOCOL_ID
            or complete.get('learning_rate') != LR or provenance.get('seed') != seed):
        raise RuntimeError('Incomplete factorial run or wrong seed/protocol')
    if {k: v for k, v in provenance.items() if k not in ('seed', 'protocol')} != complete['fingerprint']:
        raise RuntimeError('Factorial provenance mismatch')
    summaries = complete['summaries']
    if [s['cell'] for s in summaries] != list('ABCD'):
        raise RuntimeError('Expected A B C D completion records')
    for s in summaries:
        if (s['status'] != 'COMPLETE' or s['epochs'] != 15 or s['optimizer_steps'] != FULL_STEPS
                or s['seed'] != seed or s['learning_rate'] != LR):
            raise RuntimeError('Incomplete cell')
        if (s['common_initial_sha256'] != summaries[0]['common_initial_sha256']
                or s['batch_order_sha256'] != summaries[0]['batch_order_sha256']):
            raise RuntimeError('Pairing mismatch')
    for name, digest in complete['fingerprint']['files'].items():
        if name.startswith('src/') and sha256(ROOT/name) != digest:
            raise RuntimeError(f'Training source changed: {name}')
    data = yaml.safe_load((root/'A_proposed/config_resolved.yaml').read_text())['data']
    for key in ('manifest', 'component_stats', 'normalization_stats'):
        if sha256(data[key]) != complete['fingerprint']['files'][key]:
            raise RuntimeError(f'Data metadata changed: {key}')
    return [root/f'{cell}_proposed' for cell in 'ABCD']


def pilot_dir(root, model, seed):
    matches = sorted(root.glob(f'*_{model}'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {model} directory under {root}')
    directory = matches[0]
    record = json.loads((directory/'recipe_confirmation.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if (record.get('protocol') != PROTOCOL_ID or record.get('stage') != 'run' or record.get('seed') != seed
            or summary.get('status') != 'COMPLETE' or summary.get('test_used') is not False):
        raise RuntimeError(f'{model} run incomplete or wrong seed/protocol')
    return directory


def jobs_for(directory, seed):
    cfg = yaml.safe_load((directory/'config_resolved.yaml').read_text())
    if cfg['training']['learning_rate'] != LR:
        raise RuntimeError(f'Learning rate is not recipe R1 in {directory}')
    return rdb.jobs_for(directory, seed)


def arm_of(run, model):
    if run.endswith('_proposed'):
        return run[0]
    return {'fc_siam_diff': 'FC', 'bit_sar_v2': 'BIT'}[model]


def epoch_rows(directory, model):
    """Per-epoch validation diagnostics from the training log (read-only)."""
    rows = []
    for r in (json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()):
        total = sum(r[f'validation_global_{k}'] for k in ('tp', 'fp', 'fn', 'tn'))
        row = {'arm': arm_of(directory.name, model), 'epoch_1based': r['epoch'] + 1,
               'train_loss': r['train_loss'], 'event_macro_f1': r['validation_event_macro_f1'],
               'predicted_positive_percent': 100*(r['validation_global_tp']+r['validation_global_fp'])/total}
        for name, event in EVENTS.items():
            row[f'{name}_f1'] = r.get(f'validation_event/{event}/f1')
        rows.append(row)
    return rows


def run(seed, factorial_root, pilot_root, output):
    directories = factorial_jobs(factorial_root, seed)
    directories += [pilot_dir(pilot_root, 'fc_siam_diff', seed), pilot_dir(pilot_root, 'bit_sar_v2', seed)]
    jobs = [job for d in directories for job in jobs_for(d, seed)]
    if len(jobs) != 12:
        raise RuntimeError('Expected 12 checkpoints')
    output.mkdir(parents=True, exist_ok=False)
    runs = {directory: model for directory, _, _, _, model in jobs}
    write_csv(output/'epochs.csv', [row for directory, model in runs.items()
                                     for row in epoch_rows(directory, model)])
    args = SimpleNamespace(device='cuda', batch_size=8, object_boundary=True)
    results = []
    print(f'OUTPUT: {output}', flush=True)
    for directory, name, expected, digest, model in jobs:
        path = directory/'checkpoints'/name
        if sha256(path) != digest:
            raise RuntimeError('Checkpoint changed before review')
        result = evaluate(directory, name, output, args)
        report = json.loads((output/directory.name/Path(name).stem/'report.json').read_text())
        verify_counts(report, expected)
        if sha256(path) != digest or report['checkpoint_sha256'] != digest:
            raise RuntimeError('Checkpoint changed during review')
        rdb.check_denominators(output, directory, name)
        results.append({'seed': seed, 'model': model, 'arm': arm_of(directory.name, model), **result})
        write_csv(output/'summary.csv', results)
        print(f'CHECKPOINT VERIFIED: {directory.name}/{name}', flush=True)
    (output/'COMPLETE.json').write_text(json.dumps({
        'protocol': PROTOCOL_ID, 'seed': seed, 'evaluations': 12, 'test_used': False,
        'training_performed': False, 'pixel_counts_match_training': True,
        'denominators_match_frozen': True, 'factorial_root': str(factorial_root),
        'pilot_root': str(pilot_root), 'script_sha256': sha256(__file__)}, indent=2))
    print(f'RECIPE CONFIRMATION REVIEW s{seed} COMPLETE: {output}', flush=True)


def number(value, what):
    if value in (None, ''):
        raise RuntimeError(f'Undefined metric needed by a decision rule: {what}; stop and report')
    return 100*float(value)


def last_values(review_root):
    """Epoch-15 values in percent: values[(arm, seed)] = {macro, lumberton, interior, inner_band}."""
    values = {}
    for seed in SEEDS:
        base = review_root/f's{seed}'
        complete = json.loads((base/'COMPLETE.json').read_text())
        if (complete.get('protocol') != PROTOCOL_ID or complete.get('seed') != seed
                or complete.get('evaluations') != 12 or complete.get('test_used') is not False
                or not complete.get('pixel_counts_match_training') or not complete.get('denominators_match_frozen')):
            raise RuntimeError(f'Review for seed {seed} incomplete or invalid')
        with (base/'summary.csv').open(newline='') as handle:
            rows = [r for r in csv.DictReader(handle) if r['checkpoint'] == 'last.ckpt']
        if sorted(r['arm'] for r in rows) != sorted(ARMS) or any(r['epoch'] != '15' for r in rows):
            raise RuntimeError(f'Seed {seed}: expected six epoch-15 last checkpoints')
        for r in rows:
            with (base/r['run']/'last'/'events.csv').open(newline='') as handle:
                lumberton = next(e for e in csv.DictReader(handle) if e['event'] == EVENTS['lumberton'])
            values[(r['arm'], seed)] = {
                'macro': number(r['event_macro_f1'], 'event_macro_f1'),
                'lumberton': number(lumberton['f1'], 'Lumberton f1'),
                'interior': number(r['object_global_small_component_recall_interior'], 'interior-small recall'),
                'inner_band': number(r['object_global_boundary_inner_band_iou'], 'inner-band IoU')}
    return values


def stability(values, arm):
    macro = [values[(arm, s)]['macro'] for s in SEEDS]
    lumberton = [values[(arm, s)]['lumberton'] for s in SEEDS]
    median = sorted(lumberton)[1]
    return {'macro': dict(zip(SEEDS, macro)), 'lumberton': dict(zip(SEEDS, lumberton)),
            'range_macro': max(macro) - min(macro), 'lumberton_median': median,
            'stable': max(macro) - min(macro) <= MAX_RANGE_PP
                      and all(x >= LUMBERTON_FLOOR * median for x in lumberton)}


def paired_rule(pairs):
    """Supported if both contrasts <= 0 in every seed, contradicted if both > 0 in every seed."""
    if all(a <= 0 and b <= 0 for a, b in pairs.values()):
        return 'supported'
    if all(a > 0 and b > 0 for a, b in pairs.values()):
        return 'contradicted'
    return 'inconsistent'


def decide(values):
    """Protocol section 8 on epoch-15 values for seeds 201-203."""
    h0 = {arm: stability(values, arm) for arm in ARMS}

    def v(arm, seed, key):
        return values[(arm, seed)][key]
    size = {s: (v('B', s, 'interior') - v('A', s, 'interior'), v('D', s, 'interior') - v('C', s, 'interior'))
            for s in SEEDS}
    boundary = {s: (v('C', s, 'inner_band') - v('A', s, 'inner_band'),
                    v('D', s, 'inner_band') - v('B', s, 'inner_band')) for s in SEEDS}
    h3 = {s: (v('FC', s, 'macro') - v('A', s, 'macro'), v('FC', s, 'macro') - v('BIT', s, 'macro'))
          for s in SEEDS}
    if all(a >= 0 and b >= 0 for a, b in h3.values()):
        h3_outcome = 'supported'
    elif all(a < 0 for a, _ in h3.values()) or all(b < 0 for _, b in h3.values()):
        h3_outcome = 'contradicted'
    else:
        h3_outcome = 'inconsistent'
    label = None if h0['A']['stable'] else 'not stable'
    return {'protocol': PROTOCOL_ID, 'checkpoint': 'last (epoch 15)', 'seeds': list(SEEDS),
            'H0_recipe_transfer_A': 'stable' if h0['A']['stable'] else 'not stable',
            'H0_by_arm': h0,
            'H1_size_weighting': {'contrasts_pp': {s: {'B-A': a, 'D-C': b} for s, (a, b) in size.items()},
                                  'outcome': paired_rule(size), 'h0_label': label},
            'H1_boundary_package': {'contrasts_pp': {s: {'C-A': a, 'D-B': b} for s, (a, b) in boundary.items()},
                                    'outcome': paired_rule(boundary), 'h0_label': label},
            'H3_fc_not_worse': {'contrasts_pp': {s: {'FC-A': a, 'FC-BIT': b} for s, (a, b) in h3.items()},
                                'outcome': h3_outcome, 'h0_label': label},
            'test_used': False}


def main():
    pin_source()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['test', 'review', 'decide'])
    p.add_argument('--seed', type=int)
    p.add_argument('--factorial-root', type=Path)
    p.add_argument('--pilot-root', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.stage == 'test':
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib', '-q', 'tests/unit/test_recipe_confirmation.py',
            'tests/unit/test_direction_b.py', 'tests/unit/test_review_proposed_factorial.py',
            'tests/unit/test_review_pilot_validation.py']))
    if a.stage == 'decide':
        if a.output is None:
            p.error('decide needs --output (the review root holding s201, s202, s203)')
        result = decide(last_values(a.output))
        (a.output/'DECISION.json').write_text(json.dumps(result, indent=2))
        (a.output/'COMPLETE.json').write_text(json.dumps({
            'protocol': PROTOCOL_ID, 'evaluations': 36, 'seeds': list(SEEDS), 'test_used': False,
            'training_performed': False, 'pixel_counts_match_training': True,
            'denominators_match_frozen': True, 'script_sha256': sha256(__file__)}, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return
    if a.seed not in SEEDS or not a.factorial_root or not a.pilot_root:
        p.error('--seed (201|202|203), --factorial-root and --pilot-root are required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for full validation review')
    output = a.output or Path('artifacts/recipe_confirmation_review')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/f's{a.seed}'
    run(a.seed, a.factorial_root, a.pilot_root, output)


if __name__ == '__main__':
    main()
