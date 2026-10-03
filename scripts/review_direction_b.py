"""Validation-only review of one confirmation seed for direction_b_confirmation_v1.

Reviews best and last of factorial A/B/C/D, FC-Siam-Diff and BIT-SAR v2 (12 checkpoints)
with the unchanged review_pilot_validation.evaluate(). Never trains, never reads test.
One process per seed, because evaluate() requires PYTHONHASHSEED to equal the run seed.
"""
from __future__ import annotations
import argparse
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
from review_proposed_factorial import expected_epochs, verify_checkpoint, verify_counts

PROTOCOL_ID = 'direction_b_confirmation_v1'
CONFIRMATION_SEEDS = (1337, 2026)
PARAMETERS = {'smallflood_cdnet': (1079865, 1080805), 'fc_siam_diff': (487857,),
              'bit_sar_v2': (3492642,)}
# Frozen validation denominators (identical in every earlier review).
DENOMINATORS = {'patches': 4767, 'component_target': 6745, 'small_target_clipped': 1551,
                'small_target_interior': 370}
PIXELS = {'valid_pixels': 295762355, 'positives': 3182289}


def check_denominators(output, directory, checkpoint):
    base = output/directory.name/Path(checkpoint).stem
    report = json.loads((base/'report.json').read_text())
    detail = json.loads((base/'object_boundary_report.json').read_text())
    g = report['global_pixel']
    if g['valid_pixels'] != PIXELS['valid_pixels'] or g['tp']+g['fn'] != PIXELS['positives']:
        raise RuntimeError('Pixel denominators differ from frozen validation')
    for key, value in DENOMINATORS.items():
        if detail['counts'][key] != value:
            raise RuntimeError(f'Object denominator differs: {key}')
    if report['split'] != 'validation' or report['test_used'] is not False:
        raise RuntimeError('Report is not validation-only')


def factorial_jobs(root, seed):
    complete = json.loads((root/'COMPLETE.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    if (complete.get('stage') != 'run' or complete.get('test_used') is not False
            or complete.get('seed') != seed or complete.get('protocol') != PROTOCOL_ID
            or provenance.get('seed') != seed):
        raise RuntimeError('Incomplete factorial run or wrong seed/protocol')
    if {k: v for k, v in provenance.items() if k not in ('seed', 'protocol')} != complete['fingerprint']:
        raise RuntimeError('Factorial provenance mismatch')
    summaries = complete['summaries']
    if [s['cell'] for s in summaries] != list('ABCD'):
        raise RuntimeError('Expected A B C D completion records')
    for s in summaries:
        if s['status'] != 'COMPLETE' or s['epochs'] != 15 or s['optimizer_steps'] != 35940 or s['seed'] != seed:
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
    record = json.loads((directory/'direction_b.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if (record.get('protocol') != PROTOCOL_ID or record.get('seed') != seed
            or summary.get('status') != 'COMPLETE' or summary.get('test_used') is not False):
        raise RuntimeError(f'{model} run incomplete or wrong seed/protocol')
    return directory


def jobs_for(directory, seed):
    cfg = yaml.safe_load((directory/'config_resolved.yaml').read_text())
    if cfg['base']['seed'] != seed:
        raise RuntimeError(f'Seed mismatch in {directory}')
    summary = json.loads((directory/'summary.json').read_text())
    if summary.get('parameter_count') not in PARAMETERS[cfg['model']['name']]:
        raise RuntimeError(f'Parameter count mismatch in {directory}')
    rows = [json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    jobs = []
    for name, row in expected_epochs(rows).items():
        path = directory/'checkpoints'/name
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        verify_checkpoint(checkpoint, row, cfg)
        del checkpoint
        jobs.append((directory, name, row, sha256(path), cfg['model']['name']))
    return jobs


def run(seed, factorial_root, pilot_root, output):
    directories = factorial_jobs(factorial_root, seed)
    directories += [pilot_dir(pilot_root, 'fc_siam_diff', seed), pilot_dir(pilot_root, 'bit_sar_v2', seed)]
    jobs = [job for d in directories for job in jobs_for(d, seed)]
    if len(jobs) != 12:
        raise RuntimeError('Expected 12 checkpoints')
    output.mkdir(parents=True, exist_ok=False)
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
        check_denominators(output, directory, name)
        results.append({'seed': seed, 'model': model, **result})
        write_csv(output/'summary.csv', results)
        print(f'CHECKPOINT VERIFIED: {directory.name}/{name}', flush=True)
    (output/'COMPLETE.json').write_text(json.dumps({
        'protocol': PROTOCOL_ID, 'seed': seed, 'evaluations': 12, 'test_used': False,
        'training_performed': False, 'pixel_counts_match_training': True,
        'denominators_match_frozen': True, 'factorial_root': str(factorial_root),
        'pilot_root': str(pilot_root), 'script_sha256': sha256(__file__)}, indent=2))
    print(f'DIRECTION B REVIEW s{seed} COMPLETE: {output}', flush=True)


def main():
    pin_source()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--factorial-root', type=Path)
    p.add_argument('--pilot-root', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.self_test:
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib', '-q', 'tests/unit/test_direction_b.py',
            'tests/unit/test_review_proposed_factorial.py', 'tests/unit/test_review_pilot_validation.py']))
    if a.seed not in CONFIRMATION_SEEDS or not a.factorial_root or not a.pilot_root:
        p.error('--seed (1337|2026), --factorial-root and --pilot-root are required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for full validation review')
    output = a.output or Path('artifacts/direction_b_confirmation_review')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/f's{a.seed}'
    run(a.seed, a.factorial_root, a.pilot_root, output)


if __name__ == '__main__':
    main()
