"""Four frozen A-control/R checkpoints, validation only; no training or tuning."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import yaml

from bit_sar_v2_entry import pin_source, verify_loaded_modules
from review_pilot_validation import evaluate, sha256, write_csv
from review_proposed_factorial import expected_epochs, verify_checkpoint, verify_counts
from smallflood_cd.metrics.object_boundary_v2 import PROTOCOL

ANCHOR = ROOT / 'configs/review/candidate_r_20261003.json'
CELLS = ('A_control', 'R')
CHECKPOINTS = ('best_composite.ckpt', 'last.ckpt')
REVIEW_DEPENDENCIES = ('scripts/review_pilot_validation.py', 'scripts/review_proposed_factorial.py',
                       'scripts/next_steps.py', 'scripts/bit_sar_v2_entry.py')


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def verify_evidence(pilot, anchor):
    for name, digest in anchor['files'].items():
        if sha256(pilot / name) != digest:
            raise RuntimeError(f'Audited training evidence changed: {name}')


def verify_sources(provenance):
    paths = [n for n in provenance['files'] if n.startswith('src/')]
    for name in [*paths, *REVIEW_DEPENDENCIES]:
        if name not in provenance['files'] or sha256(ROOT / name) != provenance['files'][name]:
            raise RuntimeError(f'Frozen source changed: {name}')
    if not paths:
        raise RuntimeError('Missing training source provenance')


def verify_metadata(config, provenance):
    for key in ('manifest', 'component_stats', 'normalization_stats'):
        if sha256(config['data'][key]) != provenance['files'][key]:
            raise RuntimeError(f'Frozen data metadata changed: {key}')


def load_verified_checkpoint(path, digest, row, config):
    # Verify the recorded hash BEFORE unpickling a trusted, self-produced checkpoint.
    if sha256(path) != digest:
        raise RuntimeError(f'Checkpoint hash mismatch: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    verify_checkpoint(checkpoint, row, config)
    if not math.isclose(checkpoint['best_score'], row['_expected_best_score'], abs_tol=1e-12, rel_tol=0):
        raise RuntimeError('Checkpoint best score mismatch')
    return checkpoint


def prepare_jobs(pilot, anchor):
    verify_evidence(pilot, anchor)
    complete, provenance = read_json(pilot / 'COMPLETE.json'), read_json(pilot / 'provenance.json')
    if (complete.get('stage') != 'run' or complete.get('test_used') is not False or
            complete.get('fingerprint') != provenance or
            [s['cell'] for s in complete['summaries']] != list(CELLS)):
        raise RuntimeError('Invalid completed A-control/R run')
    verify_sources(provenance)
    jobs = []
    first = complete['summaries'][0]
    for cell, summary in zip(CELLS, complete['summaries']):
        directory = pilot / cell
        if summary != read_json(directory / 'summary.json'):
            raise RuntimeError('Completion summary mismatch')
        if (summary['status'] != 'COMPLETE' or summary['epochs'] != 15 or
                summary['optimizer_steps'] != 35940 or summary['parameter_count'] != 1079865 or
                summary['test_used'] is not False):
            raise RuntimeError('Unexpected cell scope')
        if any(summary[k] != first[k] for k in ('initial_sha256', 'batch_order_sha256')):
            raise RuntimeError('Pairing mismatch')
        config = yaml.safe_load((directory / 'config_resolved.yaml').read_text())
        if (config['protocol']['cell'] != cell or config['model']['name'] != 'smallflood_cdnet' or
                config['model']['boundary_head'] or config['selection']['probability_threshold'] != .5):
            raise RuntimeError('Unexpected model/selection protocol')
        verify_metadata(config, provenance)
        environment = read_json(directory / 'environment.json')
        if (environment['validation_patches'] != 4767 or environment['test_used'] is not False or
                environment['manifest_sha256'] != provenance['files']['manifest']):
            raise RuntimeError('Unexpected validation environment')
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        for name, row in expected_epochs(rows).items():
            expected = {**row, '_expected_best_score': max(r['validation_selection_score']
                                                        for r in rows[:row['epoch'] + 1])}
            digest = summary['checkpoint_sha256'][name]
            checkpoint = load_verified_checkpoint(directory / 'checkpoints' / name, digest, expected, config)
            del checkpoint
            jobs.append((directory, name, expected, digest))
    return jobs, provenance


def assert_score(actual, expected, name):
    if actual is None or expected is None:
        if actual is not expected:
            raise RuntimeError(f'Undefined metric mismatch: {name}')
    elif not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0):
        raise RuntimeError(f'Validation metric mismatch: {name}; do not change threshold')


def verify_reports(destination, expected, digest, provenance):
    report = read_json(destination / 'report.json')
    if (report.get('split') != 'validation' or report.get('test_used') is not False or
            report.get('training_performed') is not False or report.get('threshold') != .5 or
            report.get('validation_patches') != 4767 or report.get('checkpoint_sha256') != digest or
            report.get('manifest_sha256') != provenance['files']['manifest']):
        raise RuntimeError('Review scope/hash mismatch')
    verify_counts(report, expected)
    for metric in ('precision', 'recall', 'f1', 'iou', 'accuracy'):
        assert_score(report['global_pixel'][metric], expected['validation_global_pixel_' + metric], metric)
        assert_score(report['event_macro'][metric], expected['validation_event_macro_' + metric], metric)
        if report['event_macro_defined_counts'][metric] != expected['validation_event_macro_' + metric + '_defined_count']:
            raise RuntimeError('Event defined count mismatch')
    with (destination / 'events.csv').open(newline='') as handle:
        events = list(csv.DictReader(handle))
    expected_events = {k.split('/')[1] for k in expected if k.startswith('validation_event/')}
    if len(events) != len(expected_events) or {e['event'] for e in events} != expected_events:
        raise RuntimeError('Review event set mismatch')
    for event in events:
        for metric in ('precision', 'recall', 'f1', 'iou', 'accuracy'):
            value = float(event[metric]) if event[metric] else None
            assert_score(value, expected[f"validation_event/{event['event']}/{metric}"], metric)
    detail = read_json(destination / 'object_boundary_report.json')
    if (detail['split'] != 'validation' or detail['test_used'] is not False or
            detail['checkpoint_sha256'] != digest or detail['protocol'] != PROTOCOL or
            detail['small_threshold_pixels'] != 35 or detail['counts']['patches'] != 4767 or
            detail['component_stats_sha256'] != provenance['files']['component_stats'] or
            detail['module_sha256'] != provenance['files']['src/smallflood_cd/metrics/object_boundary_v2.py']):
        raise RuntimeError('Object/boundary protocol mismatch')
    if {e['event'] for e in detail['events']} != expected_events:
        raise RuntimeError('Object event set mismatch')
    # GT denominators must be invariant across the four predictions/checkpoints.
    keys = ('patches', 'component_target', 'small_target_clipped', 'small_target_interior',
            'boundary_target', 'boundary_valid_pixels')
    return {e['event']: {k: e[k] for k in keys} for e in detail['events']}


def run(pilot, output):
    if output.exists() or output.resolve() == pilot.resolve() or pilot.resolve() in output.resolve().parents:
        raise RuntimeError('Use a fresh output directory outside the training run')
    anchor = read_json(ANCHOR)
    jobs, provenance = prepare_jobs(pilot, anchor)
    wrapper_hash, anchor_hash = sha256(__file__), sha256(ANCHOR)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'review_protocol.json', {
        'pilot_root': str(pilot), 'audited_archive_sha256': anchor['archive_sha256'],
        'evaluations': 4, 'split': 'validation', 'test_used': False, 'training_performed': False,
        'batch_size': 8, 'threshold': .5, 'object_boundary_protocol': PROTOCOL,
        'script_sha256': wrapper_hash, 'anchor_sha256': anchor_hash,
    })
    args = SimpleNamespace(device='cuda', batch_size=8, object_boundary=True)
    results, verifications, denominators = [], [], None
    print(f'OUTPUT: {output}', flush=True)
    for directory, name, expected, digest in jobs:
        path = directory / 'checkpoints' / name
        if sha256(path) != digest:
            raise RuntimeError('Checkpoint changed before evaluation')
        result = evaluate(directory, name, output, args)
        destination = output / directory.name / Path(name).stem
        current = verify_reports(destination, expected, digest, provenance)
        if denominators is not None and denominators != current:
            raise RuntimeError('GT denominators changed across checkpoints')
        denominators = current
        if sha256(path) != digest:
            raise RuntimeError('Checkpoint changed during evaluation')
        results.append(result)
        verifications.append(dict(cell=directory.name, checkpoint=name, epoch_0based=expected['epoch'],
                                  checkpoint_sha256=digest, pixel_counts_match_training=True,
                                  global_and_event_metrics_match_training=True))
        write_csv(output / 'summary.csv', results)
        save_json(output / 'checkpoint_verification.json', verifications)
        print(f'CHECKPOINT VERIFIED: {directory.name}/{name}', flush=True)
    verify_evidence(pilot, anchor)
    verify_sources(provenance)
    for cell in CELLS:
        verify_metadata(yaml.safe_load((pilot / cell / 'config_resolved.yaml').read_text()), provenance)
    for directory, name, _, digest in jobs:
        if sha256(directory / 'checkpoints' / name) != digest:
            raise RuntimeError('Checkpoint changed during review')
    if sha256(__file__) != wrapper_hash or sha256(ANCHOR) != anchor_hash:
        raise RuntimeError('Review wrapper/protocol changed during evaluation')
    verify_loaded_modules()
    save_json(output / 'COMPLETE.json', {
        'evaluations': 4, 'test_used': False, 'training_performed': False,
        'pixel_counts_match_training': True, 'global_and_event_metrics_match_training': True,
        'gt_denominators_identical': True, 'gt_denominators_by_event': denominators,
        'pilot_root': str(pilot), 'script_sha256': wrapper_hash,
    })
    print(f'CANDIDATE R VALIDATION REVIEW COMPLETE: {output}', flush=True)


def main():
    pin_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--pilot-root', type=Path, default=Path('runs/candidate_r_v1/20261002T150633Z'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/candidate_r_validation_review') /
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    if args.self_test:
        import pytest
        code = pytest.main(['--import-mode=importlib', '-q', 'tests/unit/test_review_candidate_r.py',
                           'tests/unit/test_review_pilot_validation.py',
                           'tests/unit/test_object_boundary_v2.py'])
        verify_loaded_modules()
        raise SystemExit(code)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for full review; no CPU fallback')
    run(args.pilot_root, args.output)


if __name__ == '__main__':
    main()
