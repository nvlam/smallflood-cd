"""Batch-8 gate followed by a fresh, fixed 15-epoch BIT-SAR v2 pilot. No test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from bit_sar_v2_entry import verify_loaded_modules
import next_steps
import smoke_train
from smoke_bit_sar_v2 import run_smoke, source_hashes
from smallflood_cd.utils.config import load_experiment_config

CONFIG = 'configs/experiment/bit_sar_v2.yaml'


def fingerprint():
    config = load_experiment_config(CONFIG)
    for section, expected in {
        'model': {'name': 'bit_sar_v2', 'input_channels': 2, 'pretrained': False},
        'training': {'epochs': 15, 'batch_size': 8, 'learning_rate': 1e-4,
                     'weight_decay': 1e-4, 'gradient_clip_norm': 1.0},
        'loss': {'bce_weight': 0.5, 'tversky_weight': 0.5, 'boundary_weight': 0.0,
                 'use_size_aware': False, 'tversky_fp': 0.3, 'tversky_fn': 0.7},
    }.items():
        for key, value in expected.items():
            if config[section].get(key) != value:
                raise ValueError(f'Frozen pilot protocol mismatch: {section}.{key}')
    sources = source_hashes()
    paths = [Path(__file__), *sorted(Path('configs').rglob('*.yaml'))]
    for path in paths:
        key = str(path.relative_to(Path.cwd()) if path.is_absolute() else path)
        sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'source_sha256': sources,
            'manifest_sha256': hashlib.sha256(Path(config['data']['manifest']).read_bytes()).hexdigest(),
            'normalization_sha256': hashlib.sha256(Path(config['data']['normalization_stats']).read_bytes()).hexdigest(),
            'resolved_config': config}


def check_batch8(directory, *, require_cuda=True):
    directory = Path(directory)
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    checks = json.loads((directory / 'checks.json').read_text())
    environment = json.loads((directory / 'environment.json').read_text())
    if complete != {'status': 'PASS', 'test_used': False}:
        raise ValueError('Batch-8 preflight did not complete')
    for key, value in {'status': 'PASS', 'model': 'bit_sar_v2', 'batch_size': 8,
                       'optimizer_steps': 2, 'checkpoint_reload_exact': True,
                       'test_used': False, 'parameter_count': 3492642}.items():
        if checks.get(key) != value:
            raise ValueError(f'Invalid batch-8 preflight: {key}')
    if require_cuda and environment.get('device') != 'cuda':
        raise ValueError('A CUDA batch-8 preflight is required')
    return checks


def validate_gate(directory):
    check_batch8(directory)
    gate = json.loads((Path(directory) / 'batch8_gate.json').read_text())
    if gate != fingerprint():
        raise ValueError('Source/config/data metadata changed since batch-8 preflight')
    return gate


def run_pilot(preflight, output_root):
    validate_gate(preflight)
    verify_loaded_modules()
    # Register this model only in this process. Old CLI choices/defaults/files stay unchanged.
    if 'bit_sar_v2' in smoke_train.CONFIGS:
        raise RuntimeError('Unexpected pre-existing BIT-SAR pilot mapping')
    smoke_train.CONFIGS['bit_sar_v2'] = CONFIG
    try:
        next_steps.pilot(SimpleNamespace(model='bit_sar_v2', epochs=15, batch_size=8,
            seed=42, engineering_only=True, output_root=Path(output_root)))
    finally:
        del smoke_train.CONFIGS['bit_sar_v2']
    # A successful return means the trainer completed; retain gate for provenance.
    print('BIT-SAR v2 PILOT COMPLETE; test_used=false', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    batch = sub.add_parser('batch8')
    batch.add_argument('--output-dir', type=Path, required=True)
    pilot = sub.add_parser('pilot')
    pilot.add_argument('--preflight', type=Path, required=True)
    pilot.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    verify_loaded_modules()
    if args.command == 'batch8':
        before = fingerprint()
        run_smoke(args.output_dir, batch_size=8)
        check_batch8(args.output_dir)
        if before != fingerprint():
            raise RuntimeError('Sources or metadata changed during batch-8 preflight')
        (args.output_dir / 'batch8_gate.json').write_text(json.dumps(before, indent=2))
        print('BATCH 8 PREFLIGHT PASSED', flush=True)
    else:
        run_pilot(args.preflight, args.output_root)


if __name__ == '__main__':
    main()
