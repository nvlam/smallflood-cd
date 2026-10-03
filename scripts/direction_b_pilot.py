"""Seed-parameterized FC-Siam-Diff / BIT-SAR v2 pilots for direction_b_confirmation_v1.

FC-Siam-Diff calls next_steps.pilot exactly as `next_steps.py pilot --model fc_siam_diff`
does. BIT-SAR v2 is pilot_bit_sar_v2.run_pilot with the hard-coded seed=42 replaced by
`--seed`, behind the same batch-8 gate. Train/validation only; never test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

from bit_sar_v2_entry import pin_source, verify_loaded_modules  # noqa: E402

PROTOCOL_ID = 'direction_b_confirmation_v1'
CONFIRMATION_SEEDS = (1337, 2026)
PARAMETERS = {'fc_siam_diff': 487857, 'bit_sar_v2': 3492642}


def namespace(model, seed, output_root):
    """Arguments passed to next_steps.pilot; identical to the seed-42 pilots except seed."""
    return SimpleNamespace(model=model, epochs=15, batch_size=8, seed=seed,
                           engineering_only=True, output_root=Path(output_root))


def run_fc(seed, output_root):
    import next_steps
    next_steps.pilot(namespace('fc_siam_diff', seed, output_root))


def run_bit(seed, preflight, output_root):
    import pilot_bit_sar_v2 as original
    import next_steps
    import smoke_train
    original.validate_gate(preflight)
    verify_loaded_modules()
    if 'bit_sar_v2' in smoke_train.CONFIGS:
        raise RuntimeError('Unexpected pre-existing BIT-SAR pilot mapping')
    smoke_train.CONFIGS['bit_sar_v2'] = original.CONFIG
    try:
        next_steps.pilot(namespace('bit_sar_v2', seed, output_root))
    finally:
        del smoke_train.CONFIGS['bit_sar_v2']


def check_result(model, seed, output_root):
    matches = sorted(Path(output_root).glob(f'*_{model}'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {model} run directory, found {len(matches)}')
    summary = json.loads((matches[0] / 'summary.json').read_text())
    config = (matches[0] / 'config_resolved.yaml').read_text()
    if summary.get('status') != 'COMPLETE' or summary.get('epochs') != 15:
        raise RuntimeError(f'{model} pilot incomplete')
    if summary.get('parameter_count') != PARAMETERS[model] or summary.get('test_used') is not False:
        raise RuntimeError(f'{model} parameter count/test flag mismatch')
    if f'seed: {seed}' not in config:
        raise RuntimeError(f'{model} resolved config does not record seed {seed}')
    record = {'protocol': PROTOCOL_ID, 'model': model, 'seed': seed, 'run': str(matches[0]),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'test_used': False}
    (matches[0] / 'direction_b.json').write_text(json.dumps(record, indent=2))
    return record


def main():
    pin_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['test', 'fc_siam_diff', 'bit_sar_v2'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--preflight', type=Path, help='Completed BIT-SAR v2 batch-8 gate')
    args = parser.parse_args()
    if args.command == 'test':
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib', '-q', 'tests/unit/test_direction_b.py',
                                      'tests/unit/test_bit_sar_v2_pilot.py']))
    if args.seed not in CONFIRMATION_SEEDS:
        parser.error('--seed must be 1337 or 2026')
    if args.output_root is None:
        parser.error('Provide --output-root')
    if list(args.output_root.glob(f'*_{args.command}')):
        parser.error(f'{args.output_root} already holds a {args.command} run; no reuse or resume')
    if args.command == 'bit_sar_v2':
        if args.preflight is None:
            parser.error('BIT-SAR v2 needs --preflight')
        run_bit(args.seed, args.preflight, args.output_root)
    else:
        run_fc(args.seed, args.output_root)
    verify_loaded_modules()
    record = check_result(args.command, args.seed, args.output_root)
    print(f'{args.command.upper()} s{args.seed} PILOT COMPLETE: {record["run"]}; test_used=false', flush=True)


if __name__ == '__main__':
    main()
