"""FC-Siam-Diff / BIT-SAR v2 pilots for recipe_confirmation_v1 (phase 2 of recipe_stability).

docs/recipe_confirmation_protocol_v1_20261008.md. FC-Siam-Diff calls next_steps.pilot exactly
as direction_b_pilot does, with its config swapped for one that differs only in
learning_rate (3e-4 -> 1e-4). BIT-SAR v2 already uses 1e-4 and runs through the unchanged
direction_b_pilot.run_bit behind the same batch-8 gate. Workers stay 0. Train/validation
only; never test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

from bit_sar_v2_entry import pin_source, verify_loaded_modules  # noqa: E402
import direction_b_pilot as dbp  # noqa: E402

PROTOCOL_ID = 'recipe_confirmation_v1'
SEEDS = (201, 202, 203)
LR = 1e-4
EPOCHS = 15
FC_ORIGINAL = 'configs/experiment/fc_siam_diff.yaml'
FC_CONFIG = 'configs/experiment/recipe_confirmation_v1/fc_siam_diff.yaml'
PARAMETERS = dbp.PARAMETERS


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def check_configs():
    """The new FC config may differ from the frozen one only in learning_rate;
    BIT-SAR v2 must already be at the R1 rate."""
    import pilot_bit_sar_v2
    from smallflood_cd.utils.config import load_experiment_config
    new = load_experiment_config(ROOT / FC_CONFIG)
    old = load_experiment_config(ROOT / FC_ORIGINAL)
    if new['training']['learning_rate'] != LR or old['training']['learning_rate'] != 3e-4:
        raise RuntimeError('Unexpected FC-Siam-Diff learning rates')
    for c in (new, old):
        del c['training']['learning_rate']
    if new != old:
        raise RuntimeError('FC-Siam-Diff config differs beyond learning_rate')
    bit = load_experiment_config(ROOT / pilot_bit_sar_v2.CONFIG)
    if bit['training']['learning_rate'] != LR:
        raise RuntimeError('BIT-SAR v2 is not at the R1 learning rate')
    return {'fc_config_sha256': sha256(FC_CONFIG), 'fc_original_sha256': sha256(FC_ORIGINAL),
            'bit_config_sha256': sha256(pilot_bit_sar_v2.CONFIG)}


def namespace(model, seed, output_root, epochs):
    """direction_b_pilot arguments; only the epoch count differs (1 for the preflight)."""
    args = dbp.namespace(model, seed, output_root)
    args.epochs = epochs
    return args


def run_fc(seed, output_root, epochs):
    import next_steps
    import smoke_train
    check_configs()
    if smoke_train.CONFIGS.get('fc_siam_diff') != FC_ORIGINAL:
        raise RuntimeError('Unexpected FC-Siam-Diff config mapping')
    smoke_train.CONFIGS['fc_siam_diff'] = FC_CONFIG
    try:
        next_steps.pilot(namespace('fc_siam_diff', seed, output_root, epochs))
    finally:
        smoke_train.CONFIGS['fc_siam_diff'] = FC_ORIGINAL


def run_bit(seed, preflight, output_root):
    check_configs()
    dbp.run_bit(seed, preflight, output_root)


def check_result(model, seed, output_root, stage, configs):
    import yaml
    matches = sorted(Path(output_root).glob(f'*_{model}'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {model} run directory, found {len(matches)}')
    summary = json.loads((matches[0] / 'summary.json').read_text())
    config = yaml.safe_load((matches[0] / 'config_resolved.yaml').read_text())
    epochs = 1 if stage == 'preflight' else EPOCHS
    if summary.get('status') != 'COMPLETE' or summary.get('epochs') != epochs:
        raise RuntimeError(f'{model} {stage} incomplete')
    if summary.get('parameter_count') != PARAMETERS[model] or summary.get('test_used') is not False:
        raise RuntimeError(f'{model} parameter count/test flag mismatch')
    if config['base']['seed'] != seed or config['base']['num_workers'] != 0:
        raise RuntimeError(f'{model} resolved config seed/workers mismatch')
    if config['training']['learning_rate'] != LR or config['pilot']['scheduler'] is not None:
        raise RuntimeError(f'{model} resolved config is not recipe R1')
    record = {'protocol': PROTOCOL_ID, 'stage': stage, 'model': model, 'seed': seed,
              'run': str(matches[0]), 'learning_rate': LR, 'num_workers': 0, **configs,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'test_used': False}
    (matches[0] / 'recipe_confirmation.json').write_text(json.dumps(record, indent=2))
    return record


def verify_fc_preflight(root, seed, configs):
    matches = sorted(Path(root).glob('*_fc_siam_diff'))
    if len(matches) != 1:
        raise RuntimeError('Expected one FC-Siam-Diff preflight directory')
    record = json.loads((matches[0] / 'recipe_confirmation.json').read_text())
    if (record.get('protocol') != PROTOCOL_ID or record.get('stage') != 'preflight'
            or record.get('seed') != seed
            or any(record.get(k) != v for k, v in configs.items())):
        raise RuntimeError('Invalid or stale FC-Siam-Diff preflight')


def main():
    pin_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['test', 'fc_preflight', 'fc_siam_diff', 'bit_sar_v2'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--preflight', type=Path,
                        help='FC: completed fc_preflight root; BIT-SAR v2: completed batch-8 gate')
    args = parser.parse_args()
    if args.command == 'test':
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib', '-q', 'tests/unit/test_recipe_confirmation.py',
                                      'tests/unit/test_direction_b.py', 'tests/unit/test_bit_sar_v2_pilot.py']))
    if args.seed not in SEEDS:
        parser.error('--seed must be 201, 202 or 203')
    if args.output_root is None:
        parser.error('Provide --output-root')
    model = 'fc_siam_diff' if args.command == 'fc_preflight' else args.command
    if list(args.output_root.glob(f'*_{model}')):
        parser.error(f'{args.output_root} already holds a {model} run; no reuse or resume')
    configs = check_configs()
    if args.command == 'fc_preflight':
        run_fc(args.seed, args.output_root, 1)
    elif args.command == 'fc_siam_diff':
        if args.preflight is None:
            parser.error('FC-Siam-Diff needs --preflight')
        verify_fc_preflight(args.preflight, args.seed, configs)
        run_fc(args.seed, args.output_root, EPOCHS)
    else:
        if args.preflight is None:
            parser.error('BIT-SAR v2 needs --preflight')
        run_bit(args.seed, args.preflight, args.output_root)
    verify_loaded_modules()
    if check_configs() != configs:
        raise RuntimeError('Configs changed during the run')
    stage = 'preflight' if args.command == 'fc_preflight' else 'run'
    record = check_result(model, args.seed, args.output_root, stage, configs)
    print(f'{args.command.upper()} s{args.seed} COMPLETE: {record["run"]}; test_used=false', flush=True)


if __name__ == '__main__':
    main()
