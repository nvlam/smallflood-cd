"""Manual strict-reproducibility checks and train/validation-only engineering pilots."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def strict_seed(seed, deterministic=True):
    import numpy as np
    import random
    import torch

    if not deterministic:
        raise ValueError('This launcher requires deterministic=True')
    if os.environ.get('PYTHONHASHSEED') != str(seed):
        raise RuntimeError(f'Export PYTHONHASHSEED={seed} BEFORE starting Python')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('Export CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.use_deterministic_algorithms(True, warn_only=False)


def exact_equal(a, b, path='root'):
    import torch

    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or not torch.equal(a, b):
            raise AssertionError(f'Tensor mismatch: {path}')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f'Key mismatch: {path}')
        for key in a:
            exact_equal(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(f'Sequence mismatch: {path}')
        for index, (left, right) in enumerate(zip(a, b)):
            exact_equal(left, right, f'{path}.{index}')
    elif a != b:
        raise AssertionError(f'Value mismatch: {path}: {a!r} != {b!r}')


def compare(left, right):
    import torch
    from smoke_train import CONFIGS

    for model in CONFIGS:
        for filename in ('subset.json', 'metrics.jsonl'):
            a, b = left / model / filename, right / model / filename
            exact_equal(a.read_text(), b.read_text(), f'{model}/{filename}')
        for filename in ('last.ckpt', 'best_composite.ckpt'):
            a = torch.load(left / model / 'checkpoints' / filename,
                           map_location='cpu', weights_only=False)
            b = torch.load(right / model / 'checkpoints' / filename,
                           map_location='cpu', weights_only=False)
            for key in ('model', 'optimizer', 'scheduler', 'epoch', 'global_step',
                        'best_score', 'torch_rng_state', 'config'):
                exact_equal(a[key], b[key], f'{model}/{filename}/{key}')
        print(f'{model}: EXACT MATCH', flush=True)
    print('REPRODUCIBILITY CHECK PASSED (these smoke runs only)')


def band_provenance(config):
    """Check prepared-data metadata, not the binary contents of every raster."""
    path = Path(config['data']['normalization_stats'])
    raw = path.read_bytes()
    metadata = json.loads(raw)
    expected = {'pre_bands_1based': [5, 6], 'post_bands_1based': [7, 8],
                'fitted_split': 'train', 'shared_pre_post': True,
                'units': 'dB', 'already_applied_to_npy': True}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Prepared-data metadata mismatch: {key}; expected {value!r}')
    if config['data'].get('channels') != 2 or config['data'].get('include_coherence') is not False:
        raise ValueError('This pilot requires two intensity channels, no coherence')
    return {'published_band_convention_verified': True,
            'verification_scope': 'published convention and preparation metadata; not per-file audit',
            'pre_bands_1based': [5, 6], 'post_bands_1based': [7, 8],
            'channel_order': ['VH', 'VV'],
            'normalization_sha256': hashlib.sha256(raw).hexdigest(),
            'sources': [
                'https://github.com/jie666-6/UrbanSARFloods/issues/7#issuecomment-3771836010',
                'https://github.com/jie666-6/UrbanSARFloods/issues/8#issuecomment-3777054622']}


def pilot(args):
    import torch
    from torch.utils.data import DataLoader
    import yaml
    from smoke_train import CONFIGS
    from smallflood_cd.engine.experiment import _dataset
    from smallflood_cd.engine.trainer import fit
    from smallflood_cd.engine.validator import SELECTION_PROTOCOL, validate
    from smallflood_cd.losses import SmallFloodLoss
    from smallflood_cd.models.registry import build_model
    from smallflood_cd.utils.config import load_experiment_config

    if not args.engineering_only:
        raise RuntimeError('Use --engineering-only: this is a train/validation pilot, not a final experiment')
    strict_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback for this GPU pilot')
    device = torch.device('cuda')
    config = load_experiment_config(CONFIGS[args.model])
    provenance = band_provenance(config)
    config['base'].update(seed=args.seed, device='cuda', amp=False, num_workers=0)
    config['training'].update(epochs=args.epochs, batch_size=args.batch_size,
                              patience=args.epochs + 1)
    config['selection'] = dict(SELECTION_PROTOCOL)
    config['pilot'] = {'engineering_only': True, 'band_provenance': provenance,
                       'test_used': False, 'precision': 'FP32', 'scheduler': None,
                       'strict_deterministic': True, 'attention_backend': 'math',
                       'validation_aggregation': 'event_pixel_v2: pooled event counts then event macro'}
    directory = args.output_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                                    + '_' + args.model)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'config_resolved.yaml').write_text(yaml.safe_dump(config))
    train, _ = _dataset(config, 'train', config['loss'])
    val, _ = _dataset(config, 'validation', config['loss'])
    if not len(train) or not len(val):
        raise RuntimeError('Empty train/validation split')
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']))
    loss = config['loss']
    criterion = SmallFloodLoss(float(loss['bce_weight']), float(loss['tversky_weight']),
        float(loss['boundary_weight']), float(loss['tversky_fp']), float(loss['tversky_fn']))
    progress = {'steps': 0, 'epoch': 0, 'last_time': time.perf_counter()}

    def step_hook(opt, _args, _kwargs):
        grads = [p.grad for g in opt.param_groups for p in g['params'] if p.grad is not None]
        if not grads or not torch.stack([g.isfinite().all() for g in grads]).all().item():
            raise RuntimeError('Missing or nonfinite gradients; pilot stopped')
        progress['steps'] += 1
        if progress['steps'] % 100 == 0:
            print(f"{args.model}: step {progress['steps']}/{len(train_loader)*args.epochs}", flush=True)

    def validation_callback():
        metrics = validate(model, val_loader, device)
        torch.cuda.synchronize()
        now = time.perf_counter()
        progress['epoch'] += 1
        timing = {'epoch': progress['epoch'], 'cycle_seconds': now - progress['last_time'],
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3}
        progress['last_time'] = now
        with (directory / 'timing.jsonl').open('a') as handle:
            handle.write(json.dumps(timing) + '\n')
        print(json.dumps({**timing, 'validation': metrics}), flush=True)
        return metrics

    manifest = Path(config['data']['manifest'])
    environment = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(),
                   'python': sys.version, 'train_patches': len(train), 'validation_patches': len(val),
                   'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   'parameter_count': sum(p.numel() for p in model.parameters()),
                   'test_used': False, 'band_provenance': provenance,
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / 'environment.json').write_text(json.dumps(environment, indent=2))
    print(f'START {args.model}: {len(train)} train / {len(val)} validation; output={directory}', flush=True)
    hook = optimizer.register_step_pre_hook(step_hook)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = progress['last_time'] = time.perf_counter()
    result = fit(model, train_loader, val_loader, criterion, optimizer, None, device,
                 args.epochs, directory, config, patience=args.epochs + 1,
                 gradient_clip_norm=float(config['training']['gradient_clip_norm']),
                 validation_callback=validation_callback)
    torch.cuda.synchronize()
    hook.remove()
    summary = {**environment, 'status': 'COMPLETE', 'epochs': result.epochs_completed,
               'elapsed_seconds': time.perf_counter() - start,
               'checkpoint': str(result.best_checkpoint), 'engineering_only': True,
               'not_a_deployment_benchmark': True}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('smoke', help='remaining arguments forwarded to smoke_train.py')
    comparison = sub.add_parser('compare')
    comparison.add_argument('left', type=Path)
    comparison.add_argument('right', type=Path)
    p = sub.add_parser('pilot')
    p.add_argument('--model', choices=['proposed', 'fc_siam_diff', 'bit'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output-root', type=Path, default=Path('runs/pilot'))
    p.add_argument('--engineering-only', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.command == 'smoke':
        import smoke_train
        smoke_train.seed_everything = strict_seed
        sys.argv = [sys.argv[0], *remaining]
        smoke_train.main()
    else:
        if remaining:
            parser.error(f'Unexpected arguments: {remaining}')
        if args.command == 'compare':
            compare(args.left, args.right)
        else:
            if args.epochs < 1 or args.batch_size < 2:
                parser.error('epochs >= 1 and batch-size >= 2 required')
            pilot(args)


if __name__ == '__main__':
    main()
