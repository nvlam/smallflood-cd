"""Re-evaluate trusted local pilot checkpoints on validation only; never trains."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from next_steps import strict_seed
from smallflood_cd.data.datasets import ManifestDataset
from smallflood_cd.data.preprocessing import load_array
from smallflood_cd.models.registry import build_model
from smallflood_cd.metrics import object_boundary_v2
from smallflood_cd.metrics.object_boundary_v2 import PROTOCOL, add_counts, patch_counts, summarize


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def scores(counts):
    tp, fp, fn, tn = map(int, counts)
    n = tp + fp + fn + tn
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'valid_pixels': n,
            'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'f1': ratio(2 * tp, 2 * tp + fp + fn),
            'iou': ratio(tp, tp + fp + fn), 'accuracy': ratio(tp + tn, n),
            'gt_positive_fraction': ratio(tp + fn, n),
            'pred_positive_fraction': ratio(tp + fp, n),
            'false_positive_rate': ratio(fp, fp + tn)}


def confusion(pred, truth, valid):
    p, t = pred[valid], truth[valid]
    return np.array([(p & t).sum(), (p & ~t).sum(), (~p & t).sum(),
                     (~p & ~t).sum()], dtype=np.int64)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preview(path, pre, post, truth, pred, valid):
    """Channel 1 displayed with common pre/post contrast; not physical SAR units."""
    values = np.concatenate([pre[0][valid], post[0][valid]])
    low, high = np.percentile(values, [2, 98])
    panels = []
    for array in (pre[0], post[0]):
        gray = (np.clip((array - low) / max(high - low, 1e-6), 0, 1) * 255).astype('uint8')
        panels.append(np.repeat(gray[..., None], 3, axis=-1))
    for mask in (truth, pred):
        panels.append(np.repeat((mask.astype('uint8') * 255)[..., None], 3, axis=-1))
    error = np.zeros((*truth.shape, 3), dtype='uint8')
    error[truth & pred] = [0, 200, 0]
    error[~truth & pred] = [255, 0, 0]
    error[truth & ~pred] = [0, 100, 255]
    panels.append(error)
    for panel in panels:
        panel[~valid] = [100, 100, 100]
    h, w = truth.shape
    result = Image.new('RGB', (5 * w, h + 24), 'white')
    draw = ImageDraw.Draw(result)
    for i, (panel, name) in enumerate(zip(panels, ('Pre ch1', 'Post ch1', 'GT', 'Prediction', 'TP green FP red FN blue'))):
        result.paste(Image.fromarray(panel), (i * w, 24))
        draw.text((i * w + 3, 4), name, fill='black')
    result.save(path)


@torch.inference_mode()
def evaluate(directory, checkpoint_name, output, args):
    checkpoint_path = directory / 'checkpoints' / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    strict_seed(int(config['base'].get('seed', 42)))
    manifest = Path(config['data']['manifest'])
    manifest_hash = sha256(manifest)
    original = json.loads((directory / 'environment.json').read_text())
    if manifest_hash != original['manifest_sha256']:
        raise RuntimeError(f'Manifest changed since pilot: {directory}')
    dataset = ManifestDataset(manifest, 'validation', load_array, use_uncertain_mask=False)
    dataset.records.sort(key=lambda r: (r.event_id, r.patch_id))
    if not dataset.records or len(dataset) != original['validation_patches']:
        raise RuntimeError('Validation count differs from pilot')
    if any(r.coherence_path or r.uncertain_mask_path for r in dataset.records):
        raise RuntimeError('This primary SAR review does not support coherence/uncertainty inputs')
    if len({r.patch_id for r in dataset.records}) != len(dataset):
        raise RuntimeError('Duplicate validation patch IDs')
    model_config = dict(config['model'], pretrained=False)
    model = build_model(model_config).to(args.device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    destination = output / directory.name / checkpoint_path.stem
    destination.mkdir(parents=True)
    (destination / 'images').mkdir()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    events = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    groups = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    examples, selected, patches = defaultdict(int), [], []
    object_events = defaultdict(dict)
    object_patches = []
    component_stats = Path(config['data'].get('component_stats', 'data/metadata/component_stats_v1.json'))
    small_threshold = None
    if args.object_boundary:
        stats = json.loads(component_stats.read_text())
        if stats.get('fitted_split') != 'train':
            raise RuntimeError('Component threshold must have fitted_split=train provenance')
        small_threshold = int(stats['small_area_threshold'])
    negative_patches = negative_alarm_patches = invalid_patches = 0
    for step, batch in enumerate(loader, 1):
        pre, post = batch['pre'].to(args.device), batch['post'].to(args.device)
        if not torch.isfinite(pre).all() or not torch.isfinite(post).all():
            raise RuntimeError('Nonfinite input')
        logits = model(pre, post).change_logits
        if not torch.isfinite(logits).all() or logits.shape != batch['target'].shape:
            raise RuntimeError('Invalid model output')
        prediction = (logits.sigmoid() >= 0.5).cpu().numpy()[:, 0]
        for i, event in enumerate(batch['event_id']):
            target = batch['target'][i, 0].numpy()
            valid = batch['valid_mask'][i, 0].numpy() > 0.5
            if not np.isin(target[valid], [0, 1]).all():
                raise RuntimeError('Target not binary on valid pixels')
            truth = target > 0.5
            counts = confusion(prediction[i], truth, valid)
            category = ('invalid' if not valid.any() else
                        'positive' if (truth & valid).any() else 'negative')
            events[event] += counts
            groups[category] += counts
            if category == 'invalid':
                invalid_patches += 1
            if category == 'negative':
                negative_patches += 1
                negative_alarm_patches += int(counts[1] > 0)
            patch_id = batch['patch_id'][i]
            if args.object_boundary:
                detail = patch_counts(prediction[i], truth, valid, small_threshold)
                add_counts(object_events[event], detail)
                object_patches.append({'event': event, 'patch_id': patch_id, **detail})
            patches.append({'event': event, 'patch_id': patch_id, 'category': category,
                            **scores(counts)})
            key = (event, category)
            if category != 'invalid' and examples[key] < 2:
                image_path = destination / 'images' / f'{len(selected):03d}.png'
                preview(image_path, batch['pre'][i].numpy(), batch['post'][i].numpy(),
                        truth, prediction[i], valid)
                selected.append({'event': event, 'patch_id': patch_id, 'category': category,
                                 'image': str(image_path)})
                examples[key] += 1
        if step % 100 == 0:
            print(f'{directory.name}/{checkpoint_path.stem}: {step}/{len(loader)} batches', flush=True)
    event_rows = [{'event': event, **scores(counts)} for event, counts in sorted(events.items())]
    global_scores = scores(sum(events.values(), np.zeros(4, dtype=np.int64)))
    macro, macro_n = {}, {}
    for name in ('precision', 'recall', 'f1', 'iou', 'accuracy'):
        values = [r[name] for r in event_rows if r[name] is not None]
        macro[name] = sum(values) / len(values) if values else None
        macro_n[name] = len(values)
    report = {'run_directory': str(directory), 'checkpoint': str(checkpoint_path),
              'checkpoint_sha256': sha256(checkpoint_path),
              'checkpoint_epoch_1based': int(checkpoint['epoch']) + 1,
              'split': 'validation', 'test_used': False, 'training_performed': False,
              'threshold': 0.5, 'global_pixel': global_scores,
              'event_macro': macro, 'event_macro_defined_counts': macro_n,
              'by_patch_category': {k: scores(v) for k, v in groups.items()},
              'negative_patches': negative_patches,
              'negative_patches_with_any_false_positive': negative_alarm_patches,
              'negative_patch_alarm_rate': ratio(negative_alarm_patches, negative_patches),
              'all_invalid_patches': invalid_patches, 'validation_patches': len(dataset),
              'manifest_sha256': manifest_hash, 'torch': str(torch.__version__),
              'device': args.device, 'script_sha256': sha256(__file__),
              'undefined_metric_policy': 'zero denominator = null; excluded from event macro, counts reported',
              'image_selection': 'first two positive and negative patches per event sorted by patch_id; not representative sampling',
              'scope': 'valid pixels of prepared patches; no scene reconstruction or component/boundary evaluation'}
    (destination / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (destination / 'images' / 'index.json').write_text(json.dumps(selected, indent=2))
    write_csv(destination / 'events.csv', event_rows)
    write_csv(destination / 'patches.csv', patches)
    extra = {}
    if args.object_boundary:
        detail_report = summarize(object_events)
        detail_report.update({'protocol': PROTOCOL, 'small_threshold_pixels': small_threshold,
                              'component_stats_sha256': sha256(component_stats),
                              'module_sha256': sha256(object_boundary_v2.__file__),
                              'split': 'validation', 'test_used': False,
                              'checkpoint_sha256': report['checkpoint_sha256']})
        (destination / 'object_boundary_report.json').write_text(json.dumps(detail_report, indent=2, allow_nan=False))
        write_csv(destination / 'object_boundary_events.csv', detail_report['events'])
        write_csv(destination / 'object_boundary_patch_counts.csv', object_patches)
        extra.update({f'object_global_{k}': v for k, v in detail_report['global'].items()})
        extra.update({f'object_event_macro_{k}': v for k, v in detail_report['event_macro'].items()})
    print(json.dumps({'run': directory.name, 'checkpoint': checkpoint_name,
                      'global_pixel': global_scores, 'event_macro': macro}), flush=True)
    return {'run': directory.name, 'checkpoint': checkpoint_name,
            'epoch': report['checkpoint_epoch_1based'], **global_scores,
            **{f'event_macro_{k}': v for k, v in macro.items()}, **extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/validation_review'))
    parser.add_argument('--models', nargs='+', choices=['proposed', 'fc_siam_diff', 'bit', 'bit_sar_v2'],
                        default=['proposed', 'fc_siam_diff', 'bit'])
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--object-boundary', action='store_true',
                        help='Add patch-scoped connected-component and masked boundary diagnostics')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    directories = []
    for name in args.models:
        matches = sorted(args.pilot_root.glob(f'*_{name}'))
        if len(matches) != 1:
            raise RuntimeError(f'Expected one directory for {name}, found {len(matches)}')
        for checkpoint in ('best_composite.ckpt', 'last.ckpt'):
            if not (matches[0] / 'checkpoints' / checkpoint).is_file():
                raise FileNotFoundError(matches[0] / 'checkpoints' / checkpoint)
        directories.append(matches[0])
    output = args.output_root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    print(f'OUTPUT: {output}', flush=True)
    rows = []
    for directory in directories:
        for checkpoint in ('best_composite.ckpt', 'last.ckpt'):
            rows.append(evaluate(directory, checkpoint, output, args))
            write_csv(output / 'summary.csv', rows)
    (output / 'COMPLETE.json').write_text(json.dumps({'evaluations': len(rows), 'test_used': False}))
    print(f'VALIDATION REVIEW COMPLETE: {output}', flush=True)


if __name__ == '__main__':
    main()
