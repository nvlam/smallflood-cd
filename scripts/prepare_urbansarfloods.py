#!/usr/bin/env python3
"""Prepare the intensity-only binary UrbanSARFloods experiment dataset.

Requires: numpy, rasterio, scipy, PyYAML. Does not require CUDA or modify raw data.
Outputs already-normalized NPY inputs compatible with SmallFlood-CD's load_array.
Resume by rerunning the SAME command; input/config/script changes are rejected.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
import yaml


FIELDS = [
    'patch_id', 'event_id', 'pre_path', 'post_path', 'label_path',
    'valid_mask_path', 'coherence_path', 'uncertain_mask_path', 'split',
    'semantic_label_path', 'scene_id', 'source_sar', 'source_gt', 'row', 'col',
    'height', 'width', 'valid_pixels', 'flood_pixels', 'urban_flood_pixels',
]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def group_event(name):
    event = name.split('_ID_')[0]
    for suffix in ('_SAR', '_GT'):
        event = event.removesuffix(suffix)
    for prefix in ('20230805_Hebei', '20231201_Jubba'):
        if event.startswith(prefix):
            return prefix
    return event


def inventory(raw):
    development = raw / 'extracted_v1/urban_sar_floods'
    sources = []
    seen = set()

    def add(sar, gt, partition):
        if not gt.is_file() or min(sar.stat().st_size, gt.stat().st_size) == 0:
            raise ValueError(f'Missing/empty pair: {sar}, {gt}')
        scene = sar.stem.removesuffix('_SAR')
        if scene in seen:
            raise ValueError(f'Duplicate source tile name: {scene}')
        seen.add(scene)
        with rasterio.open(sar) as image, rasterio.open(gt) as label:
            if not (image.count == 8 and label.count == 1 and image.crs is not None
                    and image.crs == label.crs and image.shape == label.shape
                    and image.transform.almost_equals(label.transform)):
                raise ValueError(f'Band count or grid mismatch: {sar}')
            sources.append({
                'scene_id': scene, 'event_id': group_event(scene), 'partition': partition,
                'sar': str(sar.resolve()), 'gt': str(gt.resolve()),
                'height': image.height, 'width': image.width,
                'crs': str(image.crs), 'transform': list(image.transform)[:6],
                'sar_bytes': sar.stat().st_size, 'gt_bytes': gt.stat().st_size,
                'sar_mtime_ns': sar.stat().st_mtime_ns,
                'gt_mtime_ns': gt.stat().st_mtime_ns,
            })

    for category in ('01_NF', '02_FO', '03_FU'):
        sar_files = sorted((development / category / 'SAR').glob('*_SAR.tif'))
        gt_files = set((development / category / 'GT').glob('*.tif'))
        if not sar_files:
            raise ValueError(f'No SAR files in {development / category}')
        matched = set()
        for sar in sar_files:
            gt = development / category / 'GT' / sar.name.replace('_SAR.tif', '_GT.tif')
            add(sar, gt, 'development')
            matched.add(gt)
        if matched != gt_files:
            raise ValueError(f'Unpaired ground truth in {category}')
        print(f'Indexed {category}: {len(sar_files)} pairs', flush=True)
    folders = sorted(p for p in (raw / 'testing_case_orig').iterdir() if p.is_dir())
    if not folders:
        raise ValueError('No original test scenes found')
    for folder in folders:
        images, labels = sorted(folder.glob('*_SAR.tif')), sorted(folder.glob('*_GT.tif'))
        if len(images) != 1 or len(labels) != 1:
            raise ValueError(f'Ambiguous test scene: {folder}')
        add(images[0], labels[0], 'test')
    return sources


def choose_splits(sources, seed, fraction):
    development = {s['event_id'] for s in sources if s['partition'] == 'development'}
    test = {s['event_id'] for s in sources if s['partition'] == 'test'}
    if development & test:
        raise ValueError(f'Event leakage: {development & test}')
    if len(development) < 2:
        raise ValueError('Need at least two development events')
    ordered = sorted(development, key=lambda e: hashlib.sha256(f'{seed}:{e}'.encode()).hexdigest())
    n = min(len(ordered) - 1, max(1, math.ceil(fraction * len(ordered))))
    validation = set(ordered[:n])
    return {'train': sorted(development - validation), 'validation': sorted(validation),
            'test': sorted(test)}


def windows(source, size):
    for row in range(0, source['height'], size):
        for col in range(0, source['width'], size):
            yield Window(col, row, min(size, source['width'] - col),
                         min(size, source['height'] - row))


def read_window(image, gt, window, bands):
    values = image.read(bands, window=window, out_dtype='float32')
    labels = gt.read(1, window=window)
    declared_valid = gt.read_masks(1, window=window) > 0
    if not np.isin(labels[declared_valid], [0, 1, 2]).all():
        raise ValueError(f'Unexpected ground-truth values in {gt.name}, window {window}')
    valid = declared_valid & np.all(image.read_masks(bands, window=window) > 0, axis=0)
    valid &= np.isfinite(values).all(axis=0)
    return values, labels, valid


class Moments:
    def __init__(self):
        self.count = 0
        self.mean = np.zeros(2, dtype=np.float64)
        self.m2 = np.zeros(2, dtype=np.float64)

    def update(self, values):
        if values.shape[1] == 0:
            return
        values = values.astype(np.float64)
        count = values.shape[1]
        mean = values.mean(axis=1)
        delta = mean - self.mean
        total = self.count + count
        self.m2 += ((values - mean[:, None]) ** 2).sum(axis=1)
        self.m2 += delta**2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def result(self):
        if not self.count:
            raise ValueError('No valid training pixels')
        return {'mean': self.mean.tolist(),
                'std': np.sqrt(self.m2 / self.count).clip(1e-6).tolist(),
                'count_per_channel': self.count, 'fitted_split': 'train',
                'shared_pre_post': True, 'units': 'dB', 'already_applied_to_npy': True}


def fit_normalization(sources, bands):
    moments = Moments()
    training = [s for s in sources if s['split'] == 'train']
    for index, source in enumerate(training, 1):
        with rasterio.open(source['sar']) as image, rasterio.open(source['gt']) as gt:
            for window in windows(source, 512):
                values, _, valid = read_window(image, gt, window, bands)
                moments.update(values[:2, valid])
                moments.update(values[2:, valid])
        if index % 500 == 0:
            print(f'Normalization: {index}/{len(training)} training sources', flush=True)
    return moments.result()


def cache_components(mask, cache_root):
    """Write exactly the v1/c8 format consumed by ConnectedComponentCache."""
    binary = np.ascontiguousarray(mask, dtype=np.uint8)
    sha = hashlib.sha256(binary.tobytes())
    sha.update(str(binary.shape).encode())
    sha.update(b'v1:c8')
    key = sha.hexdigest()
    path = cache_root / key[:2] / f'{key}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as value:
            return value['areas'].copy()
    labels, _ = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    labels = labels.astype(np.int32)
    areas = np.bincount(labels.ravel())[1:].astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, labels=labels, areas=areas,
                        metadata=json.dumps({'version': 1, 'connectivity': 8}))
    temporary.replace(path)
    return areas


def convert_source(source, output, size, bands, normalization):
    status = output / '_sources' / (source['scene_id'] + '.json')
    if status.exists():
        return json.loads(status.read_text())
    folder = output / 'patches' / source['scene_id']
    folder.mkdir(parents=True, exist_ok=True)
    mean = np.asarray(normalization['mean'], dtype=np.float32)[:, None, None]
    std = np.asarray(normalization['std'], dtype=np.float32)[:, None, None]
    records, areas_hist = [], Counter()
    skipped = 0
    with rasterio.open(source['sar']) as image, rasterio.open(source['gt']) as gt:
        for window in windows(source, size):
            values, labels, valid = read_window(image, gt, window, bands)
            if not valid.any():
                skipped += 1
                continue
            h, w = valid.shape
            normalized = np.concatenate(((values[:2] - mean) / std,
                                         (values[2:] - mean) / std))
            normalized[:, ~valid] = 0
            if not np.isfinite(normalized).all():
                raise ValueError(f'Nonfinite normalized input: {source["scene_id"]}')
            pre, post = np.zeros((2, size, size), np.float32), np.zeros((2, size, size), np.float32)
            pre[:, :h, :w], post[:, :h, :w] = normalized[:2], normalized[2:]
            mask = np.zeros((1, size, size), np.uint8)
            mask[0, :h, :w] = valid
            semantic = np.full((1, size, size), 255, np.uint8)
            semantic[0, :h, :w] = np.where(valid, labels, 255).astype(np.uint8)
            target = ((semantic == 1) | (semantic == 2)).astype(np.uint8)
            patch = f'{source["scene_id"]}__r{int(window.row_off)}_c{int(window.col_off)}'
            row = {
                'patch_id': patch, 'event_id': source['event_id'], 'split': source['split'],
                'coherence_path': '', 'uncertain_mask_path': '', 'scene_id': source['scene_id'],
                'source_sar': source['sar'], 'source_gt': source['gt'],
                'row': int(window.row_off), 'col': int(window.col_off), 'height': h, 'width': w,
                'valid_pixels': int(valid.sum()), 'flood_pixels': int(target.sum()),
                'urban_flood_pixels': int((semantic == 2).sum()),
            }
            for name, array in [('pre', pre), ('post', post), ('label', target),
                                ('valid_mask', mask), ('semantic_label', semantic)]:
                path = folder / f'{patch}_{name}.npy'
                temporary = path.with_suffix('.tmp.npy')
                np.save(temporary, array, allow_pickle=False)
                temporary.replace(path)
                row[name + '_path'] = str(path.resolve())
            areas = cache_components(target[0], output / 'component_cache')
            if source['split'] == 'train':
                areas_hist.update(map(int, areas))
            records.append(row)
    result = {'rows': records, 'train_component_area_histogram': dict(areas_hist),
              'skipped_all_invalid': skipped}
    atomic_json(status, result)
    return result


def histogram_quantile(histogram, q):
    """NumPy-compatible linear quantile without expanding all component areas."""
    values = np.asarray(sorted(histogram), dtype=np.int64)
    cumulative = np.cumsum([histogram[int(v)] for v in values])
    if not len(values):
        raise ValueError('No positive components in training labels')
    rank = (int(cumulative[-1]) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    a = int(values[np.searchsorted(cumulative, lo, side='right')])
    b = int(values[np.searchsorted(cumulative, hi, side='right')])
    return a + (b - a) * (rank - lo)


def check_rows(rows, size):
    """Check every NPY header/path and event assignment before publishing manifest."""
    events = defaultdict(set)
    ids = set()
    for row in rows:
        if row['patch_id'] in ids:
            raise ValueError('Duplicate patch ID')
        ids.add(row['patch_id'])
        events[row['event_id']].add(row['split'])
        for name in ('pre', 'post', 'label', 'valid_mask', 'semantic_label'):
            array = np.load(row[name + '_path'], mmap_mode='r', allow_pickle=False)
            expected = (2 if name in ('pre', 'post') else 1, size, size)
            if array.shape != expected:
                raise ValueError(f'Bad shape: {row[name + "_path"]}')
    if any(len(splits) != 1 for splits in events.values()):
        raise ValueError('Event leakage in generated manifest')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--pre-bands', type=int, nargs=2, default=[5, 6])
    parser.add_argument('--post-bands', type=int, nargs=2, default=[7, 8])
    parser.add_argument('--small-quantile', type=float, default=0.25)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or not 0 <= args.small_quantile <= 1:
        parser.error('Invalid fraction/quantile')
    bands = args.pre_bands + args.post_bands
    if len(set(bands)) != 4 or any(b < 1 or b > 8 for b in bands) or args.patch_size <= 0:
        parser.error('Specify four different bands from 1..8 and a positive patch size')
    project = args.project_root.resolve()
    if not (project / 'pyproject.toml').is_file():
        parser.error('--project-root must contain pyproject.toml')
    output = project / 'data/processed/urbansarfloods'
    metadata = project / 'data/metadata'
    split_path = project / 'data/splits/split_v1.yaml'
    normalization_path = metadata / 'normalization_v1.json'
    components_path = metadata / 'component_stats_v1.json'
    sources = inventory(args.raw_root.resolve())
    splits = choose_splits(sources, args.seed, args.validation_fraction)
    for source in sources:
        source['split'] = next(s for s, events in splits.items() if source['event_id'] in events)
    estimate = sum(math.ceil(s['height'] / args.patch_size)
                   * math.ceil(s['width'] / args.patch_size) for s in sources)
    bytes_estimate = estimate * (19 * args.patch_size**2 + 5 * 128)
    print('Event split:', json.dumps(splits, indent=2), flush=True)
    print(f'Max patches: {estimate}; NPY estimate: {bytes_estimate / 1024**3:.1f} GiB '
          '(plus cache and metadata)', flush=True)
    print(f'Band mapping, 1-based: pre={args.pre_bands}, post={args.post_bands}', flush=True)
    if args.plan_only:
        return
    signature_input = {'sources': sources, 'splits': splits, 'bands': bands,
                       'size': args.patch_size, 'small_quantile': args.small_quantile,
                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    state = output / '_preparation_state.json'
    if state.exists():
        if json.loads(state.read_text())['signature'] != signature:
            raise ValueError('Inputs/config/script changed. Use a separate project output directory.')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Refusing to overwrite existing output: {output}')
        if any(p.exists() for p in (split_path, normalization_path, components_path)):
            raise FileExistsError('Existing split/statistics files: back up and review before proceeding')
        if shutil.disk_usage(project).free < bytes_estimate * 1.3:
            raise RuntimeError('Insufficient free space for NPY files and cache')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(state, {'signature': signature, **signature_input})
    lock = output / '_preparation.lock'
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f'Another run or stale lock exists: {lock}; inspect before retrying') from error
    try:
        (lock / 'pid').write_text(str(os.getpid()))
        with rasterio.Env(GDAL_CACHEMAX=256 * 1024 * 1024):
            norm_work = output / '_normalization.json'
            if norm_work.exists():
                normalization = json.loads(norm_work.read_text())
            else:
                normalization = fit_normalization(sources, bands)
                normalization.update({'pre_bands_1based': args.pre_bands,
                                      'post_bands_1based': args.post_bands})
                atomic_json(norm_work, normalization)
            rows, histogram = [], Counter()
            skipped = 0
            for index, source in enumerate(sources, 1):
                result = convert_source(source, output, args.patch_size, bands, normalization)
                rows.extend(result['rows'])
                histogram.update({int(k): v for k, v in result['train_component_area_histogram'].items()})
                skipped += result['skipped_all_invalid']
                if index % 100 == 0 or source['partition'] == 'test':
                    print(f'Prepared {index}/{len(sources)} sources; patches={len(rows)}', flush=True)
        check_rows(rows, args.patch_size)
        summary = {}
        for split in splits:
            selected = [r for r in rows if r['split'] == split]
            summary[split] = {'patches': len(selected), 'events': splits[split],
                             'valid_pixels': sum(r['valid_pixels'] for r in selected),
                             'flood_pixels': sum(r['flood_pixels'] for r in selected),
                             'urban_flood_pixels': sum(r['urban_flood_pixels'] for r in selected)}
            if not selected or summary[split]['flood_pixels'] == 0:
                raise ValueError(f'Empty split or no flood pixels: {split}')
        components = {'reference_area': histogram_quantile(histogram, 0.5),
                      'small_area_threshold': max(1, math.floor(histogram_quantile(histogram, args.small_quantile))),
                      'small_quantile': args.small_quantile, 'connectivity': 8,
                      'fitted_split': 'train',
                      'component_scope': f'{args.patch_size}-patch clipped (including tile edges)',
                      'area_histogram': dict(histogram)}
        manifest = output / 'patch_manifest.csv'
        temporary = manifest.with_suffix('.tmp.csv')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(manifest)
        atomic_json(normalization_path, normalization)
        atomic_json(components_path, components)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        temp_split = split_path.with_suffix('.tmp.yaml')
        temp_split.write_text(yaml.safe_dump({'seed': args.seed, 'policy': 'event_holdout_v1',
                                             'events': splits, 'official_train_val_repartitioned': True}))
        temp_split.replace(split_path)
        report = {'signature': signature, 'status': 'complete', 'splits': summary,
                  'manifest': str(manifest), 'skipped_all_invalid_patches': skipped,
                  'binary_target': 'FO(1) OR FU(2)', 'semantic_ignore_value': 255,
                  'inputs_normalized': True, 'coherence_ready': False,
                  'uncertain_border_masks_available': False,
                  'test_edge_policy': 'zero-pad inputs; mask padding; retain all valid pixels',
                  'evaluation_note': 'Reassemble scenes for scene-level component metrics; patch metrics clip components.'}
        atomic_json(metadata / 'preparation_report_v1.json', report)
        atomic_json(output / 'PREPARATION_COMPLETE.json', report)
        print(json.dumps(report, indent=2), flush=True)
        print('PREPARATION COMPLETE', flush=True)
    finally:
        (lock / 'pid').unlink(missing_ok=True)
        lock.rmdir()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
