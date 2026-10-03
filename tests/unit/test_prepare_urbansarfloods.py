"""Meaningful integration checks for the standalone server preparation script."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from smallflood_cd.data.connected_components import ConnectedComponentCache, label_connected_components
from smallflood_cd.data.datasets import ManifestDataset
from smallflood_cd.data.preprocessing import load_array

rasterio = pytest.importorskip('rasterio')
pytest.importorskip('scipy')
from_origin = rasterio.transform.from_origin

SCRIPT = Path(__file__).parents[2] / 'scripts/prepare_urbansarfloods.py'
spec = importlib.util.spec_from_file_location('prepare_urban', SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def write_pair(sar, gt, base, shape=(13, 17)):
    sar.parent.mkdir(parents=True, exist_ok=True)
    gt.parent.mkdir(parents=True, exist_ok=True)
    values = np.stack([np.full(shape, base + i, np.float32) for i in range(8)])
    labels = np.zeros(shape, np.uint8)
    labels[2:5, 2:5] = 1
    labels[8:11, 8:11] = 2
    profile = dict(driver='GTiff', height=shape[0], width=shape[1],
                   crs='EPSG:4326', transform=from_origin(10, 20, 0.01, 0.01))
    with rasterio.open(sar, 'w', count=8, dtype='float32', **profile) as ds:
        ds.write(values)
    with rasterio.open(gt, 'w', count=1, dtype='uint8', **profile) as ds:
        ds.write(labels, 1)


def test_cache_matches_existing_methodology(tmp_path):
    mask = np.zeros((8, 8), np.uint8)
    mask[0, 0] = mask[1, 1] = mask[7, 7] = 1
    areas = prepare.cache_components(mask, tmp_path)
    cached = ConnectedComponentCache(tmp_path).get_or_create(mask)
    assert cached.cache_hit
    np.testing.assert_array_equal(cached.labels, label_connected_components(mask))
    np.testing.assert_array_equal(areas, [2, 1])


def test_histogram_quantiles_are_numpy_linear_quantiles():
    values = np.array([1, 1, 2, 3, 50, 70, 70])
    hist = dict(zip(*np.unique(values, return_counts=True)))
    for q in (0, 0.25, 0.5, 1):
        assert prepare.histogram_quantile(hist, q) == np.quantile(values, q)


def test_preparation_end_to_end_and_resume(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
    raw = tmp_path / 'raw'
    for index, (category, event) in enumerate([
        ('01_NF', 'event_A'), ('02_FO', 'event_B'),
        ('03_FU', '20230805_Hebei_1'), ('03_FU', '20230805_Hebei_2'),
    ]):
        parent = raw / 'extracted_v1/urban_sar_floods' / category
        name = f'{event}_ID_0_0'
        write_pair(parent / 'SAR' / f'{name}_SAR.tif',
                   parent / 'GT' / f'{name}_GT.tif', index * 10)
    for event in ('20231201_Jubba_1', '20231201_Jubba_2'):
        parent = raw / 'testing_case_orig' / event
        write_pair(parent / f'{event}_SAR.tif', parent / f'{event}_GT.tif', 1000)
    command = [sys.executable, str(SCRIPT), '--raw-root', str(raw),
               '--project-root', str(project), '--patch-size', '8']
    subprocess.run(command + ['--plan-only'], check=True, capture_output=True)
    assert not (project / 'data').exists()
    subprocess.run(command, check=True, capture_output=True)
    output = project / 'data/processed/urbansarfloods'
    manifest = output / 'patch_manifest.csv'
    original = manifest.read_bytes()
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 36  # Six sources, six windows each; incomplete edges preserved.
    assert sum(int(r['valid_pixels']) for r in rows if r['split'] == 'test') == 2 * 13 * 17
    assert all(r['coherence_path'] == r['uncertain_mask_path'] == '' for r in rows)
    splits_by_event = {}
    for row in rows:
        splits_by_event.setdefault(row['event_id'], set()).add(row['split'])
        semantic = np.load(row['semantic_label_path'])
        target = np.load(row['label_path'])
        valid = np.load(row['valid_mask_path'])
        np.testing.assert_array_equal(target, (semantic == 1) | (semantic == 2))
        assert not target[valid == 0].any()
        assert np.all(np.load(row['pre_path'])[:, valid[0] == 0] == 0)
    assert all(len(s) == 1 for s in splits_by_event.values())
    assert splits_by_event['20231201_Jubba'] == {'test'}
    norm = json.loads((project / 'data/metadata/normalization_v1.json').read_text())
    state = json.loads((output / '_preparation_state.json').read_text())
    expected = []
    for source in state['sources']:
        if source['split'] == 'train':
            with rasterio.open(source['sar']) as ds:
                a = ds.read([5, 6, 7, 8])
                expected.extend([a[:2].reshape(2, -1), a[2:].reshape(2, -1)])
    expected = np.concatenate(expected, axis=1)
    np.testing.assert_allclose(norm['mean'], expected.mean(axis=1))
    np.testing.assert_allclose(norm['std'], expected.std(axis=1), rtol=1e-6)
    sample = ManifestDataset(manifest, 'train', load_array)[0]
    assert sample['pre'].shape == (2, 8, 8)
    subprocess.run(command, check=True, capture_output=True)
    assert manifest.read_bytes() == original
    rejected = subprocess.run(command + ['--seed', '17'], capture_output=True)
    assert rejected.returncode != 0
    assert manifest.read_bytes() == original


def test_invalid_labels_rejected(tmp_path):
    sar, gt = tmp_path / 'SAR.tif', tmp_path / 'GT.tif'
    write_pair(sar, gt, 0)
    with rasterio.open(gt, 'r+') as ds:
        label = ds.read(1)
        label[0, 0] = 9
        ds.write(label, 1)
    with rasterio.open(sar) as image, rasterio.open(gt) as label:
        with pytest.raises(ValueError, match='Unexpected ground-truth'):
            prepare.read_window(image, label, rasterio.windows.Window(0, 0, 8, 8), [5, 6, 7, 8])


def test_mismatched_grid_rejected(tmp_path):
    parent = tmp_path / 'extracted_v1/urban_sar_floods/01_NF'
    sar, gt = parent / 'SAR/event_ID_0_0_SAR.tif', parent / 'GT/event_ID_0_0_GT.tif'
    write_pair(sar, gt, 0)
    with rasterio.open(gt, 'r+') as ds:
        ds.transform = from_origin(11, 20, 0.01, 0.01)
    with pytest.raises(ValueError, match='grid mismatch'):
        prepare.inventory(tmp_path)
