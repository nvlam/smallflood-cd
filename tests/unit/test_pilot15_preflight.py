import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'next_steps', Path(__file__).resolve().parents[2] / 'scripts/next_steps.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def prepared(tmp_path):
    metadata = {'pre_bands_1based': [5, 6], 'post_bands_1based': [7, 8],
                'fitted_split': 'train', 'shared_pre_post': True,
                'units': 'dB', 'already_applied_to_npy': True}
    path = tmp_path / 'normalization.json'
    path.write_text(json.dumps(metadata))
    return {'data': {'normalization_stats': str(path), 'channels': 2,
                     'include_coherence': False}}, metadata, path


def test_provenance_records_scope_without_modifying_data(prepared):
    config, _, path = prepared
    original = path.read_bytes()
    result = launcher.band_provenance(config)
    assert result['published_band_convention_verified'] is True
    assert 'not per-file audit' in result['verification_scope']
    assert len(result['normalization_sha256']) == 64
    assert path.read_bytes() == original


@pytest.mark.parametrize('key,value', [
    ('pre_bands_1based', [1, 2]), ('post_bands_1based', [5, 6]),
    ('fitted_split', 'test'), ('shared_pre_post', False),
    ('units', 'linear'), ('already_applied_to_npy', False)])
def test_reject_inconsistent_metadata(prepared, key, value):
    config, metadata, path = prepared
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='metadata mismatch'):
        launcher.band_provenance(config)


def test_reject_coherence(prepared):
    config, _, _ = prepared
    config['data']['include_coherence'] = True
    with pytest.raises(ValueError, match='no coherence'):
        launcher.band_provenance(config)
