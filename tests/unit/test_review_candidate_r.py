import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).parents[2]


@pytest.fixture
def review(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('candidate_review_test', ROOT / 'scripts/review_candidate_r.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_hash_lock(review, tmp_path):
    p = tmp_path / 'summary.json'
    p.write_text('{}')
    anchor = {'files': {'summary.json': review.sha256(p)}}
    review.verify_evidence(tmp_path, anchor)
    p.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='evidence changed'):
        review.verify_evidence(tmp_path, anchor)


def test_checkpoint_hash_checked_before_unpickle(review, monkeypatch, tmp_path):
    p = tmp_path / 'last.ckpt'
    p.write_bytes(b'not a checkpoint')
    def forbidden(*a, **kw):
        pytest.fail('Must reject hash before torch.load')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(RuntimeError, match='hash mismatch'):
        review.load_verified_checkpoint(p, 'wrong', {}, {})


def test_checkpoint_epoch_config_score(review, tmp_path):
    p = tmp_path / 'last.ckpt'
    row = dict(epoch=14, global_step=35940, _expected_best_score=.8)
    cfg = {'model': {'name': 'smallflood_cdnet'}}
    ck = dict(epoch=14, global_step=35940, best_score=.8, config=cfg, model={})
    torch.save(ck, p)
    review.load_verified_checkpoint(p, review.sha256(p), row, cfg)
    for field, bad in [('epoch', 13), ('global_step', 1), ('best_score', .7), ('config', {})]:
        torch.save({**ck, field: bad}, p)
        with pytest.raises(RuntimeError):
            review.load_verified_checkpoint(p, review.sha256(p), row, cfg)


def test_prepare_four_checkpoint_jobs(review, monkeypatch, tmp_path):
    monkeypatch.setattr(review, 'verify_sources', lambda *a: None)
    monkeypatch.setattr(review, 'verify_metadata', lambda *a: None)
    provenance = {'files': {'manifest': 'fixture'}}
    summaries = []
    names = ['COMPLETE.json', 'provenance.json']
    review.save_json(tmp_path / 'provenance.json', provenance)
    for cell in review.CELLS:
        directory = tmp_path / cell
        (directory / 'checkpoints').mkdir(parents=True)
        config = {'protocol': {'cell': cell}, 'model': {'name': 'smallflood_cdnet', 'boundary_head': False},
                  'selection': {'probability_threshold': .5}}
        (directory / 'config_resolved.yaml').write_text(yaml.safe_dump(config))
        rows = [dict(epoch=i, global_step=(i+1)*2396, validation_selection_score=.5) for i in range(15)]
        (directory / 'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        hashes = {}
        for name, epoch in [('best_composite.ckpt', 0), ('last.ckpt', 14)]:
            p = directory / 'checkpoints' / name
            torch.save(dict(epoch=epoch, global_step=(epoch+1)*2396, config=config, best_score=.5), p)
            hashes[name] = review.sha256(p)
        summary = dict(cell=cell, status='COMPLETE', epochs=15, optimizer_steps=35940,
                       parameter_count=1079865, test_used=False, initial_sha256='paired',
                       batch_order_sha256=['paired']*15, checkpoint_sha256=hashes)
        review.save_json(directory / 'summary.json', summary)
        review.save_json(directory / 'environment.json', dict(validation_patches=4767,
                         test_used=False, manifest_sha256='fixture'))
        summaries.append(summary)
        names.extend(f'{cell}/{n}' for n in ('config_resolved.yaml', 'metrics.jsonl', 'summary.json', 'environment.json'))
    review.save_json(tmp_path / 'COMPLETE.json', dict(stage='run', test_used=False,
                    fingerprint=provenance, summaries=summaries))
    anchor = {'files': {n: review.sha256(tmp_path / n) for n in names}}
    jobs, prov = review.prepare_jobs(tmp_path, anchor)
    assert prov == provenance
    assert [(j[0].name, j[1], j[2]['epoch']) for j in jobs] == [
        ('A_control','best_composite.ckpt',0), ('A_control','last.ckpt',14),
        ('R','best_composite.ckpt',0), ('R','last.ckpt',14)]
    (tmp_path / 'R/checkpoints/last.ckpt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        review.prepare_jobs(tmp_path, anchor)


def test_frozen_metadata_and_sources(review, monkeypatch, tmp_path):
    paths = {}
    for name in ('manifest', 'component_stats', 'normalization_stats'):
        p = tmp_path / name
        p.write_text('{}')
        paths[name] = str(p)
    prov = {'files': {k: review.sha256(p) for k, p in paths.items()}}
    review.verify_metadata({'data': paths}, prov)
    Path(paths['manifest']).write_text('changed')
    with pytest.raises(RuntimeError, match='metadata changed'):
        review.verify_metadata({'data': paths}, prov)
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    for name in ('src/example.py', *review.REVIEW_DEPENDENCIES):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# fixture')
        prov['files'][name] = review.sha256(p)
    review.verify_sources(prov)
    (tmp_path / 'src/example.py').write_text('# changed')
    with pytest.raises(RuntimeError, match='source changed'):
        review.verify_sources(prov)


@pytest.fixture
def reports(review, tmp_path):
    from review_pilot_validation import scores
    global_scores = scores([10, 2, 3, 100])
    metrics = {k: global_scores[k] for k in ('precision', 'recall', 'f1', 'iou', 'accuracy')}
    expected = dict(epoch=11, **{f'validation_global_{k}': global_scores[k] for k in ('tp','fp','fn','tn')})
    for k, v in metrics.items():
        expected['validation_global_pixel_' + k] = v
        expected['validation_event_macro_' + k] = v
        expected['validation_event_macro_' + k + '_defined_count'] = 1
        expected[f'validation_event/event/{k}'] = v
    report = dict(split='validation', test_used=False, training_performed=False, threshold=.5,
                  validation_patches=4767, checkpoint_sha256='checkpoint', manifest_sha256='manifest',
                  checkpoint_epoch_1based=12, global_pixel=global_scores, event_macro=metrics,
                  event_macro_defined_counts={k: 1 for k in metrics})
    review.save_json(tmp_path / 'report.json', report)
    review.write_csv(tmp_path / 'events.csv', [{'event': 'event', **metrics}])
    denominators = dict(patches=4767, component_target=7, small_target_clipped=3,
                        small_target_interior=2, boundary_target=10, boundary_valid_pixels=100)
    detail = dict(split='validation', test_used=False, checkpoint_sha256='checkpoint',
                  protocol=review.PROTOCOL, small_threshold_pixels=35, counts=denominators,
                  component_stats_sha256='stats', module_sha256='module',
                  events=[{'event': 'event', **denominators}])
    review.save_json(tmp_path / 'object_boundary_report.json', detail)
    provenance = {'files': {'manifest': 'manifest', 'component_stats': 'stats',
                           'src/smallflood_cd/metrics/object_boundary_v2.py': 'module'}}
    return tmp_path, expected, provenance, report, detail


def test_full_report_matching(review, reports):
    path, row, prov, _, detail = reports
    result = review.verify_reports(path, row, 'checkpoint', prov)
    assert result['event'] == detail['counts']


@pytest.mark.parametrize('change', ['counts', 'macro', 'events', 'scope', 'protocol', 'threshold'])
def test_report_mismatch_rejected(review, reports, change):
    path, row, prov, report, detail = reports
    if change == 'counts': report['global_pixel']['tp'] += 1
    if change == 'macro': report['event_macro']['f1'] += .01
    if change == 'events': review.write_csv(path / 'events.csv', [{'event': 'wrong'}])
    if change == 'scope': report['test_used'] = True
    if change == 'protocol': detail['protocol'] = {'version': 'wrong'}
    if change == 'threshold': report['threshold'] = .4
    review.save_json(path / 'report.json', report)
    review.save_json(path / 'object_boundary_report.json', detail)
    with pytest.raises(RuntimeError):
        review.verify_reports(path, row, 'checkpoint', prov)


def test_no_nan_metric_or_undefined_coercion(review):
    review.assert_score(None, None, 'recall')
    for a, b in [(float('nan'), .5), (None, 0), (.5, .6)]:
        with pytest.raises(RuntimeError): review.assert_score(a, b, 'metric')


@pytest.fixture
def orchestrated(review, monkeypatch, tmp_path):
    pilot = tmp_path / 'pilot'
    jobs = []
    for cell in review.CELLS:
        directory = pilot / cell
        (directory / 'checkpoints').mkdir(parents=True)
        (directory / 'config_resolved.yaml').write_text('{}')
        for name in review.CHECKPOINTS:
            path = directory / 'checkpoints' / name
            path.write_bytes(b'fixture')
            jobs.append((directory, name, {'epoch': 1}, review.sha256(path)))
    anchor = tmp_path / 'anchor.json'
    anchor.write_text('{"archive_sha256":"fixture"}')
    monkeypatch.setattr(review, 'ANCHOR', anchor)
    monkeypatch.setattr(review, 'prepare_jobs', lambda *a: (jobs, {}))
    monkeypatch.setattr(review, 'verify_evidence', lambda *a: None)
    monkeypatch.setattr(review, 'verify_sources', lambda *a: None)
    monkeypatch.setattr(review, 'verify_metadata', lambda *a: None)
    monkeypatch.setattr(review, 'verify_loaded_modules', lambda: None)
    monkeypatch.setattr(review, 'verify_reports', lambda *a: {'event': {'small_target_interior': 2}})
    calls = []
    def evaluator(directory, name, output, args):
        assert args.device == 'cuda' and args.batch_size == 8 and args.object_boundary is True
        calls.append((directory.name, name))
        return {'run': directory.name, 'checkpoint': name}
    monkeypatch.setattr(review, 'evaluate', evaluator)
    def no_optimizer(*a, **kw): pytest.fail('Review must not construct an optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', no_optimizer)
    return pilot, tmp_path / 'out', calls, jobs


def test_exactly_four_evaluations_and_no_overwrite(review, orchestrated):
    pilot, output, calls, jobs = orchestrated
    review.run(pilot, output)
    assert calls == [(c, n) for c in review.CELLS for n in review.CHECKPOINTS]
    complete = review.read_json(output / 'COMPLETE.json')
    assert complete['evaluations'] == 4 and complete['test_used'] is False
    assert complete['training_performed'] is False
    assert len(review.read_json(output / 'checkpoint_verification.json')) == 4
    for directory, name, _, digest in jobs:
        assert review.sha256(directory / 'checkpoints' / name) == digest
    with pytest.raises(RuntimeError, match='fresh output'):
        review.run(pilot, output)
    with pytest.raises(RuntimeError, match='fresh output'):
        review.run(pilot, pilot / 'nested')


def test_denominator_drift_stops_without_complete(review, orchestrated, monkeypatch):
    pilot, output, _, _ = orchestrated
    values = iter([{'small': 2}, {'small': 3}])
    monkeypatch.setattr(review, 'verify_reports', lambda *a: next(values))
    with pytest.raises(RuntimeError, match='denominators changed'):
        review.run(pilot, output)
    assert not (output / 'COMPLETE.json').exists()


def test_cuda_required(review, monkeypatch, tmp_path):
    monkeypatch.setattr(review, 'pin_source', lambda: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['review_candidate_r.py', '--output', str(tmp_path / 'out')])
    with pytest.raises(RuntimeError, match='CUDA required'):
        review.main()
    assert not (tmp_path / 'out').exists()
