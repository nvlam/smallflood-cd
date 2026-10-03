"""Review eight trusted factorial checkpoints on validation only; never train."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import yaml
from bit_sar_v2_entry import pin_source
from review_pilot_validation import evaluate, sha256, write_csv


def expected_epochs(rows):
    if len(rows)!=15 or [r['epoch'] for r in rows]!=list(range(15)):
        raise RuntimeError('Expected all 15 ordered epochs')
    if [r['global_step'] for r in rows]!=[2396*i for i in range(1,16)]:
        raise RuntimeError('Unexpected optimizer steps')
    return {'best_composite.ckpt':max(rows,key=lambda r:r['validation_selection_score']),
            'last.ckpt':rows[-1]}


def verify_checkpoint(checkpoint, expected, config):
    if checkpoint['epoch']!=expected['epoch'] or checkpoint['global_step']!=expected['global_step']:
        raise RuntimeError('Checkpoint epoch/step differs from log')
    if checkpoint['config']!=config:
        raise RuntimeError('Checkpoint config differs from resolved config')


def verify_counts(report, expected):
    if report['checkpoint_epoch_1based']!=expected['epoch']+1:
        raise RuntimeError('Reviewed epoch mismatch')
    for k in ('tp','fp','fn','tn'):
        if report['global_pixel'][k]!=expected[f'validation_global_{k}']:
            raise RuntimeError(f'Validation count mismatch: {k}; do not change threshold')


def run(pilot, output):
    complete=json.loads((pilot/'COMPLETE.json').read_text())
    provenance=json.loads((pilot/'provenance.json').read_text())
    if complete.get('stage')!='run' or complete.get('test_used') is not False or complete['fingerprint']!=provenance:
        raise RuntimeError('Incomplete factorial or provenance mismatch')
    summaries=complete['summaries']
    if [s['cell'] for s in summaries]!=list('ABCD'):
        raise RuntimeError('Expected A B C D completion records')
    for s in summaries:
        if s['status']!='COMPLETE' or s['epochs']!=15 or s['optimizer_steps']!=35940:
            raise RuntimeError('Incomplete cell')
        if s['common_initial_sha256']!=summaries[0]['common_initial_sha256'] or s['batch_order_sha256']!=summaries[0]['batch_order_sha256']:
            raise RuntimeError('Pairing mismatch')
    # Review code can be new; model/loss/data source used by training must not change.
    for name,digest in provenance['files'].items():
        if name.startswith('src/') and sha256(ROOT/name)!=digest:
            raise RuntimeError(f'Training source changed: {name}')
    jobs=[]
    for cell in 'ABCD':
        directory=pilot/f'{cell}_proposed'
        cfg=yaml.safe_load((directory/'config_resolved.yaml').read_text())
        if cfg['model']['name']!='smallflood_cdnet' or cfg['protocol']['cell']!=cell:
            raise RuntimeError('Wrong cell/model')
        for key in ('manifest','component_stats','normalization_stats'):
            if sha256(cfg['data'][key])!=provenance['files'][key]:
                raise RuntimeError(f'Data metadata changed: {key}')
        rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
        expected=expected_epochs(rows)
        for name,row in expected.items():
            path=directory/'checkpoints'/name
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            verify_checkpoint(checkpoint,row,cfg)
            del checkpoint
            jobs.append((directory,name,row,sha256(path)))
    output.mkdir(parents=True,exist_ok=False)
    args=SimpleNamespace(device='cuda',batch_size=8,object_boundary=True)
    results=[]
    print(f'OUTPUT: {output}',flush=True)
    for directory,name,expected,digest in jobs:
        path=directory/'checkpoints'/name
        if sha256(path)!=digest:
            raise RuntimeError('Checkpoint changed before review')
        result=evaluate(directory,name,output,args)
        report=json.loads((output/directory.name/Path(name).stem/'report.json').read_text())
        verify_counts(report,expected)
        if sha256(path)!=digest or report['checkpoint_sha256']!=digest:
            raise RuntimeError('Checkpoint changed during review')
        results.append(result)
        write_csv(output/'summary.csv',results)
        print(f'CHECKPOINT VERIFIED: {directory.name}/{name}',flush=True)
    (output/'COMPLETE.json').write_text(json.dumps({'evaluations':8,'test_used':False,
        'training_performed':False,'pixel_counts_match_training':True,
        'pilot_root':str(pilot),'script_sha256':sha256(__file__)},indent=2))
    print(f'FACTORIAL VALIDATION REVIEW COMPLETE: {output}',flush=True)


def main():
    pin_source()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--pilot-root',type=Path,default=Path('runs/proposed_factorial_v1/20260927T075837Z'))
    p.add_argument('--output',type=Path,default=Path('artifacts/proposed_factorial_review')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    a=p.parse_args()
    if a.self_test:
        import pytest
        raise SystemExit(pytest.main(['--import-mode=importlib','-q',
            'tests/unit/test_review_proposed_factorial.py','tests/unit/test_review_pilot_validation.py']))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for full validation review')
    run(a.pilot_root,a.output)


if __name__=='__main__':
    main()
