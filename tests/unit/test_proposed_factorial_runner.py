import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest
import torch

ROOT=Path(__file__).parents[2]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    monkeypatch.setenv('PYTHONHASHSEED','42')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    spec=importlib.util.spec_from_file_location('factorial_test',ROOT/'scripts/proposed_factorial.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Toy(torch.utils.data.Dataset):
    def __init__(self, weighted): self.weighted=weighted
    def __len__(self): return 16
    def __getitem__(self,i):
        y=torch.zeros(1,32,32); y[:,8:12,8:12]=1
        result={'pre':torch.full((2,32,32),i/16), 'post':torch.ones(2,32,32),
                'target':y,'valid_mask':torch.ones_like(y),'patch_id':f'p{i}', 'event_id':'synthetic'}
        if self.weighted: result['component_weights']=1+y*2
        return result


def test_four_cpu_cells_checkpoint_and_pairing(runner,tmp_path):
    summaries=[]
    for cell,c in runner.configs().items():
        s=runner.run_cell(cell,c,Toy(c['loss']['use_size_aware']),Toy(False),
                          tmp_path/cell,torch.device('cpu'),1)
        assert s['optimizer_steps']==2 and s['checkpoint_reload_exact']
        assert not s['test_used']
        metrics=json.loads((tmp_path/cell/'metrics.jsonl').read_text())
        assert 0<=metrics['clipping_fraction']<=1
        assert metrics['scaled_loss_terms_mean']['boundary']==0 if cell in 'AB' else metrics['scaled_loss_terms_mean']['boundary']>0
        summaries.append(s)
    runner.check_pairing(summaries)
    summaries[-1]['batch_order_sha256']=['corrupted']
    with pytest.raises(RuntimeError,match='Batch order'):
        runner.check_pairing(summaries)


def test_no_weight_fallback_and_no_overwrite(runner,tmp_path):
    c=runner.configs()['B']
    with pytest.raises(RuntimeError,match='Missing size weights'):
        runner.run_cell('B',c,Toy(False),Toy(False),tmp_path/'missing',torch.device('cpu'),1)
    with pytest.raises(FileExistsError):
        runner.run_cell('B',c,Toy(True),Toy(False),tmp_path/'missing',torch.device('cpu'),1)


def test_datasets_only_request_train_validation(runner,monkeypatch):
    calls=[]
    class Fake:
        def __init__(self,split):
            self.records=[types.SimpleNamespace(patch_id=f'{split}{i}',event_id=split,
                           coherence_path=None,uncertain_mask_path=None)
                          for i in range(19166 if split=='train' else 4767)]
        def __len__(self): return len(self.records)
    def factory(config,split,loss):
        assert split in ['train','validation']
        calls.append(split); return Fake(split),35
    monkeypatch.setattr(runner,'_dataset',factory)
    runner.datasets(runner.configs()['A'],False)
    assert calls==['train','validation']


def test_gate_rejects_stale_and_incomplete(runner,tmp_path):
    gate={'stage':'preflight','fingerprint':{'version':1},'test_used':False,'summaries':[]}
    (tmp_path/'COMPLETE.json').write_text(json.dumps(gate))
    with pytest.raises(RuntimeError,match='stale'):
        runner.verify_gate(tmp_path,{'version':2})
    with pytest.raises(RuntimeError,match='Incomplete'):
        runner.verify_gate(tmp_path,{'version':1})


def test_run_requires_explicit_launch_approval(runner,monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'pin_source',lambda:None)
    monkeypatch.setattr(sys,'argv',['proposed_factorial.py','run','--output',str(tmp_path/'new')])
    with pytest.raises(SystemExit): runner.main()
    assert not (tmp_path/'new').exists()
