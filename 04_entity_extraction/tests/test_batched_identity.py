"""The fast path must preserve its actual-call cap and isolated result channels."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/"01_low_freq_event")]
from types import SimpleNamespace
import pytest
from batched_identity import LimitedBackend
from story_semantics import BudgetExhausted

class FakeBackend:
    def __init__(self):
        self.limit=30;self.reserve=0;self.phase='batch_global:001'
        self.ledger={'events':[{'phase':'local:s'} for _ in range(5)]}
    @property
    def remaining(self):return self.limit-len(self.ledger['events'])
    def request(self,*args,**kwargs):
        if self.remaining<=self.reserve:raise BudgetExhausted('cap')
        self.ledger['events'].append({'phase':self.phase})
        return 'ok'

def test_actual_generation_cap_survives_new_wrapper_and_other_stages():
    backend=FakeBackend();limited=LimitedBackend(backend,2)
    assert limited.request('a',None)=='ok'
    limited.phase='batch_frames:001';limited.request('b',None)
    resumed=LimitedBackend(backend,2)
    with pytest.raises(BudgetExhausted):resumed.request('c',None)
    assert resumed.used()==2 and backend.reserve==0

def test_other_reservations_remain_effective():
    backend=FakeBackend();backend.reserve=backend.remaining
    with pytest.raises(BudgetExhausted):LimitedBackend(backend,10).request('a',None)
    assert backend.reserve==backend.remaining

def test_baseline_and_new_never_fall_back_to_latest(tmp_path,monkeypatch):
    import video_shortcuts as s
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'VIDEOS',{'sintel':tmp_path/'input.mp4'})
    monkeypatch.setattr(s,'completed_result',lambda folder,video: {'status':'complete'})
    (tmp_path/'outputs').mkdir()
    (tmp_path/'outputs/baseline_sintel.json').write_text(json.dumps({'benchmark_output':str(tmp_path/'old')}))
    (tmp_path/'outputs/latest_sampled_sintel.json').write_text(json.dumps({'benchmark_output':str(tmp_path/'wrong')}))
    assert s.comparison_result('sintel','baseline')==tmp_path/'old/entities'
    with pytest.raises(ValueError,match='아직 없습니다'):s.comparison_result('sintel','new')
    (tmp_path/'outputs/new_sintel.json').write_text(json.dumps({'benchmark_output':str(tmp_path/'new')}))
    assert s.comparison_result('sintel','new')==tmp_path/'new/entities'
    assert s.comparison_result('sintel','baseline')==tmp_path/'old/entities'

def test_new_benchmark_uses_distinct_strategy_with_small_global_budget(tmp_path):
    from benchmark_video import commands
    command=commands(tmp_path/'input.mp4',tmp_path/'out',80,10,'batched')[-1][1]
    assert command[command.index('--global-strategy')+1]=='batched'
    assert command[command.index('--global-review-max-calls')+1]=='10'


def test_frame_batch_has_one_flat_frame_namespace():
    from batched_identity import BatchFrames
    value=BatchFrames.model_validate({'F001':{'assignments':{'D1':'G0001'},'evidence':{'D1':'visible torso'}}})
    assert value.frames['F001'].bodies(1,['G0001']).bodies[0].identity=='G0001'
    with pytest.raises(ValueError):BatchFrames.model_validate({'frames':value.model_dump()})


def test_label_consensus_changes_name_not_membership(tmp_path):
    from batched_identity import label_consensus
    from copy import deepcopy
    shots=[];local=[];profiles={}
    for i,hair in enumerate(['갈색','red','red']):
        sid=f's{i}';tid=f't{i}';lid=f'l{i}';o={'frame':0,'bbox':[0,0,10,10]}
        shots.append({'shot_id':sid,'tracks':[{'track_id':tid,'observations':[o]}]})
        local.append({'shot_id':sid,'entities':[{'local_id':lid,'track_ids':[tid]}]})
        profiles[lid]={'frame':0,'kind':'person','hair':hair,'facial_hair':'no','presentation':'female','target_verified':True,'evidence':'visible','label':'woman','appearance':'person'}
    bundle={'shots':shots};local={'shots':local}
    catalog={'entities':[{'entity_id':'G0001','kind':'person','label':'여성','local_ids':['l0','l1','l2']}],'mapping':{f'l{i}':'G0001' for i in range(3)}}
    (tmp_path/'compact_identity').mkdir()
    for name,value in [('reviewed_tracking.json',bundle),('local_entities.json',local),('compact_identity/profiles.json',profiles)]:
        (tmp_path/name).write_text(json.dumps(value))
    before=deepcopy(catalog['mapping']);result=label_consensus(bundle,local,catalog,tmp_path)
    assert result['entities'][0]['label']=='붉은 머리 여성'
    assert result['mapping']==before and result['entities'][0]['local_ids']==['l0','l1','l2']
