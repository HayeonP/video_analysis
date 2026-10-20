import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'01_low_freq_event')]
from low_freq_event.config import Settings
from batched_identity import LimitedBackend
from context_reassignment import reassign,ContextAssignment


def test_clean_anchors_hide_old_ids_and_reassign_only_low_detail(tmp_path,monkeypatch):
    import compact_identity,story_semantics
    from PIL import Image
    def card():return Image.new('RGB',(1800,360))
    monkeypatch.setattr(compact_identity,'visual_board',lambda *a:([('card',card())],{},{}))
    monkeypatch.setattr(story_semantics,'scene_sheet',lambda *a:card())
    monkeypatch.setattr(story_semantics,'cannot_links',lambda *a:[])
    bundle={'shots':[dict(shot_id=f's{i}',width=100,height=100,full_start_sec=i,frame_count=24,fps=24,scene_frames=[{'frame':0}],tracks=[dict(track_id=f't{i}',observations=[dict(bbox=[0,0,100 if i==0 else 10,100 if i==0 else 10],frame=0)])]) for i in range(2)]}
    local={'shots':[dict(shot_id=f's{i}',summary='A person walks.',entities=[dict(local_id=f'l{i}:RG000{i+1}',kind='person',label='인물',appearance='푸른 옷',track_ids=[f't{i}'])]) for i in range(2)]}
    catalog={'entities':[dict(entity_id=f'G000{i+1}',kind='person',label='인물',appearance='푸른 옷',local_ids=[f'l{i}:RG000{i+1}']) for i in range(2)],'mapping':{f'l{i}:RG000{i+1}':f'G000{i+1}' for i in range(2)},'physical_groups':[{'entity_id':'G0002','members':[{'track_id':'t1'}]}]}
    class Backend:
        settings=Settings();model=None;processor=None;reserve=0;phase='';ledger={'events':[]};remaining=3
        def request(self,prompt,schema,**kwargs):
            inventory=json.loads(prompt.split('INPUT: ')[1])
            assert list(inventory['targets'])==['U002']
            assert 'previous_identity' not in inventory['targets']['U002']
            assert ':RG' not in json.dumps(inventory)
            assert len(inventory['summarize_shots'])==2
            r=ContextAssignment(assignments={'U0002':'G0001'},reasons={'U002':'같은 옷과 동작'},shots=[dict(shot_id=f's{i}',summary='푸른 옷의 인물이 걷는다.',transition='unknown',transition_evidence='불명확하다.') for i in range(2)],labels={'G0001':'푸른 옷 인물'})
            return kwargs['validate'](r)
    result=reassign(bundle,local,catalog,tmp_path,LimitedBackend(Backend(),3),dict(model_id=Settings().model_id,model_revision=Settings().model_revision))
    assert set(result['mapping'].values())=={'G0001'}
    assert result['physical_groups'][0]['entity_id']=='G0001'
    assert len(result['context_reassignments'])==1
    assert result['shots'][0]['summary']=='푸른 옷의 인물이 걷는다.'


def test_sparse_animal_reassignment_moves_only_verified_target(tmp_path,monkeypatch):
    from context_reassignment import animal_partition,AnimalPairs
    import compact_identity,story_semantics
    from PIL import Image
    monkeypatch.setattr(compact_identity,'visual_board',lambda *a:([('card',Image.new('RGB',(1800,360)))],{},{}))
    monkeypatch.setattr(story_semantics,'cannot_links',lambda *a:[])
    entries={f'l{i}':dict(shot_id='s'+str(max(0,i-2)),track_ids=[f't{i}']) for i in range(1,5)}
    units={f'U00{i}':dict(local_ids=[f'l{i}'],kind='animal') for i in range(1,5)}
    mapping={'l1':'G0001','l2':'G0001','l3':'G0002','l4':'G0002'}
    entities={gid:dict(kind='animal',label='생물',local_ids=[lid for lid in entries if mapping[lid]==gid]) for gid in set(mapping.values())}
    Image.new('RGB',(100,100)).save(tmp_path/'crop.jpg')
    features={1:[1,0,0],2:[0,1,0],3:[1,0,0],4:[0,0,1]}
    bundle={'asset_root':str(tmp_path),'shots':[dict(shot_id='s',tracks=[dict(track_id=f't{i}',representatives=[dict(feature=features[i])],observations=[dict(frame=0,feature=features[i],crop='crop.jpg')]) for i in range(1,5)])]}
    class Backend:
        def request(self,prompt,schema,**kwargs):
            data=json.loads(prompt.split('INPUT: ')[1]);assert len(data['pairs'])==1
            return kwargs['validate'](AnimalPairs(compatible={'P001':True},reasons={'P001':'같은 작은 생물'}))
    decisions=[]
    animal_partition(bundle,{},entities,mapping,units,entries,{},tmp_path,Backend(),[],{},decisions)
    assert mapping['l1']==mapping['l3'] and mapping['l1']!=mapping['l2']
    assert len(decisions)==1
    import pytest
    mapping['l1']='G0001'
    with pytest.raises(ValueError,match='split a verified body'):
        animal_partition(bundle,{},entities,mapping,units,entries,{'physical_groups':[{'members':[{'track_id':'t1'},{'track_id':'t2'}]}]},tmp_path,Backend(),[],{},[])
    assert mapping['l1']=='G0001'


def test_video_without_person_or_animal_still_gets_korean_memory(tmp_path,monkeypatch):
    from PIL import Image
    import story_semantics
    monkeypatch.setattr(story_semantics,'scene_sheet',lambda *a:Image.new('RGB',(100,100)))
    bundle={'shots':[dict(shot_id='s',full_start_sec=0,frame_count=24,fps=24,tracks=[],scene_frames=[{'frame':0}])]}
    local={'shots':[dict(shot_id='s',summary='Clouds move.',entities=[])]}
    catalog={'entities':[],'mapping':{}}
    class Backend:
        settings=Settings();model=None;processor=None;reserve=0;phase='';ledger={'events':[]};remaining=2
        def request(self,prompt,schema,**kwargs):
            data=json.loads(prompt.split('INPUT: ')[1]);assert data['summarize_shots']==['s']
            return kwargs['validate'](ContextAssignment(assignments={},reasons={},shots=[dict(shot_id='s',summary='구름이 보인다.',transition='unknown',transition_evidence='첫 장면이다.')]))
    result=reassign(bundle,local,catalog,tmp_path,LimitedBackend(Backend(),2),{})
    assert result['shots'][0]['summary']=='구름이 보인다.'
