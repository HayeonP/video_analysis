from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'01_low_freq_event')]
from compact_identity import prepare_local,co_visibility,merge_allowed,canonical


def fixtures():
    tracks=[dict(track_id=k,observations=[dict(frame=0,bbox=b)]) for k,b in
            [('a',[0,0,10,10]),('b',[30,0,40,10]),('c',[1,1,9,9])]]
    bundle=dict(shots=[dict(shot_id='s',tracks=tracks)])
    local=dict(shots=[dict(shot_id='s',entities=[dict(local_id='s:L1',track_ids=['a','b','c'],kind='person',label='old',appearance='old')],events=[dict(subject='s:L1')])])
    return bundle,local


def test_local_split_keeps_nested_duplicate_with_owner_but_separates_other_body():
    bundle,local=fixtures();result,audit=prepare_local(bundle,local)
    groups=[set(e['track_ids']) for e in result['shots'][0]['entities']]
    assert {'a','c'} in groups and {'b'} in groups
    assert local['shots'][0]['entities'][0]['track_ids']==['a','b','c']
    assert len(co_visibility(bundle,result))==1 and len(audit)==1
    repeated,again=prepare_local(bundle,result)
    assert repeated==result and not again


def test_cluster_wide_veto_prevents_transitive_bridge():
    a=dict(local_ids=['left','bridge']);b=dict(local_ids=['right'])
    assert merge_allowed(a,b,{('left','right')},{})=='co_visible'
    assert merge_allowed(a,b,set(),{'left':dict(kind='person',hair='red'),
                                   'right':dict(kind='person',hair='bald')})=='incompatible_hair'


def test_unknown_does_not_contradict_and_label_does_not_use_description_length():
    profiles={'a':dict(kind='person',hair='red',facial_hair='no',presentation='female',label='observed',appearance='short'),
              'b':dict(kind='person',hair='unknown',facial_hair='unknown',presentation='unknown',label='bald man with goatee',appearance='wrong '*100)}
    a=dict(local_ids=['a']);b=dict(local_ids=['b'])
    assert merge_allowed(a,b,set(),profiles) is None
    value=canonical(dict(local_ids=['a','b'],kind='person'),profiles)
    assert value['label']=='붉은 머리 여성' and value['appearance']=='short'


def test_five_call_limit_counts_retries_and_cannot_be_reset_on_resume(tmp_path,monkeypatch):
    from unittest.mock import patch
    from PIL import Image
    import compact_identity as compact
    from story_semantics import StoryBackend,GlobalDecision
    from identity_review import ReviewBackend
    from low_freq_event.config import Settings
    entries=[dict(local_id=f's:L{i}',kind='person',label='old',appearance='',track_ids=[]) for i in range(54)]
    local=dict(shots=[dict(shot_id='s',entities=entries,relations=[])])
    bundle=dict(shots=[dict(shot_id='s',tracks=[])])
    provisional=GlobalDecision(entities=[dict(local_ids=[e['local_id']],kind='person',label='old',appearance='',reason='provisional') for e in entries])
    monkeypatch.setattr(compact,'visual_board',lambda b,e,g,keys,o:([('test',Image.new('RGB',(32,32)))],{k:[0] for k in keys},{}))
    backend=StoryBackend(Settings(json_retries=1),tmp_path,20)
    with patch.object(ReviewBackend,'_generate',return_value='{}') as generate:
        result,audit=compact.reconcile_compact(bundle,local,provisional,tmp_path,backend,max_calls=5)
        assert generate.call_count==5 and audit['generations']==5
        assert len(result.entities)==54 and len(audit['unreviewed_local_ids'])==54
        backend=StoryBackend(Settings(json_retries=1),tmp_path,20)
        compact.reconcile_compact(bundle,local,provisional,tmp_path,backend,max_calls=5)
        assert generate.call_count==5


def test_prepared_groups_export_without_duplicate_ids_or_lost_distinct_bodies():
    from story_export import suppress_duplicate_boxes
    bundle,local=fixtures();local,_=prepare_local(bundle,local)
    tracks={t['track_id']:t for t in bundle['shots'][0]['tracks']}
    observations=[]
    for entity in local['shots'][0]['entities']:
        for tid in entity['track_ids']:
            observations.append(dict(shot_id='s',source_frame=0,entity_id=entity['local_id'],
                detection_id=tid,tracklet_id=tid,bbox_pixels=tracks[tid]['observations'][0]['bbox'],confidence=.9))
    kept,audit=suppress_duplicate_boxes(observations)
    assert audit['conflict_count']==0
    assert len(kept)==2 and len({r['entity_id'] for r in kept})==2
    assert {tuple(r['bbox_pixels']) for r in kept}=={(0,0,10,10),(30,0,40,10)}
    assert audit['suppressed'][0]['removed_observation']['detection_id']=='c'


def test_omitted_group_stays_singleton_but_duplicate_or_invented_ids_fail():
    import pytest
    from compact_identity import CatalogReview,validate_catalog
    profile=dict(id='A',kind='unknown',label='unresolved',appearance='',evidence='Visible but ambiguous target',frame=0)
    result=validate_catalog(CatalogReview(profiles=[profile],groups=[]),['A'],{'A':[0]})
    assert result.groups[0].members==['A']
    for members in [['A','A'],['A','invented']]:
        with pytest.raises(ValueError):
            validate_catalog(CatalogReview(profiles=[profile],groups=[dict(members=members,reason='Proposed visual identity match')]),['A'],{'A':[0]})


def test_temporal_crossing_splits_even_with_identical_boxes_and_preserves_every_observation(tmp_path):
    from copy import deepcopy
    from compact_identity import prepare_temporal
    rows=[dict(frame=i,bbox=[0,0,10,10],crop=f'{i}.jpg',confidence=.9,feature=f)
          for i,f in enumerate([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])]
    bundle=dict(shots=[dict(shot_id='s',fps=24,tracks=[dict(track_id='t',observations=deepcopy(rows),representatives=rows)])])
    local=dict(shots=[dict(shot_id='s',entities=[dict(local_id='s:L1',track_ids=['t'],kind='person')],track_roles={'t':'whole_subject'},events=[])])
    result,entities,audit=prepare_temporal(bundle,local,tmp_path)
    assert audit[0]['cuts']==[2]
    assert [o['frame'] for t in result['shots'][0]['tracks'] for o in t['observations']]==[0,1,2,3]
    assert len(entities['shots'][0]['entities'])==2
    assert len(bundle['shots'][0]['tracks'])==1
    again,repeated,changes=prepare_temporal(result,entities,tmp_path)
    assert again==result and repeated==entities and changes==[]


def test_mixed_target_is_never_joined_to_a_confirmed_identity():
    a=dict(local_ids=['a']);b=dict(local_ids=['b'])
    profiles={'a':dict(kind='person',mixed=True), 'b':dict(kind='person')}
    assert merge_allowed(a,b,set(),profiles)=='incompatible_mixed_track'


def test_catalog_anchors_join_across_batches_in_five_calls_and_replay_free(tmp_path,monkeypatch):
    import json
    from unittest.mock import patch
    from PIL import Image
    import compact_identity as compact
    from story_semantics import StoryBackend,GlobalDecision
    from identity_review import ReviewBackend
    from low_freq_event.config import Settings
    entries=[dict(local_id=f's:L{i}',kind='person',label='old',appearance='',track_ids=[]) for i in range(54)]
    local=dict(shots=[dict(shot_id='s',entities=entries,relations=[])])
    bundle=dict(shots=[dict(shot_id='s',tracks=[])])
    provisional=GlobalDecision(entities=[dict(local_ids=[e['local_id']],kind='person',label='old',appearance='',reason='provisional') for e in entries])
    monkeypatch.setattr(compact,'visual_board',lambda b,e,g,keys,o:([('test',Image.new('RGB',(32,32)))],{k:[0] for k in keys},{}))
    def generate(prompt,*args,**kwargs):
        inventory=json.JSONDecoder().raw_decode(prompt.split('INPUT: ')[-1])[0];ids=inventory['ids']
        return json.dumps(dict(profiles=[dict(id=k,kind='person',label='person',appearance='distinctive face',hair='red',facial_hair='no',presentation='female',evidence='Same specific facial geometry',frame=0) for k in ids],groups=[dict(members=ids,reason='Same specific facial geometry across views')]))
    backend=StoryBackend(Settings(json_retries=1),tmp_path,20)
    with patch.object(ReviewBackend,'_generate',side_effect=generate) as calls:
        result,audit=compact.reconcile_compact(bundle,local,provisional,tmp_path,backend,max_calls=5)
        assert len(result.entities)==1 and len(result.entities[0].local_ids)==54
        assert audit['generations']==calls.call_count==5
        backend=StoryBackend(Settings(json_retries=1),tmp_path,20)
        replay,audit=compact.reconcile_compact(bundle,local,provisional,tmp_path,backend,max_calls=5)
        assert replay==result and calls.call_count==5 and audit['new_generations']==0


def test_invalid_frame_citation_is_unresolved_and_cannot_cause_a_join():
    from compact_identity import CatalogReview,validate_catalog
    result=validate_catalog(CatalogReview(profiles=[dict(id='A',kind='person',label='person',appearance='',evidence='Visible facial features',frame=999)],groups=[]),['A'],{'A':[0]})
    profile=result.profiles[0]
    assert profile.frame is None and not profile.verified and profile.kind=='unknown'
    assert result.groups[0].members==['A']


def test_existing_temporal_candidates_are_not_silently_reunified_before_review(tmp_path):
    from compact_identity import prepare_temporal
    tracks=[dict(track_id=k,observations=[dict(frame=i,bbox=[0,0,10,10])],representatives=[],temporal_candidate={'parent_track_id':'old'}) for i,k in enumerate(['a','b'])]
    bundle=dict(shots=[dict(shot_id='s',tracks=tracks)])
    local=dict(shots=[dict(shot_id='s',entities=[dict(local_id='s:L1',track_ids=['a','b'],kind='person')],events=[])])
    _,result,_=prepare_temporal(bundle,local,tmp_path)
    assert [e['track_ids'] for e in result['shots'][0]['entities']]==[['a'],['b']]


def test_history_failure_keeps_previous_checkpoint_and_escapes_model_text(tmp_path):
    import json
    from identity_history import write_history
    folder=tmp_path/'compact_identity';folder.mkdir()
    (folder/'audit.json').write_text('{}')
    (folder/'snapshot_000.json').write_text(json.dumps(dict(groups={'C1':dict(label='</script>',local_ids=['x'])})))
    (tmp_path/'qwen_budget.json').write_text(json.dumps(dict(events=[dict(phase='global_catalog:001',status='failed',elapsed_sec=1.2)])))
    write_history(tmp_path)
    page=(tmp_path/'IDENTITY_HISTORY.html').read_text()
    assert '\\u003c/script>' in page and '"accepted_checkpoint": false' in page


def test_thinking_budget_only_closes_unfinished_reasoning():
    import torch
    from low_freq_event.backend import ThinkingBudgetProcessor
    proc=ThinkingBudgetProcessor(prompt_length=2,budget=2,closing_token=3)
    before=torch.zeros((1,6))
    assert torch.equal(proc(torch.tensor([[9,9,1]]),before.clone()),before)
    forced=proc(torch.tensor([[9,9,1,2]]),before.clone())
    assert forced[0,3]==0 and torch.isneginf(forced[0,[0,1,2,4,5]]).all()
    after=proc(torch.tensor([[9,9,3,1,2]]),before.clone())
    assert torch.equal(after,before)


def test_optional_syntax_repair_does_not_repair_truncation_or_values():
    import pytest
    from low_freq_event.backend import parse_json_response
    text='{"profiles":[{"id":"A","mixed":true],"groups":[]}'
    with pytest.raises(ValueError):parse_json_response(text)
    repairs=[]
    assert parse_json_response(text,repair_json_closures=True,repair_log=repairs)=={'profiles':[{'id':'A','mixed':True}],'groups':[]}
    assert len(repairs)==1
    for bad in ['{"profiles":[{"id":"A"', '{"profiles":[{"id":],"groups":[]}', '<think>{"draft":1}']:
        with pytest.raises(ValueError):parse_json_response(bad,repair_json_closures=True)


def test_final_validation_detaches_temporal_switch_and_relabels_stable_member(tmp_path):
    import json
    from compact_identity import finalize_compact
    local=dict(shots=[dict(shot_id='s',entities=[dict(local_id='a',track_ids=['normal'],kind='person'),dict(local_id='b',track_ids=['switched'],kind='person')])])
    bundle=dict(shots=[dict(tracks=[dict(track_id='normal',observations=[]),dict(track_id='switched',observations=[],temporal_candidate={'parent_track_id':'old'})])])
    folder=tmp_path/'compact_identity';folder.mkdir()
    profiles={'a':dict(kind='person',hair='red',facial_hair='no',presentation='female',label='woman',appearance='red hair'), 'b':dict(kind='person',hair='bald',facial_hair='yes',presentation='male',label='bald man',appearance='long '*50)}
    (folder/'profiles.json').write_text(json.dumps(profiles))
    original=dict(entities=[dict(entity_id='G0001',local_ids=['a','b'],kind='person',label='bald man',appearance='wrong')],mapping={'a':'G0001','b':'G0001'})
    result=finalize_compact(original,bundle,local,tmp_path)
    assert result['mapping']['a']!=result['mapping']['b']
    assert result['entities'][0]['label']=='붉은 머리 여성'
    assert result['entities'][1]['uncertain'] and '미확정' in result['entities'][1]['label']
    assert original['mapping']['a']==original['mapping']['b']
    assert finalize_compact(result,bundle,local,tmp_path)==result


def test_missing_profile_stays_unverified_instead_of_discarding_valid_records():
    from compact_identity import CatalogReview,validate_catalog
    result=validate_catalog(CatalogReview(profiles=[dict(id='A',kind='person',label='person',appearance='',evidence='visible face',frame=0)],groups=[]),['A','B'],{'A':[0],'B':[1]})
    assert result.profiles[0].verified
    assert result.profiles[1].id=='B' and not result.profiles[1].verified
    assert result.profiles[1].kind=='unknown' and result.validation_issues


def test_same_source_face_requires_matching_source_coordinates_and_frame(tmp_path,monkeypatch):
    from PIL import Image
    import view_quality
    from compact_identity import same_source_face_pairs
    for name,width in [('a',100),('b',90)]:Image.new('RGB',(width,100)).save(tmp_path/f'{name}.png')
    observations=[dict(frame=3,bbox=[0,0,100,100],crop='a.png'),dict(frame=3,bbox=[10,0,100,100],crop='b.png')]
    bundle=dict(asset_root=str(tmp_path),shots=[dict(tracks=[dict(track_id=k,observations=[o]) for k,o in zip(['a','b'],observations)])])
    local=dict(shots=[dict(shot_id='s',entities=[dict(local_id=k,track_ids=[k],kind='person',target_profile={'target_verified':True}) for k in ['a','b']])])
    def detail(bundle,o,output):
        x=50 if o['crop']=='a.png' else 40
        return dict(confidence=.95,detection=[x,20,20,30,x+4,27,x+16,27,x+10,35,x+5,43,x+15,43,.95])
    monkeypatch.setattr(view_quality,'face_detail',detail)
    pairs=same_source_face_pairs(bundle,local,tmp_path)
    assert len(pairs)==1 and pairs[0]['landmark_distance']==0
    observations[1]['frame']=4
    assert not same_source_face_pairs(bundle,local,tmp_path)
    observations[1]['frame']=3;observations[1]['bbox']=[40,0,130,100]
    assert not same_source_face_pairs(bundle,local,tmp_path)
