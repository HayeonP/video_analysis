from copy import deepcopy
import numpy as np

from low_freq_event.config import Settings
from low_freq_event.perception import face_evidence_quality
from low_freq_event.person_linking import reconcile_people, appearance_evidence
from test_unified import detection


def group(times, feature, index=0):
    return [dict(detection(t,index,feature=feature),shot_segment=index) for t in times]


def run_linking(tmp_path, groups, **kwargs):
    registry={};visual={'tracks':{},'assignments':{}}
    for eid,samples in groups.items():
        registry[eid]=dict(kind='person',first_seen_sec=min(s['time_sec'] for s in samples),
            last_seen_sec=max(s['time_sec'] for s in samples),gallery=[],possible_matches=[])
        visual['tracks'][eid]=dict(frame_keys=[s['frame_key'] for s in samples],samples=samples)
        for s in samples: visual['assignments'][s['detection_id']]=dict(entity_id=eid)
    report=reconcile_people(registry,visual,deepcopy(groups),Settings(**kwargs),tmp_path)
    return registry,visual,report


def test_returning_view_recovers_original_id_after_long_absence(tmp_path):
    _,v,r=run_linking(tmp_path,{'person_0001':group([0,.1,.2],(1,0)),
        'person_0002':group([8,8.1,8.2],(.99,.141),1)})
    assert r['final_person_ids']==1
    assert {a['entity_id'] for a in v['assignments'].values()}=={'person_0001'}


def test_identical_clothes_on_covisible_people_never_merge(tmp_path):
    _,_,r=run_linking(tmp_path,{'a':group([0,.1],(1,0)), 'b':group([0,.1],(1,0),1)})
    assert r['final_person_ids']==2


def test_ambiguous_return_is_not_forced_to_either_similar_person(tmp_path):
    _,_,r=run_linking(tmp_path,{'a':group([0,.1],(1,0)),
        'b':group([0,.1],(.999,.0447),1),'c':group([4,4.1],(1,0),2)})
    assert r['final_person_ids']==3 and not r['joins']


def test_one_accidental_match_cannot_merge_a_whole_tracklet():
    a=group([0,.1,.2],(1,0));b=group([3,3.1,3.2],(0,1),1)
    b[0]['body_feature']=[1,0]
    assert appearance_evidence(a,b,Settings())['score']<.86


def test_turn_with_close_geometry_can_bridge_lower_body_similarity(tmp_path,monkeypatch):
    monkeypatch.setattr('low_freq_event.person_linking.camera_motion',lambda *args:dict(valid=True,matrix=[[1,0,0],[0,1,0]]))
    a=group([0,.1],(1,0));b=group([.2,.3],(.7,np.sqrt(1-.7**2)),1)
    _,_,r=run_linking(tmp_path,{'a':a,'b':b})
    assert r['final_person_ids']==1
    assert r['joins'][0]['method']=='camera_motion_continuity'


def test_position_does_not_bridge_camera_cut_without_background_support(tmp_path,monkeypatch):
    monkeypatch.setattr('low_freq_event.person_linking.camera_motion',lambda *args:dict(valid=False))
    _,_,r=run_linking(tmp_path,{'a':group([0,.1],(1,0)),
        'b':group([.2,.3],(.7,np.sqrt(1-.7**2)),1)})
    assert r['final_person_ids']==2


def face(size=60):
    return np.array([20,20,size,size,20+size*.3,40,20+size*.7,40,20+size*.5,50,35,65,65,65,.99],np.float32)


def test_small_and_blurred_faces_are_rejected_before_recognition():
    settings=Settings();image=np.random.default_rng(0).integers(0,256,(120,120,3),dtype=np.uint8)
    assert face_evidence_quality(image,face(25),settings)['reason']=='too_small'
    assert face_evidence_quality(np.zeros_like(image),face(),settings)['reason']=='blurred'
    assert face_evidence_quality(image,face(),settings)['usable']
    profile=face();profile[8]=75
    assert not face_evidence_quality(image,profile,settings)['usable']


def test_tiny_face_vector_does_not_override_clothes():
    a=group([0,.1],(1,0));b=group([3,3.1],(0,1),1)
    for sample in a+b:
        sample.update(face_feature=[1,0],face_quality=100,face_evidence={'usable':False})
    assert appearance_evidence(a,b,Settings())['score']<.86


def test_crowded_recovery_requires_appearance_not_position(tmp_path,monkeypatch):
    monkeypatch.setattr('low_freq_event.person_linking.camera_motion',lambda *args:dict(valid=True,matrix=[[1,0,0],[0,1,0]]))
    a=group([0,.1],(1,0));b=group([.2,.3],(.7,np.sqrt(1-.7**2)),1)
    b[0]['identity_boundary']='crowded_occlusion_recovery'
    _,_,r=run_linking(tmp_path,{'a':a,'b':b})
    assert r['final_person_ids']==2


def test_clear_face_margin_uses_raw_cosine_not_compressed_score(tmp_path):
    a=group([0,.1],(1,0));b=group([0,.1],(0,1),1)
    c=group([4,4.1],(.75,np.sqrt(1-.75**2)),2)
    for samples,feature in ((a,[1,0]),(b,[0,1]),(c,[.75,np.sqrt(1-.75**2)])):
        for sample in samples:
            sample.update(face_feature=feature,face_quality=60,face_evidence={'usable':True})
    _,v,r=run_linking(tmp_path,{'a':a,'b':b,'c':c})
    assert r['final_person_ids']==2
    assert v['assignments'][c[0]['detection_id']]['entity_id']=='a'
    assert r['joins'][0]['face_competitor_margin']>=.08


def test_joint_assignment_resolves_coupled_but_unique_returnees():
    from low_freq_event.person_linking import joint_matching
    candidates=[dict(left='a',right='c',score=.928),dict(left='a',right='d',score=.916),
                dict(left='b',right='d',score=.931)]
    chosen=joint_matching(candidates,Settings())
    assert (chosen['left'],chosen['right']) in [('a','c'),('b','d')]
    assert chosen['competitor_margin']>=.035
    # Equal competing assignments must still remain unresolved.
    ambiguous=[dict(left='a',right='c',score=.93),dict(left='b',right='c',score=.93)]
    assert joint_matching(ambiguous,Settings()) is None
