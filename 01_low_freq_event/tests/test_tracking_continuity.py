import math
import numpy as np
from low_freq_event.config import Settings
from low_freq_event.detection import associate_frame, reconcile_tracklets, select_references
from low_freq_event.motion import camera_motion
from test_unified import detection
from test_pipeline import video


def motion(time=0, scale=1, offset=0):
    return dict(valid=True, previous_time_sec=time, matrix=[[scale,0,offset],[0,scale,offset]], inlier_ratio=.9)


def test_turning_body_uses_valid_camera_continuity_but_not_coordinates_alone():
    for appearance, same in [(.75,True),(.2,False)]:
        registry,state={},{}
        first=associate_frame([detection(0,feature=(1,0))],registry,state,Settings())[0]
        later=associate_frame([detection(1,feature=(appearance,math.sqrt(1-appearance**2)))],registry,state,Settings(),motion=motion())[0]
        assert (first['entity_id']==later['entity_id']) == same
        if same: assert later['identity_method']=='appearance_camera_continuity'


def test_camera_scale_resolves_similar_clothes_without_covisible_merge():
    registry,state={},{}
    boxes=[[50,100,250,900],[700,100,900,900]]
    first=associate_frame([detection(0,i,bbox=b) for i,b in enumerate(boxes)],registry,state,Settings())
    transformed=[[v*.6+100 for v in b] for b in boxes]
    later=associate_frame([detection(1,i,bbox=b,feature=(.8,.6)) for i,b in enumerate(transformed)],registry,state,Settings(),motion=motion(scale=.6,offset=100))
    assert [o['entity_id'] for o in first]==[o['entity_id'] for o in later]
    reconcile_tracklets(registry,state,Settings())
    assert len(state['tracks'])==2


def test_cut_gap_and_good_conflicting_faces_block_motion_links():
    for t,m,face in [(1,dict(valid=False),None),(2,motion(),None),(1,motion(),(0,1))]:
        registry,state={},{}
        first=associate_frame([dict(detection(0,face=(1,0)),face_quality=100)],registry,state,Settings())[0]
        query=dict(detection(t,feature=(.75,math.sqrt(1-.75**2)),face=face),face_quality=100 if face else 0)
        later=associate_frame([query],registry,state,Settings(),motion=m)[0]
        assert first['entity_id']!=later['entity_id']


def test_small_face_does_not_veto_good_body_or_normal_border_person():
    registry,state={},{}
    first=associate_frame([dict(detection(0,face=(1,0)),face_quality=150)],registry,state,Settings())[0]
    later=associate_frame([dict(detection(1,face=(0,1),bbox=[0,100,700,900]),face_quality=30)],registry,state,Settings())[0]
    assert first['entity_id']==later['entity_id']
    sliver=associate_frame([detection(2,bbox=[0,0,40,900])],registry,state,Settings(),motion=motion(time=1))[0]
    assert sliver['entity_id']!=first['entity_id']


def test_body_only_reconciliation_is_reachable_without_weakening_covisibility():
    registry,state={},{}
    first=associate_frame([detection(0)],registry,state,Settings())[0]
    second=associate_frame([detection(1)],registry,state,Settings(identity_margin=.9))[0]
    assert first['entity_id']!=second['entity_id']
    reconcile_tracklets(registry,state,Settings())
    assert registry[second['entity_id']]['alias_of']==first['entity_id']


def test_reference_gallery_retains_rare_back_view():
    samples=[detection(i,feature=(0,1) if i==1 else (1,0)) for i in range(12)]
    chosen=select_references(samples)
    assert len(chosen)==8
    assert samples[1] in chosen and samples[-1] in chosen


def test_camera_estimation_recovers_zoom_and_rejects_blank(tmp_path):
    import cv2
    rng=np.random.default_rng(12)
    image=rng.integers(0,256,(540,960),dtype=np.uint8)
    image=cv2.GaussianBlur(image,(3,3),0)
    transform=np.array([[.85,0,60],[0,.85,30]],dtype=np.float32)
    target=cv2.warpAffine(image,transform,(960,540))
    a,b=tmp_path/'a.png',tmp_path/'b.png'
    cv2.imwrite(str(a),image);cv2.imwrite(str(b),target)
    result=camera_motion(a,b,[],[])
    assert result['valid']
    assert abs(result['matrix'][0][0]-.85)<.02
    cv2.imwrite(str(b),np.zeros_like(image))
    assert not camera_motion(a,b,[],[])['valid']


def test_crossing_appearance_conflict_is_not_overridden_by_position():
    registry,state={},{}
    first=associate_frame([detection(0,0,bbox=[50,100,250,900],feature=(1,0)),
        detection(0,1,bbox=[700,100,900,900],feature=(.75,math.sqrt(1-.75**2)))],registry,state,Settings())
    later=associate_frame([detection(1,0,bbox=[700,100,900,900],feature=(1,0)),
        detection(1,1,bbox=[50,100,250,900],feature=(.75,math.sqrt(1-.75**2)))],registry,state,Settings(),motion=motion())
    assert [o['entity_id'] for o in later]==[first[0]['entity_id'],first[1]['entity_id']]


def test_tracking_review_keeps_source_and_does_not_load_models(video,tmp_path,monkeypatch):
    from low_freq_event.pipeline import analyze
    from low_freq_event.tracking_review import review_tracking
    from low_freq_event.storage import atomic_json, read_json
    from test_pipeline import FakeBackend, EmptyPerception
    original=tmp_path/'original'
    data=analyze(video,original,Settings(window_sec=2,max_pixels=4096),backend=FakeBackend(),perception=EmptyPerception())
    before=(original/'checkpoint.json').read_bytes()
    for sample in data['video']['samples']:
        key=f"t{round(sample['time_sec']*1_000_000):012d}"
        atomic_json(original/'perception'/'frames'/f'{key}.json',dict(time_sec=sample['time_sec'],image=sample['image'],observations=[]))
    def no_models(*args):
        raise AssertionError('Review must not load an encoder')
    monkeypatch.setattr('low_freq_event.detection.ObjectEncoder',no_models)
    review_tracking(original,tmp_path/'review')
    result=read_json(tmp_path/'review'/'timeline.json')
    assert result['status']=='tracking_review' and result['events']==[]
    assert all(w['narrative'] is None for w in result['windows'])
    assert before==(original/'checkpoint.json').read_bytes()
