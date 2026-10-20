from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
from boundary_identity import boundary_candidates


def fixture(tmp_path):
    array=np.zeros((100,160,3),dtype=np.uint8);array[:,:80]=[200,120,80];array[:,80:]=[40,160,230]
    image=tmp_path/'source.jpg';crop=tmp_path/'crop.jpg'
    Image.fromarray(array).save(image);Image.fromarray(array[10:80,20:70]).save(crop)
    shots=[];entries={}
    for number,frame,time in [(0,9,.45),(1,0,.5)]:
        sid=f'arbitrary_{number}';tid=f'{sid}:track';lid=f'{sid}:local'
        row=dict(frame=frame,bbox=[20,10,70,80],confidence=.9,evidence='observed_detection',
            image=str(image),crop=str(crop),full_time_sec=time)
        shots.append(dict(shot_id=sid,width=160,height=100,frame_count=10,fps=20,tracks=[dict(track_id=tid,observations=[row])]))
        entries[lid]=dict(shot_id=sid,track_ids=[tid],kind='person')
    return {'shots':shots},entries


def test_actual_adjacent_endpoints_propose_match_without_assignment(tmp_path):
    bundle,entries=fixture(tmp_path);rows=boundary_candidates(bundle,entries,tmp_path)
    assert len(rows)==1
    assert rows[0]['left_frame']==9 and rows[0]['right_frame']==0
    assert rows[0]['bbox_iou']==rows[0]['scene_similarity']==rows[0]['crop_similarity']==1
    assert rows[0]['status']=='proposal_requires_visual_identity_review'
    assert all('entity_id' not in entry for entry in entries.values())


def test_missing_assets_unobserved_rows_and_source_gaps_do_not_propose(tmp_path):
    bundle,entries=fixture(tmp_path);row=bundle['shots'][1]['tracks'][0]['observations'][0]
    row['full_time_sec']=2
    assert boundary_candidates(bundle,entries,tmp_path)==[]
    row['full_time_sec']=.5;row['evidence']='interpolated'
    assert boundary_candidates(bundle,entries,tmp_path)==[]
    row['evidence']='observed_detection';row['crop']=None
    assert boundary_candidates(bundle,entries,tmp_path)==[]


def test_different_scene_or_body_part_does_not_propose(tmp_path):
    bundle,entries=fixture(tmp_path);row=bundle['shots'][1]['tracks'][0]['observations'][0]
    opposite=tmp_path/'different.jpg';Image.new('RGB',(160,100),'blue').save(opposite)
    row['image']=str(opposite)
    assert boundary_candidates(bundle,entries,tmp_path)==[]
    bundle,entries=fixture(tmp_path);next(iter(entries.values()))['_part_or_worn']=True
    assert boundary_candidates(bundle,entries,tmp_path)==[]


def test_strong_observed_unconfirmed_detection_is_opt_in_only(tmp_path):
    bundle,entries=fixture(tmp_path)
    row=bundle['shots'][0]['tracks'][0]['observations'][0];row['tracker_confirmed']=False
    assert boundary_candidates(bundle,entries,tmp_path)==[]
    assert boundary_candidates(bundle,entries,tmp_path,allow_unconfirmed=True)
    row['confidence']=.59
    assert boundary_candidates(bundle,entries,tmp_path,allow_unconfirmed=True)==[]
