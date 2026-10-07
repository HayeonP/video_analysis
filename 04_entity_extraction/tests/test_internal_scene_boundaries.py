from pathlib import Path
import sys
from unittest.mock import patch
from copy import deepcopy
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
from internal_scene_boundaries import cuts_from_signals,apply_internal_barriers


def signals(count=8):
    return [dict(frame=i,max_luma=255,content=2.,adaptive_ratio=1.,hist_distance=.03) for i in range(count)]


def test_requires_content_histogram_and_relative_change():
    rows=signals();rows[3].update(content=35,adaptive_ratio=4,hist_distance=.1)
    assert cuts_from_signals(rows)[0]==[]
    rows[3]['hist_distance']=.6
    assert cuts_from_signals(rows)[0]==[3]


def test_black_interval_is_a_barrier_without_dropping_source_frames():
    rows=signals()
    for i in [3,4]:rows[i]['max_luma']=0
    cuts,black=cuts_from_signals(rows)
    assert cuts==[3,5] and black==[[3,5]] and len(rows)==8


def test_only_unsupported_propagation_removed_and_candidates_split(tmp_path):
    from PIL import Image
    Image.new('RGB',(20,20),'red').save(tmp_path/'c.jpg')
    obs=[dict(frame=i,bbox=[0,0,10,10],confidence=.9,crop=str(tmp_path/'c.jpg'),image=str(tmp_path/'c.jpg'),
              evidence='sam2_mask',time_sec=i/2) for i in range(8)]
    track=dict(track_id='s:T1',source_track_ids=['s:T1'],observations=obs,representatives=obs)
    shot=dict(shot_id='s',source='unused.mp4',source_sha256='verified',frame_count=8,fps=2,tracks=[track])
    evidence=dict(fingerprint='masks',asset_root=str(tmp_path),shots=[shot])
    dense=deepcopy(evidence);dense['shots'][0]['tracks'][0]['observations']=[obs[1],obs[6]]
    rows=signals()
    for i in [3,4]:rows[i]['max_luma']=0
    with patch('internal_scene_boundaries.scan_video',return_value=rows):
        result=apply_internal_barriers(evidence,dense,tmp_path)
    children=result['shots'][0]['tracks']
    assert [[o['frame'] for o in t['observations']] for t in children]==[[0,1,2],[5,6,7]]
    assert result['internal_boundaries']['removed_sam_observations']==2
    assert result['shots'][0]['frame_count']==8
    assert len(evidence['shots'][0]['tracks'])==1
    assert all(t['temporal_candidate']['same_identity_allowed'] for t in children)


def test_detector_provenance_required(tmp_path):
    shot=dict(shot_id='s',source='unused',source_sha256='x',frame_count=8,tracks=[dict(track_id='s:T1',observations=[])])
    with patch('internal_scene_boundaries.scan_video',return_value=signals()):
        with pytest.raises(ValueError,match='provenance'):
            apply_internal_barriers(dict(fingerprint='x',shots=[shot]),dict(shots=[dict(shot_id='s',source_sha256='x',tracks=[])]),tmp_path)
