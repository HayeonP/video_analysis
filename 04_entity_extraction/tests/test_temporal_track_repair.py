from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
from temporal_track_repair import appearance_boundaries,split_temporal_tracks


def observations():
    return [dict(frame=i,bbox=[0,0,20,20] if i<5 else [25,0,45,20],confidence=.9,
                 evidence='observed_detection',feature=[1.,0.] if i<5 else [0.,1.]) for i in range(10)]


def test_stable_change_is_provisional_and_noisy_is_not_forced():
    rows=observations();features={r['frame']:r['feature'] for r in rows}
    flags=appearance_boundaries(rows,features)
    assert [r['before_frame'] for r in flags]==[5]
    assert flags[0]['status']=='provisional_boundary_for_visual_review'
    assert appearance_boundaries(rows,{i:[1.,0.] for i in range(10)})==[]


def test_discontinuous_endpoint_requires_both_extreme_signals():
    rows=observations()[:3]
    features={0:[1,0],1:[1,0],2:[0,1]}
    assert appearance_boundaries(rows,features)==[]
    rows[-1]['bbox']=[25,0,45,20]
    assert [r['before_frame'] for r in appearance_boundaries(rows,features)]==[2]


def test_rapid_switches_without_stable_windows_remain_reviewable():
    frames=[0,1,20,21,23]
    features={f:([1.,0.,0.] if f<2 else [0.,1.,0.] if f<23 else [0.,0.,1.]) for f in frames}
    rows=[dict(frame=f,bbox=[0,0,20,20] if f<23 else [30,0,50,20]) for f in frames]
    flags=appearance_boundaries(rows,features,fps=20)
    assert [r['before_frame'] for r in flags]==[20,23]
    assert flags[0]['reason']=='unobserved_gap_with_changed_appearance'
    assert flags[1]['reason']=='joint_appearance_and_position_discontinuity'
    # A time gap by itself cannot split an unchanged appearance.
    assert appearance_boundaries(rows,{f:[1.,0.,0.] for f in frames},fps=20)==[]


def test_raw_segmentation_preserves_coverage_has_no_fake_masks_and_is_idempotent(tmp_path):
    from PIL import Image
    image=tmp_path/'source.jpg';Image.new('RGB',(50,50),'gray').save(image)
    rows=observations()
    for r in rows:r.update(crop=str(image),image=str(image),track_id='s:T1')
    bundle=dict(fingerprint='raw',asset_root=str(tmp_path),shots=[dict(shot_id='s',source='unused',
        source_sha256='source',width=50,height=50,fps=2,frame_count=10,
        tracks=[dict(track_id='s:T1',observations=rows,representatives=deepcopy(rows))])])
    signals=[dict(frame=i,max_luma=255,content=1,adaptive_ratio=1,hist_distance=0) for i in range(10)]
    with patch('internal_scene_boundaries.scan_video',return_value=signals):
        repaired=split_temporal_tracks(bundle,tmp_path)
    tracks=repaired['shots'][0]['tracks']
    assert len(tracks)==2
    assert sorted(r['frame'] for t in tracks for r in t['observations'])==list(range(10))
    assert all(t['temporal_candidate']['same_identity_allowed'] for t in tracks)
    assert not repaired.get('mask_track_graph') and 'mask_graph' not in repaired['shots'][0]
    assert len(bundle['shots'][0]['tracks'])==1
    assert split_temporal_tracks(repaired,tmp_path)==repaired


def test_source_cut_applies_without_appearance_or_sam(tmp_path):
    from PIL import Image
    image=tmp_path/'source.jpg';Image.new('RGB',(50,50),'gray').save(image)
    rows=observations()
    for r in rows:r.update(crop=str(image),image=str(image),track_id='s:T1',feature=[1,0])
    bundle=dict(fingerprint='raw',shots=[dict(shot_id='s',source='unused',source_sha256='source',
        width=50,height=50,fps=2,frame_count=10,tracks=[dict(track_id='s:T1',observations=rows,representatives=deepcopy(rows))])])
    signals=[dict(frame=i,max_luma=255,content=1,adaptive_ratio=1,hist_distance=0) for i in range(10)]
    signals[5].update(content=30,adaptive_ratio=4,hist_distance=.8)
    with patch('internal_scene_boundaries.scan_video',return_value=signals):repaired=split_temporal_tracks(bundle,tmp_path)
    assert len(repaired['shots'][0]['tracks'])==2
    assert repaired['raw_temporal_repair']['source_barrier_shots']==['s']


def test_gradual_drift_proposes_change_without_adjacent_discontinuity():
    import math
    from temporal_track_repair import drift_boundary
    rows=[dict(frame=i,bbox=[i*3,0,i*3+20,20],confidence=.9) for i in range(36)]
    features={i:[math.cos(i/35*math.pi/2),math.sin(i/35*math.pi/2)] for i in range(36)}
    assert appearance_boundaries(rows,features)==[]
    change=drift_boundary(rows,features)
    assert 12<=change['before_frame']<=24
    assert change['reason']=='long_track_appearance_and_location_drift'
    assert change['status']=='provisional_boundary_for_visual_review'
    assert change['transition_end_frame']==change['before_frame']+6
    assert drift_boundary(rows,features,[change]) is None
    assert drift_boundary(rows,{i:[1,0] for i in range(36)}) is None
    for row in rows:row['bbox']=[0,0,20,20]
    assert drift_boundary(rows,features) is None
