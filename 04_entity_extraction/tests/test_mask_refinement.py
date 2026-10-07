from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mask_refinement as m


def fixture_bundle(tmp_path):
    source=tmp_path/'source.mp4'
    observations=[]
    for frame in (1,3):
        observations.append(dict(frame=frame,source_frame=frame,shot_id='shot_A',bbox=[2,2,8,8],
            confidence=.9,category='animal',evidence='observed_detection',feature=[1.,0.],
            image=None,crop=None,track_id='shot_A:T1'))
    shot=dict(shot_id='shot_A',source=str(source),fps=10.,original_fps=10.,frame_count=5,
        width=12,height=12,full_start_sec=4.,original_start_frame=40,
        tracks=[dict(track_id='shot_A:T1',observations=observations,representatives=[])],scene_frames=[])
    bundle=dict(fingerprint='fixture',asset_root=str(tmp_path),shots=[shot])
    local=dict(shots=[dict(shot_id='shot_A',entities=[dict(local_id='shot_A:L1',track_ids=['shot_A:T1'],kind='animal')],
                         rejected_tracks=[])])
    return bundle,local


class FakeMasks:
    def __init__(self,absent=(),drift=False):self.absent=set(absent);self.drift=drift;self.calls=[]
    def predict(self,frames,shot,specs,reverse=False):
        self.calls.append(reverse)
        for frame in (range(len(frames)-1,-1,-1) if reverse else range(len(frames))):
            predictions={}
            for lid in specs:
                logits=np.full((shot['height'],shot['width']),-10.,np.float32)
                if frame not in self.absent:
                    box=[2,2,8,8] if not (reverse and self.drift and frame==2) else [8,8,12,12]
                    logits[box[1]:box[3],box[0]:box[2]]=10.
                predictions[lid]=logits
            yield frame,predictions


def test_empty_mask_is_absence():
    assert m.mask_bbox(np.zeros((10,10),dtype=bool)) is None
    mask=np.zeros((10,10),dtype=bool);mask[3:7,2:5]=True
    assert m.mask_bbox(mask)==[2,3,5,7]


def test_prompts_use_observed_boxes_and_multiple_times():
    rows=[dict(frame=i,bbox=[0,0,10+i,10],confidence=.5) for i in range(9)]
    prompts=m.select_prompts(rows,4)
    assert len(prompts)==4 and len({x['frame'] for x in prompts})==4
    assert all(x in rows for x in prompts)


def test_mask_gates_reject_empty_unstable_and_drift():
    spec=dict(support=(0,4),reference_area=36,by_frame={},prompts=[],prompt_frames=set(),shape=(12,12))
    conf=m.MaskRefinementConfig()
    assert m.assess_mask(np.full((12,12),-10.),spec,1,conf)[1]=='empty_or_tiny_mask'
    assert m.assess_mask(np.full((12,12),.1),spec,1,conf)[1]=='unstable_mask'
    logits=np.full((12,12),-10.);logits[2:8,2:8]=10.
    assert m.assess_mask(logits,spec,6,conf)[1]=='outside_detector_supported_interval'
    assert m.assess_mask(np.zeros((4,4)),spec,2,conf)[1]=='invalid_mask'


def test_refinement_provenance_and_input_immutability(tmp_path,monkeypatch):
    bundle,local=fixture_bundle(tmp_path);original=deepcopy((bundle,local));backend=FakeMasks()
    monkeypatch.setattr(m,'_decode',lambda shot:[Image.new('RGB',(12,12)) for _ in range(5)])
    refined,review,summary=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',backend=backend)
    assert (bundle,local)==original
    track=refined['shots'][0]['tracks'][0]
    assert len(track['observations'])==5 and backend.calls==[False,True]
    assert summary['propagated_only_frames']==3
    assert review['shots'][0]['entities'][0]['track_ids']==[track['track_id']]
    assert track['source_track_ids']==['shot_A:T1']
    for row in track['observations']:
        assert row['evidence']=='sam2_mask' and row['full_time_sec']==4+row['frame']/10
        assert (tmp_path/row['mask']).is_file()
    assert (tmp_path/'mask_refinement/source_tracks.json').is_file()
    assert all((tmp_path/r['crop']).is_file() for r in track['representatives'])
    cached=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',backend=backend)
    assert backend.calls==[False,True] and cached[2]['fingerprint']==summary['fingerprint']


def test_absent_masks_do_not_fill_unobserved_frames(tmp_path,monkeypatch):
    bundle,local=fixture_bundle(tmp_path)
    monkeypatch.setattr(m,'_decode',lambda shot:[Image.new('RGB',(12,12)) for _ in range(5)])
    refined,_,summary=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',backend=FakeMasks(absent=range(5)))
    track=refined['shots'][0]['tracks'][0]
    assert [r['frame'] for r in track['observations']]==[1,3]
    assert all(r['evidence']=='observed_detection' for r in track['observations'])
    assert summary['mask_frames']==0 and summary['detector_fallback_frames']==2
    assert track['occlusion_gaps'][0]['start_frame']==2


def test_directional_disagreement_leaves_gap(tmp_path,monkeypatch):
    bundle,local=fixture_bundle(tmp_path)
    monkeypatch.setattr(m,'_decode',lambda shot:[Image.new('RGB',(12,12)) for _ in range(5)])
    refined,_,summary=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',backend=FakeMasks(drift=True))
    assert 2 not in {r['frame'] for r in refined['shots'][0]['tracks'][0]['observations']}
    assert summary['shots'][0]['directional_conflicts']==1


def test_nested_objects_keep_separate_physical_tracks(tmp_path,monkeypatch):
    bundle,local=fixture_bundle(tmp_path)
    child=deepcopy(bundle['shots'][0]['tracks'][0]);child['track_id']='shot_A:T2'
    for row in child['observations']:row['bbox']=[3,3,7,7];row['track_id']=child['track_id']
    bundle['shots'][0]['tracks'].append(child)
    local['shots'][0]['entities'].append(dict(local_id='shot_A:L2',track_ids=['shot_A:T2'],kind='animal'))
    monkeypatch.setattr(m,'_decode',lambda shot:[Image.new('RGB',(12,12)) for _ in range(5)])
    refined,review,_=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',backend=FakeMasks())
    assert len(refined['shots'][0]['tracks'])==2
    assert len({e['track_ids'][0] for e in review['shots'][0]['entities']})==2


def test_changed_mask_crop_does_not_reuse_detector_embedding(tmp_path):
    bundle,local=fixture_bundle(tmp_path);shot=bundle['shots'][0]
    spec=m._specs(shot,local['shots'][0],m.MaskRefinementConfig())['shot_A:L1']
    data=dict(bbox=[3,3,8,8],pixels=25,stability=1.)
    row=m._mask_row(data,shot,spec,1,.9,None)
    assert 'feature' not in row


def test_configuration_validation():
    with pytest.raises(ValueError):m.MaskRefinementConfig(prompts_per_entity=0)
    with pytest.raises(ValueError):m.MaskRefinementConfig(agreement_iou=2.)


def test_mask_collision_uses_pixels_not_nested_boxes():
    large=np.zeros((12,12),bool);large[1:11,1:11]=True
    small=np.zeros((12,12),bool);small[4:8,4:8]=True
    accepted={'large':{0:({'mask':large},1.)},'small':{0:({'mask':small},1.)}}
    assert m.reject_mask_collisions(accepted)==0
    accepted['small'][0]=({'mask':large.copy()},1.)
    assert m.reject_mask_collisions(accepted)==2
    assert not accepted['large'] and not accepted['small']


def test_shot_cache_resumes_without_repeating_model(tmp_path,monkeypatch):
    bundle,local=fixture_bundle(tmp_path);backend=FakeMasks()
    monkeypatch.setattr(m,'_decode',lambda shot:[Image.new('RGB',(12,12)) for _ in range(5)])
    first=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',backend=backend)
    (tmp_path/'mask_refinement/summary.json').unlink()
    second=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',backend=backend)
    assert backend.calls==[False,True]
    assert first[0]['shots']==second[0]['shots']
    assert first[1]==second[1]


def test_default_hub_checkpoint_is_pinned():
    identity=m.checkpoint_identity('facebook/sam2.1-hiera-small')
    assert identity['revision']=='ee5bba1d82bb8749febdf90f45e84b687142ba03'
    with pytest.raises(ValueError):m.checkpoint_identity('some-other/sam-model')
    with pytest.raises(ValueError):m.checkpoint_identity('some-other/sam-model','main')


def test_local_checkpoint_identity_changes_when_weights_change(tmp_path):
    weights=tmp_path/'model.safetensors';weights.write_bytes(b'one')
    first=m.checkpoint_identity(tmp_path)
    weights.write_bytes(b'two-larger')
    second=m.checkpoint_identity(tmp_path)
    assert first!=second and first['kind']=='local'


def test_only_sam_representatives_get_neutral_masked_crops(tmp_path,monkeypatch):
    bundle,local=fixture_bundle(tmp_path)
    monkeypatch.setattr(m,'_decode',lambda shot:[Image.new('RGB',(12,12),(200,40,20)) for _ in range(5)])
    class HoleMasks(FakeMasks):
        def predict(self,*args,**kwargs):
            for frame,rows in super().predict(*args,**kwargs):
                for logits in rows.values():logits[3:5,3:5]=-10.
                yield frame,rows
    refined,_,_=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',backend=HoleMasks(absent=[1]))
    track=refined['shots'][0]['tracks'][0]
    for row in track['representatives']:
        assert (tmp_path/row['crop']).is_file()
        if row['evidence']=='sam2_mask':
            masked=np.asarray(Image.open(tmp_path/row['masked_crop']))
            assert np.array_equal(masked[1,1],[127,127,127])
            assert np.array_equal(masked[0,0],[200,40,20])
        else:assert 'masked_crop' not in row


def test_raw_mask_evidence_preserves_raw_ids_and_overlapping_masks(tmp_path,monkeypatch):
    bundle,_=fixture_bundle(tmp_path)
    duplicate=deepcopy(bundle['shots'][0]['tracks'][0]);duplicate['track_id']='shot_A:T2'
    for row in duplicate['observations']:row['track_id']=duplicate['track_id']
    bundle['shots'][0]['tracks'].append(duplicate)
    monkeypatch.setattr(m,'_decode',lambda shot:[Image.new('RGB',(12,12)) for _ in range(5)])
    refined,summary=m.raw_track_masks(bundle,tmp_path,device='cpu',backend=FakeMasks())
    assert {t['track_id'] for t in refined['shots'][0]['tracks']}=={'shot_A:T1','shot_A:T2'}
    assert summary['identity_decisions'] is False
    assert summary['mask_frames']==10 and summary['shots'][0]['cross_identity_mask_collisions']==0
    for track in refined['shots'][0]['tracks']:
        assert all(row['track_id']==track['track_id'] for row in track['observations'])
        assert all((Path(refined['asset_root'])/row['mask']).is_file() for row in track['observations'])


def test_backend_uses_independent_sessions_for_later_born_objects():
    import torch
    from types import SimpleNamespace
    class Processor:
        def __init__(self):self.sessions=[]
        def init_video_session(self,**kwargs):
            session=SimpleNamespace(obj_ids=[],processed_frames={i:None for i in range(len(kwargs.get('video') or []))})
            self.sessions.append(session);return session
        def add_inputs_to_inference_session(self,session,frame_idx,obj_ids,input_boxes):
            assert obj_ids==1
            if obj_ids not in session.obj_ids:session.obj_ids.append(obj_ids)
        def post_process_masks(self,masks,**kwargs):return masks
    class Model:
        def __call__(self,inference_session,**kwargs):assert inference_session.obj_ids==[1]
        def propagate_in_video_iterator(self,session,start_frame_idx,reverse):
            for frame in range(2,-1,-1) if reverse else range(3):
                yield SimpleNamespace(frame_idx=frame,pred_masks=torch.ones((1,1,12,12)))
    backend=object.__new__(m.TransformersSam2Backend)
    backend.device='cpu';backend.dtype=torch.float32;backend.processor=Processor();backend.model=Model()
    specs={'early':{'prompts':[dict(frame=0,bbox=[1,1,4,4])]},
           'later':{'prompts':[dict(frame=2,bbox=[4,4,8,8])]}}
    frames=[Image.new('RGB',(12,12)) for _ in range(3)]
    for reverse in (False,True):
        outputs=list(backend.predict(frames,dict(width=12,height=12),specs,reverse))
        assert len(outputs)==6 and {key for _,row in outputs for key in row}=={'early','later'}
    assert len(backend.processor.sessions)==5
    assert len([session for session in backend.processor.sessions if session.obj_ids])==4


def fake_hub_snapshot(tmp_path,monkeypatch):
    import hashlib
    hub=tmp_path/'hub';repo=hub/'models--facebook--sam2.1-hiera-small'
    path=repo/'snapshots'/m.PINNED_REVISIONS['facebook/sam2.1-hiera-small']
    path.mkdir(parents=True);(repo/'blobs').mkdir()
    for name,data in [('config.json',b'{}'),('model.safetensors',b'weights-v1')]:
        address=(hashlib.sha256(data).hexdigest() if name.endswith('safetensors')
                 else hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest())
        blob=repo/'blobs'/address;blob.write_bytes(data)
        (path/name).symlink_to(blob)
    monkeypatch.setenv('HF_HUB_CACHE',str(hub))
    return path


def test_verified_snapshot_and_pinned_hub_share_content_identity(tmp_path,monkeypatch):
    path=fake_hub_snapshot(tmp_path,monkeypatch)
    local=m.checkpoint_identity(path);hub=m.checkpoint_identity('facebook/sam2.1-hiera-small')
    assert local==hub and len(local['verified_files'])==2
    assert local['kind']=='huggingface'


def test_corrupted_snapshot_cannot_reuse_verified_identity(tmp_path,monkeypatch):
    import os
    path=fake_hub_snapshot(tmp_path,monkeypatch)
    first=m.checkpoint_identity(path)
    weight=path/'model.safetensors';before=weight.stat()
    weight.write_bytes(b'weights-v2')
    os.utime(weight,ns=(before.st_atime_ns,before.st_mtime_ns))
    second=m.checkpoint_identity('facebook/sam2.1-hiera-small')
    assert first!=second and second['kind']=='local'
    assert 'verified_files' not in second
    payload=dict(model=second,version=m.VERSION,bundle='test')
    assert m._compatible_checkpoint_keys(payload,path,None)=={m._digest(payload)}


def test_verified_snapshot_reuses_previous_local_metadata_cache(tmp_path,monkeypatch):
    path=fake_hub_snapshot(tmp_path,monkeypatch)
    bundle,local=fixture_bundle(tmp_path);backend=FakeMasks()
    monkeypatch.setattr(m,'_decode',lambda shot:[Image.new('RGB',(12,12)) for _ in range(5)])
    identity=m.checkpoint_identity
    monkeypatch.setattr(m,'checkpoint_identity',lambda model_path,revision=None:m._legacy_local_identity(path,revision))
    original=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',model_path=path,backend=backend)
    monkeypatch.setattr(m,'checkpoint_identity',identity)
    reused=m.refine_local_tracks(bundle,local,tmp_path,device='cpu',backend=backend)
    assert backend.calls==[False,True]
    assert reused[2]['fingerprint']==original[2]['fingerprint']
