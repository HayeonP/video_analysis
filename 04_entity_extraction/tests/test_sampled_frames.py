"""Off-sample pixels must never reach inference, review, or selected export."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sampled_frames import discover_sampled_shots, iter_shot_images, SparseTracker, validate_sampled_bundle
from shot_tracking import DenseTrackingConfig, build_dense_tracks


@pytest.fixture
def adaptive(tmp_path):
    root=tmp_path/'input';shot=root/'s';shot.mkdir(parents=True)
    frames=[0,5,19]
    for n in frames:Image.new('RGB',(64,48),(n*10,100,140)).save(shot/f'{n}.png')
    meta=dict(source='/unavailable/DO_NOT_OPEN.mp4',source_frames=20,source_fps='10',original_fps='10',
              original_start_frame=100,samples=[dict(source_frame=f,source_time_sec=f/10,
              hold_until_frame=frames[i+1] if i+1<len(frames) else 20,image=f'{f}.png') for i,f in enumerate(frames)])
    (shot/'samples.json').write_text(json.dumps(meta))
    (root/'summary.json').write_text(json.dumps(dict(videos=[dict(id='s',samples='s/samples.json')])) )
    return root


def fake_predict(detector,image,config,device,**kwargs):
    return [dict(bbox=[10,8,36,40],confidence=.95,category='person',class_id=0,source='yolo_full_frame')]


def build_fixture(adaptive,tmp_path):
    weights=tmp_path/'fake.pt';weights.write_bytes(b'test model identity')
    with patch('strong_detector._weights_path',return_value=weights),patch('ultralytics.YOLO'),patch(
            'shot_tracking._predict',side_effect=fake_predict),patch('av.open',side_effect=AssertionError('original decoder forbidden')):
        return build_dense_tracks(adaptive,tmp_path/'result',device='cpu',analysis_frames='sampled',
                                  config=DenseTrackingConfig(appearance=False,tile_enabled=False))


def test_detector_assets_and_internal_review_only_use_selected_images(adaptive,tmp_path):
    from temporal_track_repair import split_temporal_tracks
    from story_semantics import ensure_representatives
    from story_relationship_images import _source_frames
    bundle=build_fixture(adaptive,tmp_path);shot=bundle['shots'][0]
    assert bundle['stats']['detector_inferred_frames']==3
    assert bundle['stats']['analyzed_frames']==3
    assert 'dense_frames' not in bundle['stats']
    assert {r['frame'] for t in shot['tracks'] for r in t['observations']}=={0,5,19}
    assert {r['frame'] for r in shot['scene_frames']}<={0,5,19}
    with patch('av.open',side_effect=AssertionError('original decoder forbidden')):
        repaired=split_temporal_tracks(bundle,tmp_path/'result')
        track=repaired['shots'][0]['tracks'][0];track['representatives']=[]
        ensure_representatives(repaired,repaired['shots'][0],track,tmp_path/'result')
        images=_source_frames(bundle,shot,tmp_path/'result',[{'frame':5}])
        assert list(images)==[5]
        validate_sampled_bundle(repaired,adaptive)
    assert not (tmp_path/'result/input_media').exists()


def test_export_only_selected_frames_with_exact_source_timing(adaptive,tmp_path):
    from story_export import _timeline_and_samples, _render_videos
    import av
    bundle=build_fixture(adaptive,tmp_path)
    timeline,samples=_timeline_and_samples(bundle,adaptive)
    output=tmp_path/'result'
    result=_render_videos(bundle,timeline,samples,[],{},output,original=True)
    assert not (output/'original_entities_full.mp4').exists()
    assert result['sampled_frames']==3 and result['source_frames']==20
    with av.open(str(output/'adaptive_entities_full.mp4')) as c:
        stamps=[float(f.pts*f.time_base) for f in c.decode(video=0)]
    assert stamps==[0.,.5,1.9]
    assert result['duration_sec']==2.


def test_missing_or_unselected_image_never_falls_back_to_original(adaptive):
    shot=discover_sampled_shots(adaptive)[0]
    with pytest.raises(ValueError,match='Unselected'):list(iter_shot_images(shot,{1}))
    (adaptive/'s/5.png').unlink()
    with pytest.raises(ValueError,match='Missing'):discover_sampled_shots(adaptive)


def test_sample_changes_invalidate_resume(adaptive,tmp_path):
    bundle=build_fixture(adaptive,tmp_path)
    Image.new('RGB',(64,48),'red').save(adaptive/'s/5.png')
    with pytest.raises(ValueError,match='changed'):validate_sampled_bundle(bundle,adaptive)


def test_elapsed_time_not_sample_index_controls_track_expiry():
    cfg=DenseTrackingConfig(track_buffer_sec=1.,appearance=False)
    tracker=SparseTracker(10,cfg)
    rows=fake_predict(None,None,None,None)
    first=tracker.update(rows,0)[0].track_id
    assert tracker.update(rows,9)[0].track_id==first
    assert tracker.update(rows,25)[0].track_id!=first


def test_assignment_is_one_to_one_and_weak_tiles_cannot_start():
    tracker=SparseTracker(10,DenseTrackingConfig())
    a=fake_predict(None,None,None,None)[0]
    b=dict(a,bbox=[40,8,62,40])
    initial=tracker.update([a,b],0)
    following=tracker.update([b,a],5)
    assert [r.track_id for r in following]==[initial[1].track_id,initial[0].track_id]
    weak=dict(a,source='yolo_small_target_tile',confidence=.3)
    assert SparseTracker(10,DenseTrackingConfig()).update([weak],0)==[]


def test_source_time_fields_cannot_inject_an_unselected_timestamp(adaptive):
    p=adaptive/'s/samples.json';meta=json.loads(p.read_text())
    meta['samples'][1]['original_frame']=999
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='original frame'):discover_sampled_shots(adaptive)


def test_sampled_report_does_not_claim_dense_inference(tmp_path):
    from story_report import write_report
    (tmp_path/'run_config.json').write_text(json.dumps(dict(analysis_frames='sampled',mask_refinement=False,
        settings=dict(model_id='local'),global_model_id='global')))
    (tmp_path/'tracking_summary.json').write_text(json.dumps(dict(stats=dict(analyzed_frames=3),shots=[dict(shot_id='s')])))
    write_report(tmp_path)
    text=(tmp_path/'REPORT.html').read_text()
    assert '선택된 분석 프레임' in text
    assert '선택 JPEG만 사용' in text
    assert '추적에는 원본 연속 프레임을 사용합니다' not in text
    assert '원본 연속 추적 영상' not in text
