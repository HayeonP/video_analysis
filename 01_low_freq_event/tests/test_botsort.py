import numpy as np
from PIL import Image
from low_freq_event.botsort import DetectionBatch, new_tracker, tracking_frames, SceneCuts, BoTSORTPerception
from low_freq_event.config import Settings
from low_freq_event.pipeline import analyze
from low_freq_event.video import VideoSource, prepare_samples
from test_pipeline import video, FakeBackend
from test_unified import detection


def batch(rows):
    from ultralytics.engine.results import Boxes
    return DetectionBatch(Boxes(np.array(rows,dtype=np.float32).reshape(-1,6),(120,160)))


def test_real_botsort_uses_original_indices_through_high_low_stages():
    tracker=new_tracker(Settings(),10,'person')
    tracker.update(batch([[10,10,40,100,.3,0],[80,10,120,100,.9,0]]),
        feats=np.array([[0.,1.],[1.,0.]],np.float32))
    original=tracker.tracked_stracks[0]
    assert original.idx==1
    tid=original.track_id
    tracker.update(batch([[81,10,121,100,.3,0],[10,10,40,100,.9,0]]),
        feats=np.array([[1.,0.],[0.,1.]],np.float32))
    recovered=next(t for t in tracker.tracked_stracks if t.track_id==tid)
    assert recovered.idx==0 and recovered.is_activated
    assert len({t.track_id for t in tracker.tracked_stracks})==2


def test_real_botsort_recovers_after_short_occlusion():
    tracker=new_tracker(Settings(),10,'person')
    tracker.update(batch([[10,10,40,100,.9,0]]),feats=np.array([[1.,0.]],np.float32))
    tid=tracker.tracked_stracks[0].track_id
    tracker.update(batch([]),feats=np.empty((0,2),np.float32))
    assert tracker.lost_stracks
    tracker.update(batch([[12,10,42,100,.9,0]]),feats=np.array([[1.,0.]],np.float32))
    assert tracker.tracked_stracks[0].track_id==tid


def test_new_class_does_not_reset_existing_track_id_counter():
    first=new_tracker(Settings(),10,'person')
    first.update(batch([[10,10,40,100,.9,0]]),feats=np.array([[1.,0.]],np.float32))
    tid=first.tracked_stracks[0].track_id
    second=new_tracker(Settings(),10,'object')
    second.update(batch([[10,10,40,100,.9,0]]),feats=np.array([[1.,0.]],np.float32))
    assert second.tracked_stracks[0].track_id>tid


def test_dense_frames_include_every_vlm_sample(video,tmp_path):
    source=VideoSource(video,4096)
    samples=prepare_samples(source,tmp_path,1)
    frames=list(tracking_frames(source,3,samples))
    assert len(frames)==9
    assert {s['time_sec'] for s in samples}<={t for t,_ in frames}
    assert all(image.size==(96,64) for _,image in frames)


def test_tracking_only_does_not_construct_vlm_and_resumes(video,tmp_path,monkeypatch):
    calls=[]
    def extract(self,image,key,time):
        calls.append(time)
        item=detection(time)
        item.update(detection_id=key+'_person_000',frame_key=key,crop=f'crops/{key}.jpg',source_size=list(image.size))
        (self.output/'crops').mkdir(exist_ok=True)
        image.save(self.output/item['crop'])
        return [item]
    monkeypatch.setattr(BoTSORTPerception,'_extract',extract)
    def forbidden(*args): raise AssertionError('VLM must not be constructed')
    monkeypatch.setattr('low_freq_event.pipeline.QwenBackend',forbidden)
    output=tmp_path/'result'
    settings=Settings(window_sec=2,tracking_fps=3,max_pixels=4096)
    tracked=analyze(video,output,settings,tracking_only=True)
    assert tracked['status']=='tracking_ready' and not tracked['events']
    assert len(calls)==9 and len(tracked['video']['samples'])==3
    assert len([e for e in tracked['entities'] if e['kind']=='person'])==1
    resumed=analyze(video,output,settings,backend=FakeBackend())
    assert resumed['status']=='complete' and resumed['events']
    assert len(calls)==9


def test_scene_guard_resets_on_large_cut_not_static_image():
    guard=SceneCuts()
    first=np.zeros((60,100,3),np.uint8)
    second=np.full_like(first,255);second[:,:,0]=0
    assert not guard.update(first)[0]
    assert not guard.update(first)[0]
    assert guard.update(second)[0]


def test_clear_face_anchor_disambiguates_overlapping_people():
    from ultralytics.engine.results import Boxes
    tracker=new_tracker(Settings(),10,'person')
    tracker.update(batch([[10,10,40,100,.9,0],[20,10,50,100,.9,0]]),
        feats=np.array([[1.,0.],[1.,0.]],np.float32))
    left,right=tracker.tracked_stracks
    left_id,right_id=left.track_id,right.track_id
    tracker.person_face_anchors={left_id:[[1.,0.]],right_id:[[0.,1.]]}
    observations=[dict(face_feature=[0.,1.],face_evidence={'usable':True}),
                  dict(face_feature=[1.,0.],face_evidence={'usable':True})]
    results=DetectionBatch(Boxes(np.array([[10,10,40,100,.9,0],[20,10,50,100,.9,0]],np.float32),
        (120,160)),observations=observations)
    tracker.update(results,feats=np.array([[1.,0.],[1.,0.]],np.float32))
    assert next(t for t in tracker.tracked_stracks if t.track_id==left_id).idx==1
    assert next(t for t in tracker.tracked_stracks if t.track_id==right_id).idx==0
