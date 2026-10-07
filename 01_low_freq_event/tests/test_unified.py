from copy import deepcopy
import json

from PIL import Image
import pytest

from low_freq_event.config import Settings
from low_freq_event.context import context_prompt
from low_freq_event.detection import associate_frame, reconcile_tracklets, WindowPerception
from low_freq_event.pipeline import analyze
from low_freq_event.schema import EntityList, GroundingResult, LocalEntity, TemporalEvidence
from low_freq_event.temporal import events_from_evidence
from low_freq_event.video import Clip, Frame
from test_pipeline import FakeBackend, video


def detection(frame, index=0, *, kind='person', feature=(1., 0.), face=None, label='cup', bbox=None):
    return dict(detection_id=f'{frame}_{kind}_{index}', frame_key=str(frame), time_sec=float(frame),
        kind=kind, label='인물' if kind=='person' else label, detector_class='person' if kind=='person' else label,
        bbox=bbox or [100,100,800,900], crop=f'{frame}_{kind}_{index}.jpg', detection_confidence=.99,
        body_feature=list(feature), object_feature=list(feature), face_feature=list(face) if face is not None else None,
        face_quality=50 if face is not None else 0)


def test_legacy_configs_select_the_same_single_pipeline():
    assert Settings().perception == Settings(perception='qwen').perception == Settings(perception='person_reid').perception == 'unified'


def test_detection_gate_removes_low_score_shapes_without_using_identity_similarity():
    from low_freq_event.detection import filter_detections
    observations = [dict(detection(0), detection_confidence=.356, identity_similarity=.99),
        dict(detection(1,kind='object',label='bird'), detection_confidence=.658),
        dict(detection(2,kind='object',label='kite'), detection_confidence=.438),
        dict(detection(3), detection_confidence=.839),
        dict(detection(4,kind='object'), detection_confidence=.9)]
    accepted, rejected = filter_detections(observations, Settings())
    assert [o['detection_id'] for o in accepted] == ['3_person_0','4_object_0']
    assert [r['threshold'] for r in rejected] == [.6,.7,.7]
    assert observations[0]['identity_similarity']==.99  # A visual match is not existence evidence.


def test_detection_gate_has_class_specific_boundaries_and_rejects_missing_scores():
    from low_freq_event.detection import filter_detections
    observations=[dict(detection(i,kind=kind), detection_confidence=score)
        for i,(kind,score) in enumerate([('person',.6),('object',.6),('object',.7),('person',None),('object',float('nan'))])]
    accepted,rejected=filter_detections(observations,Settings())
    assert [o['detection_id'] for o in accepted]==['0_person_0','2_object_0']
    assert len(rejected)==3 and rejected[-1]['detection_confidence'] is None


def test_changed_analysis_defaults_use_a_separate_repeatable_output(video,tmp_path):
    from low_freq_event.cli import default_analysis_output
    from low_freq_event.config import PIPELINE_VERSION
    from low_freq_event.pipeline import code_fingerprint
    from low_freq_event.storage import atomic_json, file_fingerprint
    settings=Settings()
    outputs=tmp_path/'outputs'
    base=default_analysis_output(video,settings,outputs)
    identity=dict(sha256=file_fingerprint(video)['sha256'],settings=settings.model_dump(),
        pipeline_version=PIPELINE_VERSION,code_sha256=code_fingerprint())
    atomic_json(base/'manifest.json',{'identity':identity})
    assert default_analysis_output(video,settings,outputs)==base
    stricter=Settings(person_confidence=.8)
    revision=default_analysis_output(video,stricter,outputs)
    assert revision!=base
    assert default_analysis_output(video,stricter,outputs)==revision
    identity['code_sha256']='old_code'
    atomic_json(base/'manifest.json',{'identity':identity})
    assert default_analysis_output(video,settings,outputs) not in {base,revision}


@pytest.mark.parametrize('available, expected', [(True, 'cuda'), (False, 'cpu')])
def test_default_device_prefers_available_cuda(monkeypatch, available, expected):
    import torch
    from low_freq_event.perception import perception_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    assert Settings().perception_device == 'auto'
    assert perception_device('auto').type == expected
    assert perception_device('cpu').type == 'cpu'


def test_partial_gpu_activation_oom_releases_weights_and_retries_next_window(monkeypatch):
    import torch
    from low_freq_event.detection import ObjectEncoder
    encoder = ObjectEncoder.__new__(ObjectEncoder)
    encoder.settings = Settings()
    encoder.device = torch.device('cpu')
    encoder.preferred_device = torch.device('cuda')
    moves = []
    class Model:
        def __init__(self, name, fail=False): self.name, self.fail = name, fail
        def to(self, device):
            moves.append((self.name, str(device)))
            if self.fail and str(device) == 'cuda':
                self.fail = False
                raise torch.cuda.OutOfMemoryError('activation failure')
    encoder.detector = Model('yolo')
    encoder.body = Model('osnet', fail=True)
    encoder.object_model = Model('dino')
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    encoder.activate()
    assert encoder.device.type == 'cpu'
    assert moves == [('yolo','cuda'),('osnet','cuda'),('yolo','cpu'),('osnet','cpu'),('dino','cpu')]
    encoder.activate()
    assert encoder.device.type == 'cuda'
    assert moves[-3:] == [('yolo','cuda'),('osnet','cuda'),('dino','cuda')]


@pytest.mark.parametrize('policy', ['auto', 'cuda'])
def test_inference_oom_only_retries_on_cpu_in_auto_mode(tmp_path, policy):
    import torch
    calls = []
    class Encoder:
        device = torch.device('cuda')
        def extract(self, *args):
            calls.append(self.device.type)
            if self.device.type == 'cuda': raise torch.cuda.OutOfMemoryError('inference failure')
            return []
        def offload(self): self.device = torch.device('cpu')
    perception = WindowPerception(Settings(perception_device=policy), tmp_path)
    perception.encoder = Encoder()
    image = Image.new('RGB', (32,32))
    if policy == 'cuda':
        with pytest.raises(torch.cuda.OutOfMemoryError): perception.extract(image, 'frame', 0.)
        assert calls == ['cuda']
    else:
        assert perception.extract(image, 'frame', 0.) == []
        assert perception.extract(image, 'frame2', 1.) == []
        assert calls == ['cuda','cpu','cpu']


@pytest.mark.parametrize('visible_devices', [None, '0', '-1'])
def test_cpu_detection_preserves_cuda_visibility_for_later_qwen(tmp_path, monkeypatch, visible_devices):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import cv2
    import torch
    import ultralytics
    from ultralytics.utils.torch_utils import select_device
    from low_freq_event import perception
    from low_freq_event.detection import ObjectEncoder
    from low_freq_event.third_party.osnet import osnet_ain

    if visible_devices is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible_devices)
    import os
    original_environment = dict(os.environ)
    monkeypatch.setattr(perception, 'model_files', lambda: tmp_path)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {'classifier.weight': torch.empty(1, 1)})
    monkeypatch.setattr(osnet_ain, 'osnet_ain_x1_0', lambda **kw: MagicMock())
    monkeypatch.setattr(cv2.FaceDetectorYN, 'create', lambda *a: MagicMock())
    monkeypatch.setattr(cv2.FaceRecognizerSF, 'create', lambda *a: MagicMock())
    detector = MagicMock()
    def predict(image, **kwargs):
        # Use the actual Ultralytics selector: passing the old string 'cpu'
        # here changes CUDA_VISIBLE_DEVICES to -1 before Qwen ever starts.
        assert select_device(kwargs['device'], verbose=False) == torch.device('cpu')
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=torch.empty(0, 4), conf=torch.empty(0), cls=torch.empty(0)))]
    detector.predict.side_effect = predict
    monkeypatch.setattr(ultralytics, 'YOLO', lambda *a: detector)
    encoder = ObjectEncoder(Settings(perception_device='cpu'))
    assert encoder.extract(Image.new('RGB', (32, 32)), 'frame', 0., tmp_path) == []
    assert dict(os.environ) == original_environment


def test_people_follow_faces_and_body_not_position_or_labels():
    registry, visual = {}, {}
    first = associate_frame([detection(0,0,feature=(1.,0.),face=(1.,0.)), detection(0,1,feature=(0.,1.),face=(0.,1.))], registry, visual, Settings())
    later = associate_frame([detection(1,0,feature=(0.,1.),bbox=[0,0,400,900],face=(0.,1.)),
                             detection(1,1,feature=(1.,0.),bbox=[600,0,1000,900],face=(1.,0.))], registry, visual, Settings())
    assert [o['entity_id'] for o in later] == [first[1]['entity_id'],first[0]['entity_id']]
    hidden = associate_frame([detection(2,feature=(1.,0.))],registry,visual,Settings())
    assert hidden[0]['entity_id']==first[0]['entity_id'] and hidden[0]['identity_method']=='OSNet-AIN'


def test_conflicting_faces_and_clothing_fragments_do_not_force_a_match():
    registry, visual = {}, {}
    first = associate_frame([dict(detection(0,face=(1.,0.)),face_quality=100)],registry,visual,Settings())[0]
    other = associate_frame([dict(detection(1,face=(0.,1.)),face_quality=100)],registry,visual,Settings())[0]
    assert first['entity_id']!=other['entity_id']
    edge = associate_frame([detection(2,bbox=[0,0,80,900])],registry,visual,Settings())[0]
    assert edge['entity_id'] not in {first['entity_id'],other['entity_id']}
    assert edge['identity_uncertain']


def test_objects_use_visual_features_not_equal_labels_and_keep_covisibility():
    registry, visual = {}, {}
    first=associate_frame([detection(0,0,kind='object'),detection(0,1,kind='object',feature=(0.,1.))],registry,visual,Settings())
    later=associate_frame([detection(1,0,kind='object',feature=(0.,1.)),detection(1,1,kind='object')],registry,visual,Settings())
    assert [o['entity_id'] for o in later]==[first[1]['entity_id'],first[0]['entity_id']]
    assert all(o['identity_method']=='DINOv2-small' for o in later)
    reconcile_tracklets(registry,visual,Settings())
    assert len(visual['tracks'])==2
    before=deepcopy(visual)
    repeated=associate_frame([detection(1,0,kind='object',feature=(0.,1.)),detection(1,1,kind='object')],registry,visual,Settings())
    assert repeated==later and visual==before


def test_indistinguishable_covisible_objects_stay_unresolved():
    registry,visual={},{}
    associate_frame([detection(0,0,kind='object'),detection(0,1,kind='object')],registry,visual,Settings())
    later=associate_frame([detection(1,0,kind='object'),detection(1,1,kind='object')],registry,visual,Settings())
    assert later[0]['entity_id']!=later[1]['entity_id']
    assert all(o['identity_uncertain'] for o in later)
    reconcile_tracklets(registry,visual,Settings())
    assert len(visual['tracks'])==4


def test_window_reconciliation_joins_face_confirmed_fragments_and_rewrites_saved_ids():
    registry, visual = {}, {}
    first = associate_frame([detection(0, face=(1.,0.))], registry, visual, Settings())[0]
    # A rear view creates a provisional fragment. A later frontal sample has
    # matching body/face references in different groups and remains provisional.
    second = associate_frame([detection(1, feature=(0.,1.))], registry, visual, Settings())[0]
    assert first['entity_id'] != second['entity_id']
    associate_frame([detection(2, feature=(0.,1.), face=(.8,.6))], registry, visual,
                    Settings(identity_margin=.9))
    # The deliberately strict online margin leaves a third face-bearing
    # fragment. Reconciliation can use all non-co-visible observations.
    reconcile_tracklets(registry, visual, Settings())
    assert registry['person_0003']['alias_of'] == first['entity_id']
    assert visual['assignments']['2_person_0']['entity_id'] == first['entity_id']
    assert not visual['assignments']['2_person_0']['identity_uncertain']
    # The matching front/side sample now also bridges the identical rear-body fragment.
    assert registry[second['entity_id']]['alias_of'] == first['entity_id']


def test_detector_objects_are_prompted_before_qwen_adds_background(video,tmp_path):
    import torch
    output=tmp_path/'result'
    settings=Settings(window_sec=2,stride_sec=1,sample_fps=2,max_pixels=4096)
    perception=WindowPerception(settings,output)
    class Encoder:
        calls=0
        activations=0
        releases=0
        device=torch.device('cpu')
        def activate(self):
            self.activations+=1
            self.device=torch.device('cuda')
        def offload(self):
            self.releases+=1
            self.device=torch.device('cpu')
        def extract(self,image,key,t,base):
            assert self.device.type=='cuda'
            self.calls+=1
            items=[detection(t,kind='person',face=(1.,0.)),detection(t,kind='object')]
            items += [dict(detection(t,1,kind='person'),detection_confidence=.356),
                      dict(detection(t,1,kind='object',label='bird'),detection_confidence=.658)]
            (base/'crops').mkdir(exist_ok=True)
            for o in items:image.save(base/o['crop'])
            return items
    perception.encoder=Encoder()
    class Backend(FakeBackend):
        def discover(self,clip,*,known_entities,grounded):
            assert perception.encoder.device.type=='cpu', 'Release detector VRAM before Qwen'
            assert {e.local_id for e in known_entities}=={'person_0001','object_0001'}
            assert all(len(f['boxes'])==2 for f in grounded)
            return EntityList(entities=[LocalEntity(local_id='s1',label='설산',kind='environment',appearance='눈')])
        def ground_frame(self,frame,entities,frame_index=0,**kwargs):
            assert {e.local_id for e in kwargs['known_entities']}=={'person_0001','object_0001'}
            return GroundingResult(boxes=[{'local_id':'s1','bbox':[0,0,1000,1000]}])
        def scan(self,clip,entities,grounded):
            assert {e.local_id for e in entities}=={'person_0001','object_0001','entity_0003'}
            visible=[{b['local_id'] for b in f['boxes']} for f in grounded]
            facts=TemporalEvidence(frames=[dict(frame_index=i,states=[],transitions_from_previous=[]) for i in range(len(clip.frames))])
            return events_from_evidence(facts,entities,visible)
    data=analyze(video,output,settings,backend=Backend(),perception=perception)
    assert data['status']=='complete'
    assert perception.encoder.calls==6  # Eight window samples, six unique source frames.
    assert perception.encoder.activations==perception.encoder.releases==1
    assert all(len(json.loads(p.read_text())['rejected'])==2 for p in (output/'perception/filters').glob('*.json'))
    shared=[o for w in data['windows'] for o in w['observations'] if o['time_sec']==1. and o['kind']=='person']
    assert len(shared)==2 and shared[0]['entity_id']==shared[1]['entity_id']
    for w in data['windows']:
        folder=output/'windows'/w['window_id']
        context=json.loads((folder/'entity_context.json').read_text())
        assert {e['id'] for e in context['entities']}=={'person_0001','object_0001','entity_0003'}
        prompt=(folder/'entity_prompt.txt').read_text()
        assert 'person_0001' in prompt and 'object_0001' in prompt and '설산' in prompt
        for f in w['frames']:
            assert Image.open(output/f['image']).size==(96,64)  # Source pixels retained.


def test_qwen_discovery_gets_identity_evidence_and_cannot_reassign_people(tmp_path,monkeypatch):
    from low_freq_event.backend import QwenBackend, ModelOutputError
    entity=LocalEntity(local_id='person_0001',label='인물 1',kind='person',appearance='외형 비교')
    clip=Clip(0,1,24,[Frame(0,Image.new('RGB',(32,32)))])
    grounded=[{'boxes':[{'local_id':'person_0001','bbox':[0,0,500,1000], 'identity_method':'SFace', 'identity_similarity':.8}]}]
    prompts=[]
    backend=QwenBackend(Settings(json_retries=0),tmp_path/'calls')
    def generate(prompt,*args):
        prompts.append(prompt)
        return json.dumps({'entities':[{'local_id':'s1','label':'산','kind':'environment','appearance':'눈'}]})
    monkeypatch.setattr(backend,'_generate',generate)
    assert backend.discover(clip,known_entities=[entity],grounded=grounded).entities[0].kind=='environment'
    assert all(value in prompts[0] for value in ['person_0001','SFace','0.8','bbox'])
    monkeypatch.setattr(backend,'_generate',lambda *args:json.dumps({'entities':[entity.model_dump()]}))
    backend.cache_dir=tmp_path/'other_cache'
    with pytest.raises(ModelOutputError):backend.discover(clip,known_entities=[entity],grounded=grounded)


def test_per_video_viewer_chooses_latest_completed_matching_source(tmp_path):
    from low_freq_event.storage import atomic_json
    from low_freq_event.viewer import latest_completed_result
    for folder,source,date,status in [('old','sintel.mp4','2026-09-22','complete'),
                                      ('new','sintel.mp4','2026-09-23','complete'),
                                      ('other','pretty.mp4','2026-09-24','complete'),
                                      ('running','sintel.mp4','2026-09-25','running')]:
        atomic_json(tmp_path/folder/'timeline.json',dict(status=status,video={'source':{'path':'/videos/'+source}},
            timing={'sessions':[{'status':status,'ended_at':date+'T00:00:00+00:00'}]}))
    assert latest_completed_result(tmp_path,source_name='sintel.mp4')==tmp_path/'new'
    assert latest_completed_result(tmp_path,source_name='pretty.mp4')==tmp_path/'other'
    with pytest.raises(ValueError):latest_completed_result(tmp_path,source_name='missing.mp4')


def test_latest_viewer_includes_tracking_ready_and_ignores_review(tmp_path):
    from low_freq_event.storage import atomic_json
    from low_freq_event.viewer import latest_completed_result
    for folder, date, status in [('analysis', '2026-09-22', 'complete'),
                                 ('tracking', '2026-09-23', 'tracking_ready'),
                                 ('review', '2026-09-24', 'tracking_review'),
                                 ('failed', '2026-09-25', 'failed')]:
        atomic_json(tmp_path/folder/'timeline.json', dict(status=status,
            video={'source': {'path': '/videos/pretty.mp4'}},
            timing={'sessions': [{'status': status, 'ended_at': date+'T00:00:00+00:00'}]}))
    # Regenerating the older result must not override recorded completion times.
    (tmp_path/'analysis'/'timeline.json').touch()
    assert latest_completed_result(tmp_path, source_name='pretty.mp4') == tmp_path/'tracking'


def test_event_only_reanalysis_prompts_saved_final_ids_for_legacy_results():
    from low_freq_event.reanalyze import global_window_context
    old={'frames':[{'boxes':[{'local_id':'e1','bbox':[0,0,1000,1000]}]}],
         'observations':[{'frame_index':0,'local_id':'e1','entity_id':'person_0003'}],
         'local_entities':[{'local_id':'e1'}]}
    registry={'person_0003':{'label':'인물 3','kind':'person','appearance':'옷'}}
    converted=global_window_context(old,registry)
    assert converted['frames'][0]['boxes'][0]['local_id']=='person_0003'
    assert converted['local_entities'][0]['local_id']=='person_0003'
    assert converted['observations'][0]['source_local_id']=='e1'
    assert old['frames'][0]['boxes'][0]['local_id']=='e1'
