from copy import deepcopy
import numpy as np

from low_freq_event.perception import associate_people, cluster_people


def observation(frame, person, *, face=True, bbox=None):
    body = np.zeros(512); body[person] = 1
    feature = np.zeros(128); feature[person] = 1
    return {'detection_id': f'{frame}_{person}', 'frame_key': str(frame), 'time_sec': float(frame),
            'bbox': bbox or [100, 100, 800, 900], 'crop': f'{frame}_{person}.jpg',
            'body_feature': body.tolist(), 'face_feature': feature.tolist() if face else None,
            'face_quality': 50 if face else 0}


def anchors():
    return [{'observations': [observation(i, p) for p in range(2)]} for i in range(6)]


def test_reid_uses_appearance_when_people_exchange_positions_and_hide_faces():
    frames = anchors()
    frames.append({'observations': [observation(6, 1, face=False, bbox=[10, 10, 400, 900]),
                                    observation(6, 0, face=False, bbox=[600, 10, 990, 900])]})
    tracks = associate_people(frames)
    assert len(tracks) == 2
    first = {o['detection_id'].split('_')[-1]: o['entity_id'] for o in frames[0]['observations']}
    last = {o['detection_id'].split('_')[-1]: o['entity_id'] for o in frames[-1]['observations']}
    assert first == last
    assert all(o['identity_method'] == 'body' for o in frames[-1]['observations'])


def test_edge_fragment_without_face_is_not_forced_onto_known_person():
    frames = anchors()
    frames.append({'observations': [observation(6, 0, face=False, bbox=[0, 100, 80, 900])]})
    tracks = associate_people(frames)
    fragment = frames[-1]['observations'][0]
    assert fragment['identity_uncertain']
    assert fragment['identity_method'] == 'unresolved'
    assert tracks[fragment['entity_id']]['identity_status'] == 'uncertain'


def test_similar_covisible_people_never_merge_even_with_identical_features():
    frames = anchors()
    for f in frames:
        f['observations'][1]['body_feature'] = deepcopy(f['observations'][0]['body_feature'])
        f['observations'][1]['face_feature'] = deepcopy(f['observations'][0]['face_feature'])
    tracks, links = cluster_people(frames)
    assert len(tracks) == 2
    for f in frames:
        assert len({o['entity_id'] for o in f['observations']}) == 2


def test_empty_person_detection_is_supported():
    assert associate_people([{'observations': []}]) == {}


def test_conflicting_continuation_flags_do_not_expand_a_short_event(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from low_freq_event.backend import QwenBackend
    from low_freq_event.config import Settings
    from low_freq_event.schema import LocalEntity
    from low_freq_event.video import Clip, Frame
    entity = LocalEntity(local_id='p', label='사람', kind='person', appearance='흰 옷')
    answer = {'entities':[entity.model_dump()], 'events':[
        {'event_type':'ongoing','action':'visible','description':'사람이 보임','entity_ids':['p'],
         'start_frame':1,'end_frame':2,'continues_before':True,'continues_after':True,
         'uncertain':False,'boundary_note':''}]}
    backend = QwenBackend(Settings(json_retries=0), tmp_path/'calls')
    monkeypatch.setattr(backend, '_generate', lambda *args: json.dumps(answer))
    clip = Clip(0,4,1,[Frame(i,Image.new('RGB',(32,32))) for i in range(4)])
    result = backend.scan_intervals(clip,[entity],[{'boxes':[{'local_id':'p'}]} for _ in range(4)])
    event = result.events[0]
    assert (event.start_frame,event.end_frame)==(1,2)
    assert not event.continues_before and not event.continues_after
    assert event.uncertain and '충돌' in event.boundary_note
    raw = json.loads(next((tmp_path/'calls').glob('*.json')).read_text())
    assert json.loads(raw['response'])['events'][0]['continues_before'] is True


def test_pipeline_reuses_person_ids_and_boxes_in_overlapping_windows(tmp_path, monkeypatch):
    import av
    from low_freq_event.config import Settings
    from low_freq_event.pipeline import analyze
    from low_freq_event.schema import EntityList, GroundingResult, ScanResult
    from low_freq_event.video import sliding_windows
    from test_pipeline import FakeBackend
    video = tmp_path / 'source.mp4'
    with av.open(str(video), 'w') as container:
        stream = container.add_stream('libx264', rate=6)
        stream.width, stream.height, stream.pix_fmt = 96, 64, 'yuv420p'
        for i in range(18):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((64,96,3), 60+i, dtype=np.uint8), format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    class Perception:
        def prepare(self, source, clip, registry, visual_state):
            registry.setdefault('person_0001', {'entity_id': 'person_0001', 'label': '인물 1', 'kind': 'person', 'appearance': '흰 옷',
                 'gallery': [], 'observations': [], 'first_seen_sec': 0., 'last_seen_sec': 2.5,
                 'possible_matches': [], 'identity_status': 'matched'})
            return [dict(frame_index=i, time_sec=f.time_sec, image=f.path,
                observations=[dict(entity_id='person_0001', bbox=[200.,100.,800.,900.], identity_uncertain=False)])
                    for i,f in enumerate(clip.frames)]
    class Backend(FakeBackend):
        def discover(self, clip, **kwargs):
            return EntityList(entities=[])
        def ground_frame(self, frame, entities, frame_index=0, **kwargs):
            return GroundingResult(boxes=[])
        def scan(self, clip, entities, grounded):
            return ScanResult.model_validate({'entities':[e.model_dump() for e in entities], 'events':[
                {'event_type':'ongoing','action':'visible','description':'인물이 보임','entity_ids':['person_0001'],
                 'start_frame':0,'end_frame':len(clip.frames)-1,'continues_before':True,'continues_after':True,
                 'uncertain':False,'boundary_note':''}]})
    backend = Backend()
    result = analyze(video, tmp_path/'result', Settings(perception='person_reid', window_sec=2, stride_sec=1, sample_fps=2), backend=backend, perception=Perception())
    assert result['status'] == 'complete'
    assert backend.identity_calls == 0
    assert len(result['entities']) == 1
    same_frame = [o for w in result['windows'] for o in w['observations'] if o['time_sec']==1.]
    assert len(same_frame) == 2
    assert len({(o['entity_id'],tuple(o['bbox'])) for o in same_frame}) == 1
    assert all(e['entity_ids']==['person_0001'] for e in result['events'])
