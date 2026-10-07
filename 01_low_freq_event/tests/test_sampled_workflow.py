from pathlib import Path

import av
import pytest
from PIL import Image

from test_pipeline import video, FakeBackend, EmptyPerception
from low_freq_event.backend import QwenBackend
from low_freq_event.config import Settings
from low_freq_event.pipeline import analyze
from low_freq_event.schema import LocalEntity, WindowNarrative
from low_freq_event.video import VideoSource, prepare_samples, sampled_clip, Clip, Frame


def test_global_grid_shared_across_fractional_windows(video, tmp_path):
    source = VideoSource(video, 4096)
    samples = prepare_samples(source, tmp_path, 1)
    assert [s['time_sec'] for s in samples] == [0, 1, 2]
    first = sampled_clip(source, tmp_path, samples, 0, 2)
    second = sampled_clip(source, tmp_path, samples, .5, 2.5)
    assert [f.time_sec for f in second.frames] == [1, 2]
    assert first.frames[1].path == second.frames[0].path
    with av.open(str(tmp_path / 'sampled.mp4')) as container:
        assert float(container.streams.video[0].average_rate) == 1
        assert (container.streams.video[0].width,container.streams.video[0].height)==(96,64)
        assert len(list(container.decode(video=0))) == 3


def test_all_tracking_precedes_vlm_and_narrative_survives_resume(video, tmp_path):
    seen = set()
    class Perception(EmptyPerception):
        def prepare(self, source, clip, registry, state):
            seen.update(f.time_sec for f in clip.frames)
            return super().prepare(source, clip, registry, state)
    class Backend(FakeBackend):
        def discover(self, clip, **kwargs):
            assert seen == {0, 1, 2}
            return super().discover(clip, **kwargs)
    output = tmp_path / 'result'
    settings = Settings(window_sec=2, max_pixels=4096)
    data = analyze(video, output, settings, backend=Backend(), perception=Perception())
    assert all(w['narrative']['background'] for w in data['windows'])
    again = analyze(video, output, settings, backend=Backend(), perception=Perception())
    assert again['video']['samples'] == data['video']['samples']
    assert again['windows'] == data['windows']


def test_narration_receives_all_images_and_requires_each_entity(tmp_path, monkeypatch):
    backend = QwenBackend(Settings(), tmp_path)
    entity = LocalEntity(local_id='person_0001', label='인물 1', kind='person', appearance='붉은 옷')
    clip = Clip(0, 2, 30, [Frame(i, Image.new('RGB', (32, 32))) for i in range(2)])
    grounded = [dict(boxes=[dict(local_id=entity.local_id, bbox=[0, 0, 1000, 1000])])] * 2
    def request(prompt, schema, *, clip, validate):
        assert len(clip.frames) == 2
        assert 'person_0001' in prompt and '한국어' in prompt
        with pytest.raises(ValueError):
            validate(WindowNarrative(background='방', summary='서 있다', activities=[dict(entity_id='unknown', description='알 수 없음', uncertain=True)]))
        return validate(WindowNarrative(background='방', summary='인물 1이 서 있다', activities=[
            dict(entity_id=entity.local_id, description='서 있다', uncertain=False)]))
    monkeypatch.setattr(backend, 'request', request)
    assert backend.narrate(clip, [entity], grounded).activities[0].entity_id == entity.local_id


def test_real_people_only_response_completes_missing_entities(tmp_path, monkeypatch):
    import json
    fixture = json.loads((Path(__file__).parent / 'fixtures/people_only_narrative.json').read_text())
    entities = [LocalEntity(**e) for e in fixture['entities']]
    clip = Clip(0, 5, 30, [Frame(i, Image.new('RGB', (32, 32))) for i in range(5)])
    grounded = [dict(boxes=[dict(local_id=e.local_id, bbox=[0, 0, 1000, 1000]) for e in entities]) for _ in clip.frames]
    backend = QwenBackend(Settings(), tmp_path)
    calls = []
    def generate(prompt, clip, images, *args):
        calls.append(prompt)
        assert len(clip.frames) == 5
        if len(calls) == 1:
            return fixture['response']
        return json.dumps(dict(description='이 대상은 정지해 있다.', uncertain=True))
    monkeypatch.setattr(backend, '_generate', generate)
    result = backend.narrate(clip, entities, grounded)
    assert [a.entity_id for a in result.activities] == [e.local_id for e in entities]
    assert len(calls) == 16  # Real response omitted 13 objects and 2 backgrounds.
    backend.narrate(clip, entities, grounded)
    assert len(calls) == 16  # Completed calls survive a retry/resume.


def test_narrative_failure_keeps_completed_scan(video, tmp_path):
    from low_freq_event.backend import ModelOutputError
    from low_freq_event.storage import read_json
    class FailsNarrative(FakeBackend):
        def narrate(self, *args):
            raise ModelOutputError('invalid narrative')
    output = tmp_path / 'result'
    settings = Settings(window_sec=3, max_pixels=4096)
    with pytest.raises(ModelOutputError):
        analyze(video, output, settings, backend=FailsNarrative(), perception=EmptyPerception())
    assert read_json(output / 'windows/window_0000/scan.json')['events']
    resumed = FakeBackend()
    result = analyze(video, output, settings, backend=resumed, perception=EmptyPerception())
    assert resumed.scan_calls == 0
    assert result['status'] == 'complete'
    assert result['windows'][0]['narrative']['activities']
