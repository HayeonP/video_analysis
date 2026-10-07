from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import av
import numpy as np
from PIL import Image
import pytest

from low_freq_event.backend import QwenBackend
from low_freq_event.config import Settings
from low_freq_event.identity import canonical, combine_matches, resolve_frame
from low_freq_event.pipeline import analyze, normalized_ids, output_lock, window_events
from low_freq_event.schema import EntityList, GroundingResult, Matches, ScanResult
from low_freq_event.storage import atomic_json, read_json
from low_freq_event.timing import RunTimer
from low_freq_event.video import Clip, Frame
from low_freq_event.viewer import generate_viewer


def scan(count=4):
    return ScanResult.model_validate({"entities": [{"local_id": "woman", "label": "여성", "kind": "character", "appearance": "붉은 머리"}],
        "events": [{"event_type": "ongoing", "action": "stand", "description": "여성이 서 있다", "entity_ids": ["woman"],
                    "start_frame": 0, "end_frame": count - 1, "continues_before": True, "continues_after": True,
                    "uncertain": False, "boundary_note": "시작과 종료 미관찰"}]}).validate_clip(count)


class FakeBackend:
    def __init__(self, interrupt_ground=None):
        self.scan_calls = self.ground_calls = self.identity_calls = self.discover_calls = 0
        self.interrupt_ground = interrupt_ground

    def narrate(self, clip, entities, grounded):
        from low_freq_event.schema import WindowNarrative
        return WindowNarrative(background="설산이 보인다.", summary="인물이 서 있다.",
            activities=[dict(entity_id=e.local_id, description="화면에 보인다.", uncertain=False) for e in entities])

    def discover(self, clip, **kwargs):
        self.discover_calls += 1
        return EntityList(entities=scan(len(clip.frames)).entities)

    def scan(self, clip, entities, grounded):
        self.scan_calls += 1
        result = scan(len(clip.frames))
        result.entities = entities
        result.events[0].entity_ids = [entities[0].local_id]
        return result

    def ground_frame(self, frame, entities, frame_index=0, **kwargs):
        self.ground_calls += 1
        if self.ground_calls == self.interrupt_ground:
            raise KeyboardInterrupt()
        return GroundingResult.model_validate({"boxes": [{"local_id": "woman", "bbox": [100, 100, 700, 900]}]})

    def identify_batch(self, queries, references):
        self.identity_calls += 1
        return Matches.model_validate({"matches": [{"query_id": q["observation_id"], "match_id": references[0]["entity_id"],
                                                   "possible_ids": [], "reason": "얼굴과 의상이 일치"} for q in queries]})


def test_optional_inference_warning_is_preserved_in_exports(video, tmp_path):
    class Backend(FakeBackend):
        def scan(self, clip, entities, grounded):
            result=super().scan(clip,entities,grounded)
            result.warnings=['행동 후보 형식 오류로 프레임 근거 분석만 사용']
            return result
    output=tmp_path/'result'
    data=analyze(video,output,Settings(window_sec=2,stride_sec=1,sample_fps=2),backend=Backend())
    assert all(w['warnings']==['행동 후보 형식 오류로 프레임 근거 분석만 사용'] for w in data['windows'])
    assert read_json(output/'windows/window_0000/scan.json')['warnings']==data['windows'][0]['warnings']


class EmptyPerception:
    def __init__(self, *args):
        pass

    def prepare(self, source, clip, registry, visual_state):
        return [dict(frame_index=i, time_sec=f.time_sec, image=f.path, observations=[]) for i, f in enumerate(clip.frames)]


@pytest.fixture(autouse=True)
def no_real_detector(monkeypatch):
    monkeypatch.setattr('low_freq_event.pipeline.make_perception', lambda *args: EmptyPerception())


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "source.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=6)
        stream.width, stream.height = 96, 64
        stream.pix_fmt = "yuv420p"
        for i in range(18):
            frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), 40 + i * 5, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_resume_keeps_every_frame_identity_and_time(video, tmp_path):
    output = tmp_path / "output"
    settings = Settings(window_sec=2, stride_sec=1, sample_fps=2, max_pixels=4096)
    first = FakeBackend()
    paused = analyze(video, output, settings, backend=first, max_windows=1)
    assert paused["status"] == "paused"
    assert first.ground_calls == 4
    assert first.identity_calls == 3
    assert len(paused["entities"]) == 1
    assert paused["events"][0]["duration_sec"] == 2
    assert paused["events"][0]["frame_indices"] == [0, 1, 2, 3]
    previous = paused["timing"]["total_active_sec"]
    time.sleep(.05)
    second = FakeBackend()
    complete = analyze(video, output, settings, backend=second)
    assert complete["status"] == "complete"
    assert second.scan_calls == 1  # Completed first window is not rerun.
    assert len(complete["events"]) == 2
    assert len(complete["entities"]) == 1
    assert len(complete["entities"][0]["observations"]) == 8
    sessions = complete["timing"]["sessions"]
    assert len(sessions) == 2 and sessions[0]["status"] == "paused"
    assert complete["timing"]["total_active_sec"] == pytest.approx(sum(s["elapsed_sec"] for s in sessions), abs=.001)
    assert complete["timing"]["total_active_sec"] >= previous
    third = FakeBackend()
    analyze(video, output, settings, backend=third)
    assert third.scan_calls == third.ground_calls == third.identity_calls == 0
    assert len(read_json(output / "timing.json")["sessions"]) == 2
    for window in complete["windows"]:
        assert len(window["frames"]) == 4
        for obs in window["observations"]:
            assert (output / obs["crop"]).is_file()
            assert obs["entity_id"] == "entity_0001"


def test_interrupt_inside_window_reuses_scan_and_grounding(video, tmp_path):
    output = tmp_path / "result"
    settings = Settings(window_sec=3, sample_fps=2, max_pixels=4096)
    with pytest.raises(KeyboardInterrupt):
        analyze(video, output, settings, backend=FakeBackend(interrupt_ground=3))
    assert read_json(output / "checkpoint.json")["status"] == "interrupted"
    assert read_json(output / "timing.json")["sessions"][0]["status"] == "interrupted"
    resumed = FakeBackend()
    result = analyze(video, output, settings, backend=resumed)
    assert resumed.discover_calls == 0 and resumed.scan_calls == 1 and resumed.ground_calls == 4
    assert result["status"] == "complete"
    assert len(result["entities"][0]["observations"]) == 6
    with pytest.raises(ValueError, match="다릅니다"):
        analyze(video, output, settings.model_copy(update={"sample_fps": 1}), backend=FakeBackend())


def test_event_reanalysis_reuses_final_identity_and_resumes_without_detection(video, tmp_path):
    from low_freq_event.reanalyze import reanalyze_events
    original = tmp_path/'original'
    destination = tmp_path/'revised'
    old = analyze(video,original,Settings(window_sec=2,stride_sec=1,sample_fps=2,max_pixels=4096),backend=FakeBackend())
    first = FakeBackend()
    paused = reanalyze_events(original,destination,backend=first,max_windows=1)
    assert paused['status']=='paused' and first.scan_calls==1
    assert first.ground_calls==first.discover_calls==first.identity_calls==0
    second = FakeBackend()
    complete = reanalyze_events(original,destination,backend=second)
    assert complete['status']=='complete' and second.scan_calls==1
    assert complete['entities']==old['entities']
    assert [w['observations'] for w in complete['windows']]==[w['observations'] for w in old['windows']]
    assert complete['timing']['total_active_sec']>=old['timing']['total_active_sec']
    assert len(complete['timing']['sessions'])==len(old['timing']['sessions'])+2
    for w in complete['windows']:
        assert all((destination/f['image']).is_file() for f in w['frames'])
    final = FakeBackend()
    reanalyze_events(original,destination,backend=final)
    assert final.scan_calls==0


def test_observed_interval_is_not_one_sampling_interval():
    clip = Clip(10, 15, 24, [Frame(10 + i * 2 / 3, Image.new("RGB", (64, 64)), f"f{i}.jpg") for i in range(8)])
    result = scan(8)
    event = result.events[0]
    event.start_frame, event.end_frame = 1, 5
    event.continues_before = event.continues_after = False
    exported = window_events(result, clip, [], {}, "w")[0]
    assert exported["start_sec"] == pytest.approx(10 + 2 / 3)
    assert exported["end_sec"] == pytest.approx(10 + 10 / 3)
    assert exported["duration_sec"] == pytest.approx(8 / 3)
    assert exported["frame_indices"] == [1, 2, 3, 4, 5]
    assert exported["participants"][0]["localization_status"] == "not_located"
    assert exported["uncertain"]
    event.end_frame = 1
    assert window_events(result, clip, [], {}, "w")[0]["duration_sec"] == 0
    assert clip.metadata()["fps"] == 24
    assert clip.metadata()["frames_indices"][5] / 24 == pytest.approx(10 / 3)


@pytest.mark.parametrize("change", [{"start_frame": 3, "end_frame": 1}, {"end_frame": 8}, {"entity_ids": ["missing"]}, {"start_frame": 1}])
def test_bad_model_spans_rejected(change):
    candidate = scan().model_dump()
    candidate["events"][0].update(change)
    with pytest.raises(ValueError):
        ScanResult.model_validate(candidate).validate_clip(4)


def test_request_caches_valid_result_and_does_not_accept_truncation(tmp_path, monkeypatch):
    backend = QwenBackend(Settings(json_retries=1), tmp_path / "model_calls")
    answers = iter(['{"boxes":', '{"boxes":[]}'])
    monkeypatch.setattr(backend, "_generate", lambda *args: next(answers))
    assert backend.request("test", GroundingResult).boxes == []
    assert len(list((tmp_path / "model_calls").glob("*.json"))) == 2
    assert backend.request("test", GroundingResult).boxes == []


def test_grounding_retries_copied_person_boxes_without_merging_overlaps(tmp_path, monkeypatch):
    entities = EntityList.model_validate({"entities": [
        {"local_id": "p1", "label": "사람", "kind": "person", "appearance": "흰 셔츠"},
        {"local_id": "p2", "label": "사람", "kind": "person", "appearance": "흰 셔츠"},
    ]}).entities
    invalid = {"boxes": [{"local_id": e.local_id, "bbox": [0, 0, 1000, 1000]} for e in entities]}
    # Near-identical overlapping boxes remain separate observations, even with
    # identical labels. Only exact copied coordinates request a model retry.
    valid = {"boxes": [{"local_id": "p1", "bbox": [100, 100, 700, 900]},
                       {"local_id": "p2", "bbox": [100, 100, 700, 901]}]}
    backend = QwenBackend(Settings(json_retries=1), tmp_path / "model_calls")
    answers = iter([json.dumps(invalid), json.dumps(valid)])
    monkeypatch.setattr(backend, "_generate", lambda *args: next(answers))
    result = backend.ground_frame(Frame(0, Image.new("RGB", (64, 64))), entities)
    assert [b.local_id for b in result.boxes] == ["p1", "p2"]
    assert len(list((tmp_path / "model_calls").glob("*.json"))) == 2
    assert "identical copied coordinates" in read_json(tmp_path / "model_calls/000001.json")["validation_error"]


def test_grounding_allows_background_and_one_subject_to_share_bounds(tmp_path, monkeypatch):
    entities = EntityList.model_validate({"entities": [
        {"local_id": "bg", "label": "배경", "kind": "environment", "appearance": "방"},
        {"local_id": "p1", "label": "인물", "kind": "person", "appearance": "클로즈업"},
    ]}).entities
    answer = {"boxes": [{"local_id": e.local_id, "bbox": [0, 0, 1000, 1000]} for e in entities]}
    backend = QwenBackend(Settings(json_retries=0), tmp_path / "model_calls")
    monkeypatch.setattr(backend, "_generate", lambda *args: json.dumps(answer))
    assert len(backend.ground_frame(Frame(0, Image.new("RGB", (64, 64))), entities).boxes) == 2


def test_grounding_failure_is_explicit_and_does_not_invent_boxes(tmp_path, monkeypatch):
    backend = QwenBackend(Settings(json_retries=1), tmp_path / "model_calls")
    monkeypatch.setattr(backend, "_generate", lambda *args: '{"boxes":')
    result = backend.ground_frame(Frame(0, Image.new("RGB", (64, 64))), [])
    assert result.boxes == [] and result.new_entities == []
    assert result.warnings and "검출 실패" in result.warnings[0]
    assert len(list((tmp_path / "model_calls").glob("*.json"))) == 2


def test_pipeline_preserves_failed_frame_and_marks_affected_event(video, tmp_path):
    class MissingBoxBackend(FakeBackend):
        def ground_frame(self, frame, entities, frame_index=0, **kwargs):
            if frame_index == 1:
                return GroundingResult(boxes=[], warnings=["bbox 검출 실패"])
            return super().ground_frame(frame, entities, frame_index)
    result = analyze(video, tmp_path / "result", Settings(window_sec=3, sample_fps=1, max_pixels=4096),
                     backend=MissingBoxBackend())
    assert result["status"] == "complete"
    assert len(result["windows"][0]["frames"]) == 3
    assert result["windows"][0]["frames"][1]["warnings"]
    assert result["events"][0]["uncertain"]
    assert "bbox 검출 실패" in result["events"][0]["boundary_note"]
    assert len(result["events"][0]["evidence"]) == 3


def test_identity_recompares_conflicts_without_discarding_other_queries(tmp_path, monkeypatch):
    backend = QwenBackend(Settings(json_retries=0), tmp_path / "model_calls")
    images = [Image.new("RGB", (64, 64), color) for color in ("red", "green", "blue")]
    queries = [{"observation_id": f"obs{i}", "image": image} for i, image in enumerate(images, 1)]
    references = [{"entity_id": f"entity_{i}", "images": [image]} for i, image in enumerate(images, 1)]
    match = lambda q, r: {"query_id": q, "match_id": r, "possible_ids": [], "reason": "시각적 비교"}
    answers = iter([
        {"matches": [match("Q1", "R1"), match("Q2", "R1"), match("Q3", "R3")]},
        {"matches": [match("Q1", "R1")]},
        {"matches": [match("Q1", "R2")]},
    ])
    monkeypatch.setattr(backend, "_generate", lambda *args: json.dumps(next(answers)))
    result = backend.identify_batch(queries, references)
    assert [(m.query_id, m.match_id) for m in result.matches] == [
        ("obs1", "entity_1"), ("obs2", "entity_2"), ("obs3", "entity_3")]
    assert "개별 비교" in result.matches[0].reason
    assert result.matches[2].reason == "시각적 비교"
    assert len(list((tmp_path / "model_calls").glob("*.json"))) == 3


def test_timer_recovery_excludes_offline_gap(tmp_path):
    path = tmp_path / "timing.json"
    atomic_json(path, {"sessions": [{"id": "old", "status": "running", "elapsed_sec": 123,
                                     "started_at": "2000-01-01", "last_heartbeat_at": "2000-01-02"}]})
    timer = RunTimer(path, interval=.01)
    time.sleep(.03)
    result = timer.close("complete")
    assert result["sessions"][0]["status"] == "interrupted"
    assert 123 <= result["total_active_sec"] < 124


def test_viewer_relocation_and_safe_embedded_json(video, tmp_path):
    original = tmp_path / "old"
    settings = Settings(window_sec=3, sample_fps=1, max_pixels=4096)
    analyze(video, original, settings, backend=FakeBackend())
    relocated = tmp_path / "new"
    original.rename(relocated)
    video.unlink()  # The viewer owns an independent video copy.
    payload = read_json(relocated / "timeline.json")
    payload["events"][0]["description"] = '</script><script>alert("x")</script>'
    atomic_json(relocated / "timeline.json", payload)
    page = generate_viewer(relocated)
    assert '<script>alert("x")</script>' not in page.read_text()
    assert (relocated / "video.mp4").is_file()


def test_output_lock_excludes_parallel_writers(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="이미 분석"):
            with output_lock(tmp_path):
                pass


@pytest.mark.parametrize("match_id", [None, "entity_0001"])
def test_identity_honors_images_even_for_identical_box(tmp_path, match_id):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "crop.jpg")
    observation = {"observation_id": "q1", "label": "different label", "kind": "character", "appearance": "red",
                   "time_sec": 0, "bbox": [0, 0, 1000, 1000], "frame": "crop.jpg", "crop": "crop.jpg"}
    registry = {"entity_0001": {"entity_id": "entity_0001", "kind": "character", "label": "old label", "appearance": "red",
                              "gallery": ["crop.jpg"], "first_seen_sec": 0, "last_seen_sec": 0, "observations": []}}
    class Backend:
        def identify_batch(self, queries, refs):
            return Matches.model_validate({"matches": [{"query_id": "q1", "match_id": match_id, "possible_ids": [], "reason": "visual"}]})
    assignments = resolve_frame([observation], registry, tmp_path, Settings(), Backend())
    assert assignments["q1"] == (match_id or "entity_0002")


def test_identity_interruption_resumes_without_duplicate_observations(video, tmp_path):
    output = tmp_path / "result"
    settings = Settings(window_sec=3, sample_fps=2, max_pixels=4096)
    class Interrupt(FakeBackend):
        def identify_batch(self, queries, references):
            if self.identity_calls == 1:
                raise KeyboardInterrupt()
            return super().identify_batch(queries, references)
    with pytest.raises(KeyboardInterrupt):
        analyze(video, output, settings, backend=Interrupt())
    state = read_json(output / "checkpoint.json")
    assert state["pending"]["next_frame"] == 2
    assert len(state["registry"]["entity_0001"]["observations"]) == 2
    resumed = FakeBackend()
    result = analyze(video, output, settings, backend=resumed)
    assert resumed.discover_calls == resumed.ground_calls == 0
    assert resumed.scan_calls == 1  # Events now run after identity assignment.
    observations = result["entities"][0]["observations"]
    assert len(observations) == len({o["observation_id"] for o in observations}) == 6


def test_new_entities_found_in_frames_are_included_in_video_events(video, tmp_path):
    class DiscoverInFrame(FakeBackend):
        def discover(self, clip, **kwargs):
            self.discover_calls += 1
            return EntityList(entities=[])

        def ground_frame(self, frame, entities, frame_index=0, **kwargs):
            item = scan(6).entities[0].model_copy(update={"local_id": "f0_new1"})
            return GroundingResult.model_validate({"boxes": [{"local_id": item.local_id, "bbox": [100, 100, 700, 900]}],
                "new_entities": [item.model_dump()] if frame_index == 0 else []})

        def scan(self, clip, entities, grounded):
            assert [e.local_id for e in entities] == ["entity_0001"]
            result = scan(len(clip.frames))
            result.entities = entities
            result.events[0].entity_ids = ["entity_0001"]
            return result
    result = analyze(video, tmp_path / "out", Settings(window_sec=3, sample_fps=2, max_pixels=4096), backend=DiscoverInFrame())
    assert result["events"][0]["entity_ids"] == ["entity_0001"]
    assert len(result["entities"][0]["observations"]) == 6


def test_conflicting_visual_matches_across_pages_stay_uncertain(tmp_path):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "crop.jpg")
    query = {"observation_id": "q", "label": "인물", "appearance": "의상", "kind": "character", "crop": "crop.jpg", "time_sec": 1}
    registry = {f"entity_{i:04d}": {"entity_id": f"entity_{i:04d}", "kind": "character", "label": "인물", "appearance": "의상",
                "first_seen_sec": 0, "last_seen_sec": 0, "gallery": ["crop.jpg"],
                "observations": [{"window_id": "same", "frame_index": 0}]} for i in (1, 2)}
    class Conflicting:
        def identify_batch(self, queries, refs):
            return Matches.model_validate({"matches": [{"query_id": "q", "match_id": refs[0]["entity_id"],
                                                        "possible_ids": [], "reason": "같아 보임"}]})
    result = resolve_frame([query], registry, tmp_path, Settings(gallery_page_size=1), Conflicting())
    assert result["q"] == "entity_0003"
    assert registry["entity_0003"]["identity_status"] == "uncertain"
    assert registry["entity_0003"]["possible_matches"] == ["entity_0001", "entity_0002"]


def test_first_confident_search_stops_and_excludes_co_visible_ids(tmp_path):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "crop.jpg")
    queries = [{"observation_id": f"q{i}", "label": "같은 라벨", "appearance": "의상", "kind": "person",
                "crop": "crop.jpg", "time_sec": 1, "window_id": "w", "frame_index": 0} for i in (1, 2)]
    registry = {f"entity_{i:04d}": {"entity_id": f"entity_{i:04d}", "kind": "person", "label": "다른 라벨",
                "appearance": "의상", "first_seen_sec": 0, "last_seen_sec": 0,
                "gallery": ["crop.jpg"], "observations": []} for i in (1, 2, 3)}
    calls = []
    class Backend:
        def identify_batch(self, queries, refs):
            calls.append((queries[0]["observation_id"], [r["entity_id"] for r in refs]))
            return Matches.model_validate({"matches": [{"query_id": queries[0]["observation_id"],
                "match_id": refs[0]["entity_id"], "possible_ids": [], "reason": "시각적 일치"}]})
    result = resolve_frame(queries, registry, tmp_path, Settings(identity_search="first_confident", gallery_page_size=1), Backend())
    assert result == {"q1": "entity_0001", "q2": "entity_0002"}
    assert calls == [("q1", ["entity_0001"]), ("q2", ["entity_0002"])]


def test_first_confident_search_checks_later_pages_when_uncertain(tmp_path):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "crop.jpg")
    query = {"observation_id": "q", "label": "인물", "appearance": "의상", "kind": "person", "crop": "crop.jpg", "time_sec": 1}
    registry = {f"entity_{i:04d}": {"entity_id": f"entity_{i:04d}", "kind": "person", "label": "인물",
                "appearance": "의상", "first_seen_sec": 0, "last_seen_sec": 0,
                "gallery": ["crop.jpg"], "observations": []} for i in (1, 2, 3)}
    calls = []
    class Backend:
        def identify_batch(self, queries, refs):
            eid = refs[0]["entity_id"];calls.append(eid)
            return Matches.model_validate({"matches": [{"query_id": "q", "match_id": eid if eid=="entity_0002" else None,
                "possible_ids": [] if eid=="entity_0002" else [eid], "reason": "시각적 비교"}]})
    result = resolve_frame([query], registry, tmp_path, Settings(identity_search="first_confident", gallery_page_size=1), Backend())
    assert result["q"] == "entity_0002"
    assert calls == ["entity_0001", "entity_0002"]
    assert registry["entity_0002"]["possible_matches"] == ["entity_0001"]


def test_positive_visual_matches_unify_separate_tracks_and_exported_ids():
    registry = {f"entity_{i:04d}": {"entity_id": f"entity_{i:04d}", "first_seen_sec": i, "last_seen_sec": i,
                "gallery": [f"crop{i}.jpg"], "observations": [{"window_id": "w", "frame_index": i,
                                                              "entity_id": f"entity_{i:04d}"}]} for i in (1, 2)}
    assert combine_matches(registry, {"entity_0001", "entity_0002"}, 2) == "entity_0001"
    assert canonical(registry, "entity_0002") == "entity_0001"
    value = {"events": [{"entity_ids": ["entity_0001", "entity_0002"], "participants": [{"entity_ids": ["entity_0002"]}]}],
             "observations": [{"entity_id": "entity_0002", "local_id": "e2"}]}
    normalized = normalized_ids(value, registry)
    assert normalized["events"][0]["entity_ids"] == ["entity_0001"]
    assert normalized["events"][0]["participants"][0]["entity_ids"] == ["entity_0001"]
    assert normalized["observations"][0] == {"entity_id": "entity_0001", "local_id": "e2"}


def test_model_page_tokens_translate_back_to_persistent_ids(tmp_path, monkeypatch):
    backend = QwenBackend(Settings(), tmp_path / "model_calls")
    def generate(prompt, clip, images, *args):
        assert [label for label, image in images] == ["QUERY Q1", "REFERENCE R1"]
        assert "외형" not in prompt  # Model-generated descriptions must not bias crop identity.
        return '{"matches":[{"query_id":"Q1","match_id":"R1","possible_ids":[],"reason":"same"}]}'
    monkeypatch.setattr(backend, "_generate", generate)
    image = Image.new("RGB", (64, 64))
    result = backend.identify_batch([{"observation_id": "window_0099_f07_o01", "label": "인물", "appearance": "외형", "image": image}],
                                    [{"entity_id": "entity_0823", "label": "인물", "appearance": "외형", "images": [image]}])
    assert result.matches[0].query_id == "window_0099_f07_o01"
    assert result.matches[0].match_id == "entity_0823"


def test_background_crop_containing_a_person_cannot_become_that_person(tmp_path):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "crop.jpg")
    registry = {"entity_0001": {"entity_id": "entity_0001", "kind": "environment", "label": "설산",
                "appearance": "배경과 인물", "gallery": ["crop.jpg"], "observations": [],
                "first_seen_sec": 0, "last_seen_sec": 0}}
    query = {"observation_id": "q", "kind": "character", "label": "인물", "appearance": "붉은 머리",
             "crop": "crop.jpg", "time_sec": 1}
    class RejectCall:
        def identify_batch(self, queries, refs):
            raise AssertionError("Environment and person are not identity candidates for each other")
    result = resolve_frame([query], registry, tmp_path, Settings(), RejectCall())
    assert result["q"] == "entity_0002"
    assert registry["entity_0002"]["kind"] == "character"
