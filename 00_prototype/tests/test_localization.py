from __future__ import annotations

import json

from PIL import Image
import pytest

from video_events.backend import QwenBackend
from video_events.config import Settings
from video_events.events import global_events, merge_events
from video_events.export import export_results
from video_events.schema import Observation, ScanResult
from video_events.video import Clip, Frame


@pytest.fixture
def clip():
    return Clip(0, 3, 24, [Frame(float(index), Image.new("RGB", (64, 64), (index, 0, 0)),
                               f"frames/{index}.jpg") for index in range(3)])


def observation(local_id="e1", frame_index=0):
    return Observation(local_id=local_id, kind="character", label="캐릭터",
                       appearance="붉은 망토", frame_index=frame_index, bbox=(100, 100, 300, 900))


def response(*local_ids):
    return json.dumps({"boxes": [{"local_id": key, "bbox": [200, 100, 500, 900]} for key in local_ids]})


def fake_model(monkeypatch, backend, answers):
    calls = []
    replies = iter(answers)

    def generate(prompt, clip, images, max_tokens, image_pixels):
        calls.append({"prompt": prompt, "frame": images[0][1].getpixel((0, 0))[0]})
        return next(replies)

    monkeypatch.setattr(backend, "_generate", generate)
    return calls


def test_empty_box_response_relocates_entity_without_reusing_unconfirmed_box(tmp_path, clip, monkeypatch):
    backend = QwenBackend(Settings(), tmp_path / "calls")
    # Regression: the actual Sintel failure returned {"boxes": []} twice for a
    # representative frame where the character was not visible.
    calls = fake_model(monkeypatch, backend, [response(), response("e1")])
    original = observation()
    localized = backend.localize(clip, [original])
    assert [call["frame"] for call in calls] == [0, 2]
    assert localized[0].frame_index == 2
    assert localized[0].bbox == (200, 100, 500, 900)
    assert original.frame_index == 0 and original.bbox == (100, 100, 300, 900)
    assert "validation_error" not in json.loads((tmp_path / "calls/000001.json").read_text())


def test_partial_detection_searches_only_missing_entities(tmp_path, clip, monkeypatch):
    backend = QwenBackend(Settings(), tmp_path / "calls")
    calls = fake_model(monkeypatch, backend, [response("e1"), response("e2")])
    localized = backend.localize(clip, [observation("e1"), observation("e2")])
    assert [(item.local_id, item.frame_index) for item in localized] == [("e1", 0), ("e2", 2)]
    assert '"local_id": "e1"' not in calls[1]["prompt"]
    assert '"local_id": "e2"' in calls[1]["prompt"]


def test_absent_target_checks_each_frame_once_without_fabricating_a_crop(tmp_path, clip, monkeypatch):
    backend = QwenBackend(Settings(), tmp_path / "calls")
    calls = fake_model(monkeypatch, backend, [response(), response(), response()])
    assert backend.localize(clip, [observation()]) == []
    assert [call["frame"] for call in calls] == [0, 2, 1]


@pytest.mark.parametrize("invalid_ids", [("unexpected",), ("e1", "e1")])
def test_unknown_or_duplicate_ids_still_fail_validation_and_are_retried(tmp_path, clip, monkeypatch, invalid_ids):
    backend = QwenBackend(Settings(json_retries=1), tmp_path / "calls")
    calls = fake_model(monkeypatch, backend, [response(*invalid_ids), response("e1")])
    localized = backend.localize(clip, [observation()])
    assert localized[0].frame_index == 0
    assert [call["frame"] for call in calls] == [0, 0]
    trace = json.loads((tmp_path / "calls/000001.json").read_text())
    assert "Grounding IDs" in trace["validation_error"]


def test_exhausted_invalid_output_can_recover_on_another_frame(tmp_path, clip, monkeypatch):
    backend = QwenBackend(Settings(json_retries=1), tmp_path / "calls")
    calls = fake_model(monkeypatch, backend, ["not JSON", "not JSON", response("e1")])
    assert backend.localize(clip, [observation()])[0].frame_index == 2
    assert [call["frame"] for call in calls] == [0, 0, 2]


def test_runtime_failures_are_not_treated_as_missing_detections(tmp_path, clip, monkeypatch):
    backend = QwenBackend(Settings(), tmp_path / "calls")

    def fail(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(backend, "_generate", fail)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        backend.localize(clip, [observation()])


def test_unlocalized_actor_is_preserved_as_uncertain_without_false_merge_or_assigned_identity(tmp_path, clip):
    scan = ScanResult.model_validate({
        "entities": [observation().model_dump()],
        "events": [{"event_type": "appear", "start_sec": 1, "end_sec": 2,
                    "actor_id": "e1", "action": "become_visible", "description": "캐릭터가 등장한다",
                    "evidence_frames": [1]}],
    })
    first = global_events(scan, clip, {}, "coarse_0", "coarse")[0]
    second = global_events(scan, clip, {}, "coarse_1", "coarse")[0]
    assert first.actor_id is None
    assert first.uncertain and first.unresolved_local_ids == ["e1"]
    assert first.review_reason == "unlocalized_entities"
    assert first.evidence and first.source_windows == ["coarse_0"]
    assert len(merge_events([first, second], 0.35)) == 2
    assert not scan.events[0].uncertain  # Keep the cached model response intact.
    export_results(tmp_path, tmp_path / "video.mp4", {}, [first], [], [])
    exported = json.loads((tmp_path / "timeline.json").read_text())["events"][0]
    assert exported["identity_status"] == "unresolved"
    assert exported["unresolved_local_ids"] == ["e1"]


def test_successful_mapping_does_not_mark_events_unresolved(clip):
    scan = ScanResult.model_validate({
        "entities": [observation().model_dump()],
        "events": [{"event_type": "action", "start_sec": 1, "end_sec": 2,
                    "actor_id": "e1", "action": "walk", "description": "걷는다", "evidence_frames": [1]}],
    })
    event = global_events(scan, clip, {"e1": "entity_0001"}, "coarse_0", "coarse")[0]
    assert event.actor_id == "entity_0001" and not event.uncertain
    assert event.unresolved_local_ids == []
