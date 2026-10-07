"""A matching frame/box must never override the visual identity decision."""
import json

from PIL import Image
import pytest

from video_events.backend import QwenBackend
from video_events.config import Settings
from video_events.identity import EntityRegistry
from video_events.schema import Observation
from video_events.video import Clip, Frame


@pytest.mark.parametrize("decision", ["match", "different", "uncertain"])
def test_identical_frame_and_box_respects_visual_identity_decision(tmp_path, monkeypatch, decision):
    settings = Settings()
    registry = EntityRegistry(tmp_path, settings)
    backend = QwenBackend(settings, tmp_path / "model_calls")
    clip = Clip(0, 1, 24, [Frame(0, Image.new("RGB", (100, 100), "red"), "frames/0.jpg")])
    observation = Observation(local_id="first", kind="object", label="물체",
                              appearance="붉은 물체", frame_index=0, bbox=(100, 100, 500, 900))
    calls = []

    def generate(prompt, clip, images, max_tokens, image_pixels):
        calls.append(images)
        return json.dumps({"match_id": "entity_0001" if decision == "match" else None,
                           "possible_ids": ["entity_0001"] if decision == "uncertain" else [],
                           "reason": "대표 이미지 비교 결과"})

    monkeypatch.setattr(backend, "_generate", generate)
    assert registry.resolve([observation], clip, backend) == {"first": "entity_0001"}
    assert calls == []  # No existing entity to compare when registering the first one.

    next_observation = observation.model_copy(update={"local_id": "next"})
    mapping = registry.resolve([next_observation], clip, backend)
    assert len(calls) == 1
    assert calls[0][0][0] == "QUERY entity crop"
    assert calls[0][1][0].startswith("REFERENCE entity_0001:")
    entity_id = "entity_0001" if decision == "match" else "entity_0002"
    assert mapping == {"next": entity_id}
    entry = registry.entries[entity_id]
    assert entry["identity_status"] == {"match": "matched", "different": "new", "uncertain": "provisional"}[decision]
    assert entry["possible_matches"] == (["entity_0001"] if decision == "uncertain" else [])
    assert entry["match_evidence"][-1]["reasons"] == ["대표 이미지 비교 결과"]
