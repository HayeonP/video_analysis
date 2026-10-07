"""Reconstruct a first-pass review from saved JSON, without re-running analysis."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from .paths import result_directory
from .viewer import write_viewer


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def observation_key(frame: str, time_sec: float, bbox: list) -> tuple:
    return frame, round(time_sec, 6), tuple(bbox)


def first_round_payload(result: Path) -> dict:
    base = result_directory(result, "candidates.json")
    required = ("candidates.json", "checkpoint.json", "manifest.json")
    for name in required:
        if not (base / name).is_file():
            raise ValueError(f"저장된 1차 분석 기록이 필요합니다: {base / name}")
    events = deepcopy(read(base / "candidates.json"))
    if not isinstance(events, list) or any(event.get("stage") != "coarse" for event in events):
        raise ValueError("candidates.json에는 1차(coarse) 이벤트 목록만 있어야 합니다.")
    checkpoint = read(base / "checkpoint.json")
    metadata = deepcopy(read(base / "manifest.json"))
    entries = checkpoint.get("registry", {}).get("entries", {})
    coarse_boxes, cuts = set(), set()
    windows = 0
    for path in sorted((base / "windows").glob("coarse_*.json")):
        if path.name.endswith(".entities.json"):
            continue
        window = read(path)
        windows += 1
        for cut in window["result"].get("scene_cuts_sec", []):
            cuts.add(round(window["start_sec"] + cut, 6))
        localized = path.with_name(path.stem + ".entities.json")
        if not localized.is_file():
            continue  # Do not substitute unverified scan boxes for localization.
        for observation in read(localized):
            index = observation["frame_index"]
            if not 0 <= index < len(window["frames"]):
                continue
            frame = window["frames"][index]
            coarse_boxes.add(observation_key(frame["image"], frame["time_sec"], observation["bbox"]))

    # Final reconciliation appends source observations to destination entities.
    # The saved crop path still identifies the ORIGINAL assignment. Recover that
    # owner rather than applying final aliases or inventing a new identity match.
    by_owner: dict[str, dict] = {}
    for entry in entries.values():
        for observation in entry.get("observations", []):
            key = observation_key(observation["frame"], observation["time_sec"], observation["bbox"])
            if key not in coarse_boxes:
                continue
            parts = PurePosixPath(observation.get("crop", "")).parts
            if len(parts) < 3 or parts[0] != "entities" or parts[1] not in entries:
                continue  # An original owner cannot be established from this record.
            owner = parts[1]
            by_owner.setdefault(owner, {})[key] = deepcopy(observation)

    used_ids = {value for event in events for value in (event.get("actor_id"), event.get("object_id")) if value}
    entities = []
    for entity_id in sorted(set(by_owner) | used_ids):
        original = entries.get(entity_id, {})
        observations = sorted(by_owner.get(entity_id, {}).values(), key=lambda value: value["time_sec"])
        gallery = list(dict.fromkeys(observation["crop"] for observation in observations))
        entities.append({
            "entity_id": entity_id, "kind": original.get("kind", "other"),
            "label": original.get("label", entity_id), "appearance": original.get("appearance", ""),
            "identity_status": "unverified", "observations": observations,
            "gallery": gallery[:1] + gallery[-1:] if len(gallery) > 1 else gallery,
            "first_seen_sec": observations[0]["time_sec"] if observations else None,
            "last_seen_sec": observations[-1]["time_sec"] if observations else None,
            "aliases": [], "possible_matches": [],
        })
    for event in events:
        event["identity_status"] = "unresolved" if event.get("unresolved_local_ids") else "unverified"
    errors = [deepcopy(error) for error in checkpoint.get("analysis_errors", []) if error.get("stage") == "coarse"]
    metadata.update(analysis_status="first_round_unverified", error_count=len(errors), coarse_windows=windows)
    # source_fps is optional in the first-pass manifest; leaving it absent lets the
    # viewer use its conservative timing tolerance without opening the video here.
    return {
        "schema_version": 1, "view_mode": "first_round", "video": metadata,
        "events": events, "entities": entities, "scene_cuts_sec": sorted(cuts), "analysis_errors": errors,
        "review_notice": (
            "1차 분석 후보 · 미검증. candidates.json의 후보를 그대로 표시합니다. "
            "정밀 분석과 최종 ID 병합은 적용하지 않았습니다. "
            "설명·시간·동일 대상 판단에 모델 오류가 있을 수 있습니다. "
            "박스는 1차 위치 기록과 일치하고 원래 대상 ID를 확인할 수 있는 저장 관찰만 표시합니다."
        ),
        "provenance": {"events": "candidates.json", "entity_ids": "checkpoint.json / original crop owner",
                       "boxes": "windows/coarse_*.entities.json + matching saved observations"},
    }


def generate_first_round_viewer(result: Path, *, output: Path | None = None,
                                video: Path | None = None) -> Path:
    base = result_directory(result, "candidates.json")
    return write_viewer(first_round_payload(base), base=base,
                        destination=output or base / "first_round.html", video=video)
