"""Build an offline review page from completed results, using only the stdlib.

This module must not import the analysis pipeline, torch, or video decoders.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from .paths import resolve_video, result_directory
from .viewer_template import HTML


def generate_viewer(result: Path, *, output: Path | None = None, video: Path | None = None) -> Path:
    source = result_directory(result, "timeline.json") / "timeline.json"
    if not source.is_file():
        raise ValueError(f"완료된 timeline.json이 없습니다. 분석 완료 후 실행하세요: {source}")
    base = source.parent
    checkpoint = base / "checkpoint.json"
    if checkpoint.exists() and json.loads(checkpoint.read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("아직 분석 중이거나 중단된 결과입니다. 분석 완료 후 시각화를 생성하세요.")
    data = json.loads(source.read_text(encoding="utf-8"))
    return write_viewer(data, base=base, destination=output or base / "viewer.html", video=video)


def write_viewer(data: dict, *, base: Path, destination: Path, video: Path | None = None) -> Path:
    """Render an already assembled result without loading models or decoding video."""
    base = base.resolve()
    if not isinstance(data, dict) or not isinstance(data.get("video"), dict):
        raise ValueError("영상 정보가 포함된 timeline.json이 필요합니다.")
    if not isinstance(data.get("events"), list) or not isinstance(data.get("entities"), list):
        raise ValueError("timeline.json에 events와 entities 목록이 필요합니다.")
    destination = destination.resolve()
    if destination.suffix.lower() != ".html":
        raise ValueError("출력 파일은 .html 확장자를 사용하세요.")
    video_path = resolve_video(data["video"].get("source", {}), base, video)

    def url(path: str | Path) -> str:
        # Always treat model-supplied asset names as local paths, never HTML/URLs.
        return quote(os.path.relpath((base / path).resolve(), destination.parent), safe="/")

    payload = deepcopy(data)
    payload["video_url"] = url(video_path)
    payload["video_name"] = video_path.name
    for event in payload["events"]:
        for evidence in event.get("evidence", []):
            evidence["image"] = url(evidence["image"])
    for entity in payload["entities"]:
        entity["gallery"] = [url(path) for path in entity.get("gallery", [])]
        for observation in entity.get("observations", []):
            observation["frame"] = url(observation["frame"])
            if observation.get("crop"):
                observation["crop"] = url(observation["crop"])
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    document = HTML.replace("__DATA__", encoded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".viewer-", suffix=".html", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(document)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
