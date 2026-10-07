"""First-pass provenance checks: no final aliases, refined boxes, or inference."""
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess

import pytest

from video_events.first_round_viewer import first_round_payload, generate_first_round_viewer
from video_events.viewer_server import RangeRequestHandler, serve


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def first_round_result(tmp_path):
    base = tmp_path / "분석 결과"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"never decoded")
    obs = {"frame": "frames/first.jpg", "time_sec": 1, "bbox": [100, 200, 300, 400],
           "crop": "entities/entity_0001/first.jpg"}
    refined = {"frame": "frames/refined.jpg", "time_sec": 1.25, "bbox": [120, 210, 320, 430],
               "crop": "entities/entity_0002/refined.jpg"}
    candidates = [{"event_id": "event_1", "stage": "coarse", "start_sec": 1, "end_sec": 2,
                   "actor_id": "entity_0001", "object_id": None, "description": "1차 설명", "action": "walk",
                   "evidence": [{"time_sec": 1, "image": "frames/first.jpg"}], "uncertain": False}]
    save(base / "candidates.json", candidates)
    save(base / "manifest.json", {"source": {"path": str(video)}, "analyzed_duration_sec": 10})
    save(base / "checkpoint.json", {"status": "running", "registry": {
        "aliases": {"entity_0001": "entity_0002"}, "entries": {
            "entity_0001": {"entity_id": "entity_0001", "label": "원래 캐릭터", "kind": "character",
                            "appearance": "붉은 옷", "observations": [obs]},
            "entity_0002": {"entity_id": "entity_0002", "label": "최종 대상", "kind": "character",
                            "appearance": "다른 설명", "observations": [refined, deepcopy(obs)]},
        }}, "analysis_errors": [
            {"stage": "coarse", "operation": "scan", "start_sec": 2, "end_sec": 7},
            {"stage": "refined", "operation": "scan", "start_sec": 5, "end_sec": 10},
        ]})
    save(base / "windows/coarse_000000.json", {"start_sec": 1, "end_sec": 6,
         "frames": [{"time_sec": 1, "image": "frames/first.jpg"}],
         "result": {"entities": [], "events": [], "scene_cuts_sec": [2]}})
    save(base / "windows/coarse_000000.entities.json", [
        {"frame_index": 0, "local_id": "local_1", "bbox": obs["bbox"]}])
    save(base / "windows/refined_000000.json", {"start_sec": 1, "end_sec": 6,
         "frames": [{"time_sec": 1.25, "image": "frames/refined.jpg"}],
         "result": {"entities": [], "events": [], "scene_cuts_sec": [3]}})
    save(base / "windows/refined_000000.entities.json", [
        {"frame_index": 0, "local_id": "local_1", "bbox": refined["bbox"]}])
    return base


def test_first_round_retains_original_ids_without_final_timeline_or_aliases(first_round_result):
    base = first_round_result
    assert not (base / "timeline.json").exists()
    before = {str(path): path.read_bytes() for path in base.rglob("*.json")}
    data = first_round_payload(base)
    event = data["events"][0]
    assert (event["start_sec"], event["end_sec"], event["actor_id"], event["description"]) == (1, 2, "entity_0001", "1차 설명")
    assert event["uncertain"] is False  # Preserve original predictions; label the page separately.
    assert data["view_mode"] == "first_round"
    assert data["scene_cuts_sec"] == [3]
    assert len(data["analysis_errors"]) == 1 and data["analysis_errors"][0]["stage"] == "coarse"
    assert [e["entity_id"] for e in data["entities"]] == ["entity_0001"]
    entity = data["entities"][0]
    assert entity["aliases"] == [] and entity["identity_status"] == "unverified"
    assert len(entity["observations"]) == 1 and entity["observations"][0]["time_sec"] == 1
    assert entity["gallery"] == ["entities/entity_0001/first.jpg"]
    page = generate_first_round_viewer(base)
    assert page.name == "first_round.html"
    assert "1차 분석 후보" in page.read_text()
    assert {str(path): path.read_bytes() for path in base.rglob("*.json")} == before


def test_first_round_does_not_invent_boxes_when_localization_is_missing(first_round_result):
    (first_round_result / "windows/coarse_000000.entities.json").unlink()
    data = first_round_payload(first_round_result)
    assert len(data["events"]) == 1
    assert data["entities"][0]["entity_id"] == "entity_0001"
    assert data["entities"][0]["observations"] == []
    assert data["entities"][0]["gallery"] == []


def test_first_round_script_uses_saved_logs_without_analysis_imports(first_round_result, tmp_path):
    root = Path(__file__).resolve().parents[1]
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text('''
import sys
class DenyAnalysis:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'bitsandbytes', 'av', 'PIL'} or fullname in {'video_events.pipeline', 'video_events.backend', 'video_events.identity', 'video_events.video'}:
            raise RuntimeError('Inference/decode import in viewer: ' + fullname)
sys.meta_path.insert(0, DenyAnalysis())
''')
    env = dict(os.environ, PYTHONPATH=str(guard), VIDEO_EVENTS_PYTHON="/usr/bin/python3")
    run = subprocess.run([str(root / "first_round_visualize.sh"), str(first_round_result), "--generate-only"],
                         cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10)
    assert run.returncode == 0, run.stderr
    assert (first_round_result / "first_round.html").exists()


def test_first_round_server_root_targets_first_round_without_starting_listener(tmp_path, monkeypatch):
    page = tmp_path / "output space" / "first_round.html"
    page.parent.mkdir()
    page.write_text("first round")
    captured = {}

    class FakeServer:
        server_port = 5001
        def __init__(self, address, handler):
            captured.update(address=address, handler=handler)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def serve_forever(self): raise KeyboardInterrupt

    monkeypatch.setattr("video_events.viewer_server.ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr("video_events.viewer_server.lan_address", lambda: "192.168.0.106")
    serve(tmp_path, 5001, home=page)
    assert captured["address"] == ("0.0.0.0", 5001)
    home = captured["handler"].keywords["home_page"]

    class Connection:
        def __init__(self): self.output = bytearray()
        def makefile(self, *args): return BytesIO(b"GET / HTTP/1.0\r\n\r\n")
        def sendall(self, data): self.output.extend(data)

    class QuietHandler(RangeRequestHandler):
        def log_message(self, *args): pass

    connection = Connection()
    QuietHandler(connection, ("127.0.0.1", 0), object(), directory=tmp_path, home_page=home)
    assert b"Location: /output%20space/first_round.html" in connection.output
