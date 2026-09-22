"""Result-only viewer checks; no model, torch, decoder, or GPU calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote

import pytest

from video_events.viewer import generate_viewer


@pytest.fixture
def viewer_result(tmp_path):
    folder = tmp_path / "분석 결과"
    folder.mkdir()
    video = tmp_path / "원본 #1.mp4"
    video.write_bytes(b"fixture only; do not decode")
    payload = {
        "video": {"source": {"path": str(video)}, "analyzed_duration_sec": 10, "source_fps": 24},
        "events": [{"event_id": "event_1", "start_sec": 1, "end_sec": 2, "actor_id": "entity_1",
                    "description": "</script><script>window.injected=true</script>",
                    "evidence": [{"time_sec": 1, "image": "frames/한 글#1.jpg"}]}],
        "entities": [{"entity_id": "entity_1", "label": "캐릭터", "gallery": ["entities/crop.jpg"],
                      "observations": [{"time_sec": 1, "bbox": [100, 200, 300, 400],
                                        "frame": "frames/한 글#1.jpg", "crop": "entities/crop.jpg"}]}],
        "analysis_errors": [{"operation": "scan", "start_sec": 4, "end_sec": 5, "message": "실패"}],
    }
    (folder / "timeline.json").write_text(json.dumps(payload), encoding="utf-8")
    (folder / "checkpoint.json").write_text('{"status":"complete"}', encoding="utf-8")
    return folder


def embedded(page):
    return json.loads(page.read_text().split('<script type="application/json" id="data">')[1].split('</script>')[0])


def test_viewer_preserves_results_and_rebases_assets_without_injecting_html(viewer_result, tmp_path):
    before = {p.name: p.read_bytes() for p in viewer_result.iterdir()}
    output = tmp_path / "별도 화면" / "index.html"
    page = generate_viewer(viewer_result, output=output)
    data = embedded(page)
    assert data["events"][0]["description"].startswith("</script>")
    assert "</script><script>window.injected" not in page.read_text()
    assert data["analysis_errors"][0]["operation"] == "scan"
    assert data["entities"][0]["observations"][0]["bbox"] == [100, 200, 300, 400]
    paths = [(data["video_url"], tmp_path / "원본 #1.mp4"),
             (data["events"][0]["evidence"][0]["image"], viewer_result / "frames/한 글#1.jpg"),
             (data["entities"][0]["observations"][0]["frame"], viewer_result / "frames/한 글#1.jpg"),
             (data["entities"][0]["gallery"][0], viewer_result / "entities/crop.jpg")]
    for url, expected in paths:
        assert (page.parent / unquote(url)).resolve() == expected
        assert "#" not in url and " " not in url
    assert {p.name: p.read_bytes() for p in viewer_result.iterdir()} == before


def test_viewer_requires_finished_result_and_supports_video_relocation(viewer_result, tmp_path):
    checkpoint = viewer_result / "checkpoint.json"
    checkpoint.write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="아직 분석"):
        generate_viewer(viewer_result)
    checkpoint.write_text('{"status":"complete"}')
    (tmp_path / "원본 #1.mp4").unlink()
    with pytest.raises(ValueError, match="--video"):
        generate_viewer(viewer_result)
    moved = tmp_path / "new.mp4"
    moved.write_bytes(b"fixture")
    page = generate_viewer(viewer_result / "timeline.json", video=moved)
    assert page.name == "viewer.html"
    assert embedded(page)["video_url"] == "../new.mp4"
    with pytest.raises(ValueError, match=".html"):
        generate_viewer(viewer_result, output=viewer_result / "timeline.json")


def test_viewer_accepts_empty_final_results(viewer_result):
    source = viewer_result / "timeline.json"
    data = json.loads(source.read_text())
    data.update(events=[], entities=[])
    source.write_text(json.dumps(data))
    assert embedded(generate_viewer(viewer_result))["events"] == []


def test_viewer_cli_uses_project_venv_and_never_imports_gpu_stack(viewer_result, tmp_path):
    root = Path(__file__).resolve().parents[1]
    # sitecustomize runs both before and after run.py re-execs into the venv.
    # Forbid imports entirely, stronger than merely hiding CUDA devices.
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text('''
import sys
class DenyAnalysis:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'bitsandbytes', 'av'} or fullname in {'video_events.pipeline', 'video_events.backend'}:
            raise RuntimeError('Viewer attempted an analysis import: ' + fullname)
sys.meta_path.insert(0, DenyAnalysis())
''')
    env = dict(os.environ, PYTHONPATH=str(guard), VIDEO_EVENTS_PYTHON="/usr/bin/python3")
    command = [str(root / "visualize.sh"), str(viewer_result)]
    process = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert (viewer_result / "viewer.html").exists()
    assert "시각화 생성 완료" in process.stdout
    # Same import guard covers the standalone Python API.
    process = subprocess.run([sys.executable, "-c", "from video_events.viewer import generate_viewer"],
                             cwd=tmp_path, env=env, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
