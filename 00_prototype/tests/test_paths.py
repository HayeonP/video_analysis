"""Moved projects must reopen saved results without inference or JSON rewrites."""
import json
from pathlib import Path
from urllib.parse import unquote

import pytest

from video_events.first_round_viewer import generate_first_round_viewer
from video_events.paths import find_project_root, resolve_video, result_directory
from video_events.storage import file_fingerprint
from video_events.viewer import generate_viewer


@pytest.fixture
def relocated_result(tmp_path):
    original = tmp_path / "old project"
    video = original / "videos" / "원본 #1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original video bytes")
    (original / "run.py").write_text("")
    (original / "pyproject.toml").write_text("")
    base = original / "outputs" / "sample"
    base.mkdir(parents=True)
    metadata = {"source": file_fingerprint(video), "analyzed_duration_sec": 10}
    records = {
        "manifest.json": metadata,
        "candidates.json": [],
        "checkpoint.json": {"status": "complete", "registry": {}},
        "timeline.json": {"video": metadata, "events": [], "entities": []},
    }
    for name, data in records.items():
        (base / name).write_text(json.dumps(data))
    moved = tmp_path / "prototype"
    original.rename(moved)
    return moved / "outputs" / "sample", moved / "videos" / video.name


@pytest.mark.parametrize("first_round", [False, True])
@pytest.mark.parametrize("file_argument", [False, True])
def test_saved_viewers_relocate_old_absolute_video_paths(relocated_result, tmp_path, monkeypatch,
                                                       first_round, file_argument):
    base, video = relocated_result
    before = {p.name: p.read_bytes() for p in base.glob("*.json")}
    monkeypatch.chdir(tmp_path)
    renderer = generate_first_round_viewer if first_round else generate_viewer
    filename = "candidates.json" if first_round else "timeline.json"
    page = renderer(base / filename if file_argument else base)
    data = json.loads(page.read_text().split('<script type="application/json" id="data">')[1].split('</script>')[0])
    assert (page.parent / unquote(data["video_url"])).resolve() == video
    assert {p.name: p.read_bytes() for p in base.glob("*.json")} == before


def test_relocation_rejects_same_name_with_different_contents(relocated_result):
    base, video = relocated_result
    source = json.loads((base / "manifest.json").read_text())["source"]
    video.write_bytes(b"x" * video.stat().st_size)
    with pytest.raises(ValueError, match="--video"):
        resolve_video(source, base)


def test_relocation_requires_explicit_video_when_candidates_are_ambiguous(relocated_result):
    base, video = relocated_result
    source = json.loads((base / "manifest.json").read_text())["source"]
    duplicate = video.parent.parent / video.name
    duplicate.write_bytes(video.read_bytes())
    with pytest.raises(ValueError, match="여러 개"):
        resolve_video(source, base)
    assert resolve_video(source, base, video) == video
    with pytest.raises(ValueError, match="--video"):
        resolve_video(source, base, base / "missing.mp4")


def test_saved_relative_path_survives_move_with_external_video(tmp_path):
    old = tmp_path / "old"
    base = old / "results"
    base.mkdir(parents=True)
    video = old / "media" / "clips" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"source")
    source = file_fingerprint(video, relative_to=base)
    moved = tmp_path / "moved"
    old.rename(moved)
    assert resolve_video(source, moved / "results") == moved / "media/clips/clip.mp4"


@pytest.mark.parametrize("package", ["src/video_events", "video_events"])
def test_project_root_supports_source_and_flat_layouts(tmp_path, package):
    (tmp_path / "run.py").touch()
    (tmp_path / "pyproject.toml").touch()
    location = tmp_path / package
    location.mkdir(parents=True)
    assert find_project_root(location) == tmp_path


@pytest.mark.parametrize("first_round", [False, True])
@pytest.mark.parametrize("file_argument", [False, True])
def test_relative_result_from_parent_directory_uses_its_workspace(relocated_result, tmp_path, monkeypatch,
                                                                 first_round, file_argument):
    base, video = relocated_result
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("video_events.paths.PROJECT_ROOT", base.parent.parent)
    filename = "candidates.json" if first_round else "timeline.json"
    argument = Path("outputs/sample") / filename if file_argument else Path("outputs/sample")
    renderer = generate_first_round_viewer if first_round else generate_viewer
    page = renderer(argument, video=Path("videos") / video.name)
    assert page.parent == base
    assert video.name in page.read_text()


def test_existing_caller_result_takes_precedence_over_workspace(relocated_result, tmp_path, monkeypatch):
    base, _ = relocated_result
    caller_result = tmp_path / "outputs/sample"
    caller_result.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("video_events.paths.PROJECT_ROOT", base.parent.parent)
    assert result_directory(Path("outputs/sample"), "timeline.json") == caller_result
