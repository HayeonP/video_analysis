from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from video_events.backend import ModelOutputError, parse_json_response, video_processor_arguments
from video_events.config import Settings
from video_events.demo import make_demo
from video_events.events import merge_events, meaningful, refinement_windows
from video_events.identity import EntityRegistry
from video_events.pipeline import analyze
from video_events.schema import Event, IdentityDecision, ScanResult
from video_events.video import VideoSource, sliding_windows


def event(start=1.0, end=2.0, source="coarse_0", **kwargs):
    values = dict(event_type="interaction", start_sec=start, end_sec=end, actor_id="entity_0001",
                  object_id="entity_0002", action="pick_up", description="캐릭터가 상자를 집어 든다",
                  state_before="상자가 탁자에 있음", state_after="상자를 들고 있음", uncertain=False,
                  continues_before=False, continues_after=False, evidence=[], source_windows=[source], stage="coarse")
    return Event(**(values | kwargs))


def test_windows_cover_tail_and_short_video():
    assert list(sliding_windows(6.4, 5, 1)) == [(0, 5), (1, 6), (2, 6.4)]
    assert list(sliding_windows(0.2, 5, 1)) == [(0, 0.2)]
    assert list(sliding_windows(5, 5, 1)) == [(0, 5)]
    with pytest.raises(ValueError):
        list(sliding_windows(10, 5, 6))


def test_no_event_loss_from_confidence_filter():
    assert meaningful(event(uncertain=True))
    assert meaningful(event(event_type="appear"))
    assert not meaningful(event(event_type="ongoing", state_before="정지", state_after="정지"))


def test_merge_duplicates_but_preserve_repeated_actions():
    first, duplicate = event(), event(1.1, 2.1, "coarse_1")
    repeated = event(3, 4, "coarse_1")
    result = merge_events([first, duplicate, repeated], 0.35)
    assert len(result) == 2
    assert result[0].source_windows == ["coarse_0", "coarse_1"]
    assert len(merge_events([event(), event(1.2, 2.2)], 0.35)) == 2


def test_refinement_regions_cover_candidate_boundaries_without_duplicate_windows():
    windows = refinement_windows([event(0, 7), event(5, 9)], 10, Settings())
    assert windows[0][0] == 0 and windows[-1][1] == 10
    assert len(windows) == len(set(windows))
    assert all(end - start <= 5 for start, end in windows)
    assert all(a[1] >= b[0] for a, b in zip(windows, windows[1:]))


def test_model_json_must_be_complete_and_grounded():
    assert parse_json_response('```json\n{"entities":[],"events":[]}\n```')["events"] == []
    with pytest.raises(ValueError):
        parse_json_response('{"entities": [{"local_id":"e1"}')
    candidate = dict(event_type="action", start_sec=0, end_sec=1, action="walk", description="걷는다",
                     evidence_frames=[0], actor_id="missing")
    with pytest.raises(ValueError):
        ScanResult.model_validate({"entities": [], "events": [candidate]})


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "space in name.mp4"
    make_demo(path, duration=6.4, fps=10)
    return path


def test_sampling_preserves_source_time_and_fractional_fps(video):
    source = VideoSource(video, 65536)
    clip = source.sample(1, 6, 1.5)
    assert len(clip.frames) == 8
    times = [frame.time_sec for frame in clip.frames]
    assert times == sorted(set(times))
    assert all(1 <= time < 6 for time in times)
    assert all(abs(index / source.fps - (time - 1)) < 1e-8
               for index, time in zip(clip.metadata()["frames_indices"], times))
    assert len(source.sample(0, 5, 4).frames) == 20


def test_qwen_processor_receives_real_video_timestamps(video):
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor, Qwen3VLProcessorKwargs
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    clip = VideoSource(video, 65536).sample(1, 6, 1.5)
    arguments = video_processor_arguments(clip, Settings(max_pixels=65536))
    processor = object.__new__(Qwen3VLProcessor)
    options = processor._merge_kwargs(Qwen3VLProcessorKwargs, **arguments)
    data = Qwen3VLVideoProcessor()(arguments["videos"], **options["videos_kwargs"], return_tensors="pt")
    metadata = data["video_metadata"][0]
    assert metadata.fps == clip.source_fps
    assert metadata.frames_indices == clip.metadata()["frames_indices"]
    timestamps = processor._calculate_timestamps(metadata.frames_indices, metadata.fps, 2)
    assert timestamps[0] == pytest.approx((clip.frames[0].time_sec + clip.frames[1].time_sec) / 2 - 1)
    assert timestamps[-1] > 4  # Would be ~0.27 seconds with the library's fps=24 fallback.


class FakeBackend:
    """Deterministic test double; production CLI always uses the real Qwen model."""
    def __init__(self, fail_at: int | None = None):
        self.calls = 0
        self.fail_at = fail_at
        self.identity_calls = 0

    def scan(self, clip, *, stage, context, hints):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated interruption")
        result = {
            "entities": [{"local_id": "e1", "kind": "character", "label": "붉은 옷 캐릭터",
                          "appearance": "붉은 옷과 검은 머리", "frame_index": 0, "bbox": [300, 100, 550, 950]}],
            "events": [{"event_type": "action", "start_sec": max(0, 2 - clip.start_sec),
                        "end_sec": min(clip.duration, 4 - clip.start_sec), "actor_id": "e1",
                        "action": "reach", "description": "캐릭터가 손을 뻗는다", "evidence_frames": [0],
                        "continues_before": clip.start_sec > 2, "continues_after": clip.end_sec < 4}],
        }
        if clip.start_sec >= 4 or clip.end_sec <= 2:
            result["events"] = []
        return ScanResult.model_validate(result)

    def identify(self, query, description, references):
        self.identity_calls += 1
        return IdentityDecision(match_id=references[0]["entity_id"], possible_ids=[], reason="같은 캐릭터")

    def localize(self, clip, observations):
        return observations


def test_end_to_end_resume_keeps_global_identity_and_valid_exports(video, tmp_path):
    out = tmp_path / "result"
    broken = FakeBackend(fail_at=2)
    with pytest.raises(RuntimeError, match="interruption"):
        analyze(video, out, Settings(max_pixels=65536), backend=broken, fail_fast=True)
    checkpoint = json.loads((out / "checkpoint.json").read_text())
    assert checkpoint["coarse_index"] == 1
    resumed = FakeBackend()
    result = analyze(video, out, Settings(max_pixels=65536), backend=resumed)
    assert len(result["entities"]) == 1
    assert result["entities"][0]["kind"] == "character"
    assert {row["actor_id"] for row in result["events"]} == {"entity_0001"}
    assert result["video"]["coarse_windows"] == 3
    assert result["events"][0]["stage"] == "refined"
    assert (out / "timeline.html").exists() and (out / "events.csv").exists()
    assert all((out / evidence["image"]).exists() for row in result["events"] for evidence in row["evidence"])
    no_calls = FakeBackend(fail_at=1)
    assert analyze(video, out, Settings(max_pixels=65536), backend=no_calls) == result
    assert no_calls.calls == 0
    with pytest.raises(ValueError, match="설정"):
        analyze(video, out, Settings(coarse_fps=2, max_pixels=65536), backend=FakeBackend())


def test_launch_from_other_directory_uses_project_venv(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(["/usr/bin/python3", str(root / "run.py"), "doctor"], cwd=tmp_path,
                               text=True, capture_output=True)
    # A CPU-only test machine can return 1 while still proving the automatic re-exec.
    assert completed.returncode in (0, 1), completed.stderr
    report = json.loads(completed.stdout)
    assert report["project_environment_active"] is True
    assert report["python"] == str(root / ".venv/bin/python")


def test_resume_after_localization_failure_keeps_cached_scan_and_commits_next_window(video, tmp_path):
    class BrokenLocalization(FakeBackend):
        def localize(self, clip, observations):
            raise RuntimeError("Return exactly one box per requested local_id")

    class UnlocalizedBackend(FakeBackend):
        def localize(self, clip, observations):
            return []

    out = tmp_path / "resume_localization"
    settings = Settings(max_pixels=65536)
    with pytest.raises(RuntimeError, match="exactly one box"):
        analyze(video, out, settings, backend=BrokenLocalization(), fail_fast=True)
    assert (out / "windows/coarse_000000.json").exists()
    assert not (out / "windows/coarse_000000.entities.json").exists()
    before = (out / "windows/coarse_000000.json").read_bytes()
    backend = UnlocalizedBackend()
    result = analyze(video, out, settings, backend=backend)
    assert (out / "windows/coarse_000000.json").read_bytes() == before
    checkpoint = json.loads((out / "checkpoint.json").read_text())
    assert checkpoint["status"] == "complete" and checkpoint["coarse_index"] == 3
    assert result["events"]
    assert all(row["uncertain"] and row["identity_status"] == "unresolved" for row in result["events"])
    assert all(row["unresolved_local_ids"] == ["e1"] for row in result["events"])
    assert result["entities"] == [] and backend.identity_calls == 0
    assert backend.calls == result["video"]["coarse_windows"] - 1 + result["video"]["refined_windows"]


@pytest.mark.parametrize("failure", [ModelOutputError("missing action"), RuntimeError("CUDA out of memory")])
def test_failed_window_is_recorded_and_later_windows_still_finish(video, tmp_path, failure):
    class FailingWindow(FakeBackend):
        recovered = 0

        def scan(self, clip, **kwargs):
            if self.calls == 1:
                self.calls += 1
                raise failure
            return super().scan(clip, **kwargs)

        def recover(self):
            self.recovered += 1

    backend = FailingWindow()
    out = tmp_path / "continued"
    result = analyze(video, out, Settings(max_pixels=65536), backend=backend)
    assert result["video"]["analysis_status"] == "complete_with_errors"
    assert result["video"]["error_count"] == 1
    assert result["events"] and backend.recovered == 1
    failure_record = result["analysis_errors"][0]
    assert failure_record["window_id"] == "coarse_000001"
    assert failure_record["operation"] == "scan" and failure_record["recovery"] == "skipped_window"
    state = json.loads((out / "checkpoint.json").read_text())
    assert state["coarse_index"] == 3 and state["status"] == "complete"
    assert state["analysis_errors"] == json.loads((out / "analysis_errors.json").read_text())
    assert not (out / "last_error.json").exists()
    assert (out / "windows/coarse_000002.json").exists()


def test_every_scan_can_fail_without_losing_error_report_or_exports(video, tmp_path):
    class AlwaysFails(FakeBackend):
        def scan(self, *args, **kwargs):
            raise ModelOutputError("invalid JSON")

    out = tmp_path / "all_failed"
    result = analyze(video, out, Settings(max_pixels=65536), backend=AlwaysFails())
    assert result["events"] == [] and result["video"]["refined_windows"] == 0
    assert result["video"]["error_count"] == 3
    assert {item["window_id"] for item in result["analysis_errors"]} == {
        "coarse_000000", "coarse_000001", "coarse_000002"}
    assert result["video"]["analysis_status"] == "complete_with_errors"
    assert all((out / name).exists() for name in ("timeline.json", "timeline.html", "events.csv", "entities.json"))
    assert json.loads((out / "checkpoint.json").read_text())["coarse_index"] == 3


def test_refinement_failure_preserves_coarse_candidate_with_explicit_status(video, tmp_path):
    class RefinementFails(FakeBackend):
        def scan(self, clip, *, stage, **kwargs):
            if stage == "refined":
                raise ModelOutputError("refinement JSON invalid")
            return super().scan(clip, stage=stage, **kwargs)

    out = tmp_path / "refinement_failed"
    result = analyze(video, out, Settings(max_pixels=65536), backend=RefinementFails())
    assert result["events"]
    assert all(row["stage"] == "coarse" and row["uncertain"] for row in result["events"])
    assert all(row["review_reason"] == "refinement_failed" for row in result["events"])
    assert all(row["actor_id"] == "entity_0001" for row in result["events"])
    assert result["video"]["error_count"] == result["video"]["refined_windows"]
    assert json.loads((out / "checkpoint.json").read_text())["refined_index"] == result["video"]["refined_windows"]
    # Never merge a coarse fallback into a refined event and erase its provenance.
    assert len(merge_events([event(review_reason="refinement_failed", uncertain=True),
                            event(source="refined_0", stage="refined")], 0.35)) == 2


def test_localization_runtime_failure_preserves_events_and_continues(video, tmp_path):
    class LocalizationFails(FakeBackend):
        def localize(self, *args):
            raise RuntimeError("localization unavailable")

    result = analyze(video, tmp_path / "localization_failed", Settings(max_pixels=65536), backend=LocalizationFails())
    assert result["events"] and result["entities"] == []
    assert all(row["identity_status"] == "unresolved" and row["uncertain"] for row in result["events"])
    assert all(item["operation"] == "localize" and item["recovery"] == "unresolved_identity"
               for item in result["analysis_errors"])


def test_identity_failure_rolls_back_partial_registry_but_keeps_events(video, tmp_path, monkeypatch):
    original = EntityRegistry.resolve

    def partial_failure(self, *args):
        original(self, *args)
        raise RuntimeError("identity comparison failed after registering a target")

    monkeypatch.setattr(EntityRegistry, "resolve", partial_failure)
    out = tmp_path / "identity_failed"
    result = analyze(video, out, Settings(max_pixels=65536), backend=FakeBackend())
    assert result["events"] and result["entities"] == []
    assert all(row["actor_id"] is None and row["uncertain"] for row in result["events"])
    assert all(item["operation"] == "identity" for item in result["analysis_errors"])
    registry = json.loads((out / "checkpoint.json").read_text())["registry"]
    assert registry["next_id"] == 1 and registry["entries"] == {} and registry["distinct_pairs"] == []


def test_failed_final_reconciliation_restores_ids_and_still_exports(video, tmp_path, monkeypatch):
    def partial_failure(self, backend):
        self.aliases["entity_0001"] = "bad_id"
        raise RuntimeError("final comparison failed")

    monkeypatch.setattr(EntityRegistry, "reconcile", partial_failure)
    result = analyze(video, tmp_path / "reconcile_failed", Settings(max_pixels=65536), backend=FakeBackend())
    assert result["events"] and result["entities"][0]["aliases"] == []
    assert all(row["actor_id"] == "entity_0001" for row in result["events"])
    assert result["analysis_errors"][0]["operation"] == "reconcile"
    assert result["analysis_errors"][0]["recovery"] == "kept_existing_ids"


def test_decode_failure_skips_only_affected_window(video, tmp_path, monkeypatch):
    original = VideoSource.sample

    def damaged_frame(self, start, end, fps):
        if start == 1 and fps == 1.5:
            raise ValueError("damaged video packet")
        return original(self, start, end, fps)

    monkeypatch.setattr(VideoSource, "sample", damaged_frame)
    result = analyze(video, tmp_path / "decode_failed", Settings(max_pixels=65536), backend=FakeBackend())
    assert result["events"]
    assert len(result["analysis_errors"]) == 1
    assert result["analysis_errors"][0]["operation"] == "sample"
    assert result["analysis_errors"][0]["window_id"] == "coarse_000001"


def test_skipped_window_is_checkpointed_and_keyboard_interrupt_still_stops(video, tmp_path):
    class SkipThenInterrupt(FakeBackend):
        def scan(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ModelOutputError("first window failed")
            raise KeyboardInterrupt()

    out = tmp_path / "skip_then_interrupt"
    settings = Settings(max_pixels=65536)
    with pytest.raises(KeyboardInterrupt):
        analyze(video, out, settings, backend=SkipThenInterrupt())
    state = json.loads((out / "checkpoint.json").read_text())
    assert state["coarse_index"] == 1 and len(state["analysis_errors"]) == 1
    resumed = FakeBackend()
    result = analyze(video, out, settings, backend=resumed)
    assert result["video"]["error_count"] == 1
    assert resumed.calls == 2 + result["video"]["refined_windows"]
    assert not (out / "windows/coarse_000000.json").exists()  # The skipped window wasn't retried.
    assert not (out / "last_error.json").exists()
