from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import logging
from pathlib import Path
import time

from .backend import QwenBackend
from .config import Settings
from .events import global_events, merge_events, refinement_windows
from .export import export_results
from .identity import EntityRegistry
from .schema import Event, Observation, ScanResult
from .storage import atomic_json, file_fingerprint, read_json
from .video import VideoSource, sliding_windows

LOG = logging.getLogger(__name__)


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("같은 출력 폴더에서 다른 분석이 실행 중입니다.") from exc
        yield


def analyze(video: Path, output: Path, settings: Settings, *, limit_seconds: float | None = None,
            backend=None, fail_fast: bool = False) -> dict:
    video, output = video.resolve(), output.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"영상 파일을 찾지 못했습니다: {video}")
    source = VideoSource(video, settings.max_pixels)
    duration = min(source.duration, limit_seconds) if limit_seconds is not None else source.duration
    if duration <= 0:
        raise ValueError("분석 길이는 0보다 커야 합니다.")
    manifest = {"schema_version": 1, "pipeline_version": 4, "source": file_fingerprint(video), "settings": settings.model_dump(),
                "analyzed_duration_sec": duration}
    backend = backend or QwenBackend(settings, output / "model_calls")
    with output_lock(output):
        manifest_path, checkpoint = output / "manifest.json", output / "checkpoint.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise ValueError("입력 영상 또는 설정이 기존 실행과 다릅니다. --output으로 새 폴더를 지정하세요.")
        else:
            if checkpoint.exists() or (output / "timeline.json").exists():
                raise ValueError("기존 결과의 manifest.json이 없습니다. 새 출력 폴더를 지정하세요.")
            atomic_json(manifest_path, manifest)
        state = read_json(checkpoint) if checkpoint.exists() else {
            "coarse_index": 0, "refined_index": 0, "coarse_events": [], "refined_events": [],
            "scene_cuts_sec": [], "registry": {}, "status": "running",
        }
        state.setdefault("analysis_errors", [])
        if state["status"] == "complete":
            LOG.info("이미 완료된 결과: %s", output / "timeline.json")
            return read_json(output / "timeline.json")
        registry = EntityRegistry(output, settings, state["registry"])
        started = time.monotonic()

        def save():
            state["registry"] = registry.dump()
            atomic_json(checkpoint, state)
            atomic_json(output / "analysis_errors.json", state["analysis_errors"])

        def record_failure(exc: Exception, *, stage: str, window_id: str | None, left: float,
                           right: float, operation: str, recovery: str) -> None:
            # Never swallow KeyboardInterrupt/SystemExit. These are not Exceptions.
            if fail_fast:
                raise exc
            record = {"stage": stage, "window_id": window_id, "start_sec": left, "end_sec": right,
                      "operation": operation, "error_type": type(exc).__name__,
                      "message": str(exc), "recovery": recovery}
            state["analysis_errors"].append(record)
            atomic_json(output / "analysis_errors.json", state["analysis_errors"])
            LOG.warning("오류 후 계속 · %s · %s · %s: %s", window_id or stage, operation, recovery, exc)
            # Exception tracebacks can retain CUDA tensors from a failed generation.
            # Clear them before collecting garbage/emptying the allocator cache.
            exc.__traceback__ = None
            recover = getattr(backend, "recover", None)
            if recover is not None:
                try:
                    recover()
                except Exception as cleanup_error:
                    LOG.warning("모델 메모리 정리 실패, 다음 구간 시도: %s", cleanup_error)

        def process(stage: str, index: int, left: float, right: float, fps: float, hints: list[dict]):
            nonlocal registry
            window_id = f"{stage}_{index:06d}"
            before_registry = deepcopy(registry.dump())
            operation = "sample"

            def failed(exc: Exception, recovery: str):
                record_failure(exc, stage=stage, window_id=window_id, left=left, right=right,
                               operation=operation, recovery=recovery)

            try:
                clip = source.sample(left, right, fps)
                clip.save_frames(output)
                operation = "scan"
                cache = output / "windows" / f"{window_id}.json"
                recent = [{key: event[key] for key in ("start_sec", "end_sec", "description", "state_after")}
                          for event in state["coarse_events"]
                          if event["end_sec"] >= left - settings.window_sec and event["start_sec"] < left][-4:]
                if stage == "refined":
                    recent = []  # The focused candidate hints already contain the relevant context.
                if cache.exists():
                    result = ScanResult.model_validate(read_json(cache)["result"]).validate_clip(clip.duration, len(clip.frames))
                else:
                    result = backend.scan(clip, stage=stage, context={"entities": registry.context(left, right),
                                          "recent_events_absolute_seconds": recent}, hints=hints)
                    atomic_json(cache, {"window_id": window_id, "start_sec": left, "end_sec": right,
                                        "target_fps": fps,
                                        "frames": [{"time_sec": frame.time_sec, "image": frame.path} for frame in clip.frames],
                                        "result": result.model_dump()})
                operation = "localize"
                localization_cache = output / "windows" / f"{window_id}.entities.json"
                try:
                    if localization_cache.exists():
                        localized = [Observation.model_validate(item) for item in read_json(localization_cache)]
                    else:
                        localized = backend.localize(clip, result.entities)
                        atomic_json(localization_cache, [item.model_dump() for item in localized])
                except Exception as exc:
                    failed(exc, "unresolved_identity")
                    localized = []
                operation = "identity"
                try:
                    mapping = registry.resolve(localized, clip, backend)
                except Exception as exc:
                    # resolve() can have registered some IDs before a later comparison
                    # failed. Roll back that incomplete transaction, keep the events.
                    registry = EntityRegistry(output, settings, deepcopy(before_registry))
                    failed(exc, "unresolved_identity")
                    mapping = {}
                operation = "events"
                events = global_events(result, clip, mapping, window_id, stage)
                state["scene_cuts_sec"].extend(left + value for value in result.scene_cuts_sec)
                return events
            except Exception as exc:
                registry = EntityRegistry(output, settings, before_registry)
                failed(exc, "skipped_window")
                return []

        try:
            windows = list(sliding_windows(duration, settings.window_sec, settings.stride_sec))
            for index in range(state["coarse_index"], len(windows)):
                left, right = windows[index]
                LOG.info("후보 추출 %d/%d · %.2f~%.2f초 · %.1ffps", index + 1, len(windows), left, right, settings.coarse_fps)
                events = process("coarse", index, left, right, settings.coarse_fps, [])
                state["coarse_events"].extend(event.model_dump() for event in events)
                state["coarse_index"] = index + 1
                save()
            candidates = merge_events([Event.model_validate(event) for event in state["coarse_events"]], settings.merge_tolerance_sec)
            atomic_json(output / "candidates.json", [event.model_dump() for event in candidates])
            selected = refinement_windows(candidates, duration, settings)
            atomic_json(output / "refinement_plan.json", [{"start_sec": left, "end_sec": right} for left, right in selected])
            for index in range(state["refined_index"], len(selected)):
                left, right = selected[index]
                LOG.info("정밀 분석 %d/%d · %.2f~%.2f초 · %.1ffps", index + 1, len(selected), left, right, settings.refine_fps)
                hints = []
                for candidate in candidates:
                    if candidate.start_sec <= right and candidate.end_sec >= left:
                        hint = candidate.model_dump(include={"action", "description", "actor_id", "object_id"})
                        hint.update(start_sec=max(0, candidate.start_sec - left), end_sec=min(right - left, candidate.end_sec - left))
                        hints.append(hint)
                events = process("refined", index, left, right, settings.refine_fps, hints)
                state["refined_events"].extend(event.model_dump() for event in events)
                state["refined_index"] = index + 1
                save()
            LOG.info("대상 ID 연결 및 전체 타임라인 정리")
            before_reconcile = deepcopy(registry.dump())
            try:
                registry.reconcile(backend)
            except Exception as exc:
                registry = EntityRegistry(output, settings, before_reconcile)
                record_failure(exc, stage="final", window_id=None, left=0, right=duration,
                               operation="reconcile", recovery="kept_existing_ids")
            final = [Event.model_validate(event) for event in state["refined_events"]]
            failed_refinements = [failure for failure in state["analysis_errors"]
                                  if failure["stage"] == "refined" and failure["recovery"] == "skipped_window"]
            # A failed detailed pass must not erase a coarse candidate. These remain
            # explicitly coarse/uncertain and are not merged into refined results.
            for candidate in candidates:
                if any((candidate.start_sec < failure["end_sec"] and candidate.end_sec > failure["start_sec"])
                       or (candidate.start_sec == candidate.end_sec and
                           failure["start_sec"] <= candidate.start_sec < failure["end_sec"])
                       for failure in failed_refinements):
                    final.append(candidate.model_copy(deep=True, update={"uncertain": True,
                                                                        "review_reason": "refinement_failed"}))
            for event in final:
                event.actor_id, event.object_id = registry.canonical(event.actor_id), registry.canonical(event.object_id)
            final = merge_events(final, settings.merge_tolerance_sec)
            metadata = {**manifest, "duration_sec": source.duration, "width": source.width, "height": source.height,
                        "source_fps": source.fps, "coarse_windows": len(windows), "refined_windows": len(selected),
                        "error_count": len(state["analysis_errors"]),
                        "analysis_status": "complete_with_errors" if state["analysis_errors"] else "complete"}
            export_results(output, video, metadata, final, registry.export(), state["scene_cuts_sec"], state["analysis_errors"])
            state["status"] = "complete"
            state.pop("error", None)
            save()
            (output / "last_error.json").unlink(missing_ok=True)
            LOG.info("완료: 이벤트 %d개 / 대상 %d개 / 오류 %d건 / 이번 실행 %.1f초 / %s", len(final), len(registry.export()),
                     len(state["analysis_errors"]), time.monotonic() - started, output)
            return read_json(output / "timeline.json")
        except BaseException as exc:
            # Never commit a partially resolved window: last on-disk checkpoint is the restart boundary.
            atomic_json(output / "last_error.json", {"type": type(exc).__name__, "message": str(exc),
                                                    "resume": "같은 명령을 다시 실행하면 마지막 완료 구간부터 이어집니다."})
            raise
