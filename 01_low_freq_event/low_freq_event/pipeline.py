from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import logging
from pathlib import Path
import signal
import threading

from .backend import QwenBackend
from .config import PIPELINE_VERSION
from .identity import canonical, crop_observations, resolve_frame
from .schema import EntityList, LocalEntity, ScanResult
from .storage import atomic_json, file_fingerprint, read_json
from .timing import RunTimer
from .video import VideoSource, sliding_windows, prepare_samples, sampled_clip
from .detection import WindowPerception
from .context import entity_context, context_prompt

LOG = logging.getLogger(__name__)


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".analysis.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"이미 분석 중인 결과 폴더입니다: {output}") from None
        yield


def code_fingerprint():
    digest = hashlib.sha256()
    for name in ("config.py", "schema.py", "prompts.py", "video.py", "backend.py", "identity.py", "pipeline.py", "perception.py", "temporal.py", "reanalyze.py", "detection.py", "context.py", "motion.py", "botsort.py", "person_linking.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def window_events(scan, clip, observations, assignments, window_id, registry=None):
    events = []
    by_local = {entity.local_id: entity for entity in scan.entities}
    for index, candidate in enumerate(scan.events):
        indices = list(range(candidate.start_frame, candidate.end_frame + 1))
        participating = []
        uncertain = candidate.uncertain
        for local_id in candidate.entity_ids:
            obs = [o for o in observations if o["local_id"] == local_id and o["frame_index"] in indices]
            if any(o.get('identity_uncertain') for o in obs):
                uncertain = True
            ids = sorted({assignments[o["observation_id"]] for o in obs})
            if len(ids) != 1 or any((registry or {}).get(eid, {}).get("possible_matches") for eid in ids):
                uncertain = True
            participating.append({"local_id": local_id, "label": by_local[local_id].label,
                                  "entity_ids": ids, "observation_ids": [o["observation_id"] for o in obs],
                                  "localization_status": "located" if obs else "not_located"})
        start = clip.start_sec if candidate.continues_before else clip.frames[candidate.start_frame].time_sec
        end = clip.end_sec if candidate.continues_after else clip.frames[candidate.end_frame].time_sec
        temporal_evidence = []
        for fact in candidate.temporal_evidence:
            local_id = fact['entity_id']
            global_ids = sorted({assignments[o['observation_id']] for o in observations
                                 if o['local_id'] == local_id and o['frame_index'] == fact['frame_index']})
            for eid in global_ids:
                temporal_evidence.append(dict(fact, entity_id=eid, local_id=local_id))
        events.append({**candidate.model_dump(), "event_id": f"{window_id}_event_{index:03d}",
                       "window_id": window_id, "start_sec": round(start, 6), "end_sec": round(end, 6),
                       "duration_sec": round(end - start, 6), "uncertain": uncertain,
                       "entity_ids": sorted({eid for p in participating for eid in p["entity_ids"]}),
                       "participants": participating, "frame_indices": indices,
                       "temporal_evidence": temporal_evidence,
                       "evidence": [{"frame_index": i, "time_sec": clip.frames[i].time_sec,
                                     "image": clip.frames[i].path} for i in indices],
                       "time_basis": candidate.temporal_basis if candidate.temporal_basis != 'legacy_interval'
                       else "observed_sample_span_with_censored_clip_boundaries"})
    return events


def normalized_ids(value, registry):
    if isinstance(value, list):
        return [normalized_ids(item, registry) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "entity_id":
            result[key] = canonical(registry, item)
        elif key in {"entity_ids", "possible_matches"}:
            result[key] = sorted({canonical(registry, eid) for eid in item})
        else:
            result[key] = normalized_ids(item, registry)
    return result


def export(output, manifest, state, timing=None):
    if timing is None and (output / "timing.json").exists():
        timing = read_json(output / "timing.json")
    registry = state["registry"]
    entities = [normalized_ids(e, registry) for e in registry.values() if not e.get("alias_of")]
    for entity in entities:
        entity["possible_matches"] = [eid for eid in entity["possible_matches"] if eid != entity["entity_id"]]
    payload = {"schema_version": 1, "video": manifest, "status": state["status"],
               "windows": normalized_ids(state["windows"], registry), "events": normalized_ids(state["events"], registry),
               "entities": entities, "timing": timing or {},
               "entity_aliases": {eid: canonical(registry, eid) for eid in registry if registry[eid].get("alias_of")},
               "tracks": normalized_ids(state.get("tracks", []), registry),
               "errors": state.get("errors", []),
               "event_policy": "Window-level events are preserved; overlapping windows are not deduplicated."}
    atomic_json(output / "timeline.json", payload)
    atomic_json(output / "events.json", payload["events"])
    atomic_json(output / "entities.json", payload["entities"])
    from .viewer import generate_viewer
    generate_viewer(output)
    return payload


def make_perception(settings, output):
    if settings.tracking_backend == 'botsort':
        from .botsort import BoTSORTPerception
        return BoTSORTPerception(settings, output)
    return WindowPerception(settings, output)


def tracking_windows(source, output, samples, windows, perception, state):
    result=[]
    for number,(start,end) in enumerate(windows):
        clip=sampled_clip(source,output,samples,start,end)
        detected=perception.prepare(source,clip,state['registry'],state['visual_identity']) if clip.frames else []
        observations=[dict(o,frame_index=i,local_id=o['entity_id'],observation_id=o['detection_id'])
            for i,f in enumerate(detected) for o in f['observations']]
        frames=[dict(frame_index=i,time_sec=f['time_sec'],image=f['image'],
            boxes=[dict(local_id=o['entity_id'],bbox=o['bbox']) for o in f['observations']]) for i,f in enumerate(detected)]
        result.append(dict(window_id=f'window_{number:04d}',start_sec=start,end_sec=end,frames=frames,
            observations=observations,events=[],local_entities=[],narrative=None,warnings=[]))
    return result


def analyze(video: Path, output: Path, settings, *, backend=None, max_windows: int | None = None, perception=None, tracking_only=False):
    video, output = video.resolve(), output.resolve()
    with output_lock(output):
        source = VideoSource(video, settings.max_pixels)
        fingerprint = file_fingerprint(video, relative_to=output)
        identity = {"sha256": fingerprint["sha256"], "settings": settings.model_dump(),
                    "pipeline_version": PIPELINE_VERSION, "code_sha256": code_fingerprint()}
        manifest_path = output / "manifest.json"
        manifest = {"identity": identity, "source": fingerprint, "duration_sec": source.duration,
                    "source_fps": source.fps, "width": source.width, "height": source.height,
                    "settings": settings.model_dump()}
        if manifest_path.exists() and read_json(manifest_path)["identity"] != identity:
            raise ValueError("설정/코드/영상이 저장 결과와 다릅니다. 다른 --output을 지정하세요.")
        if (output / 'samples.json').exists():
            manifest.update(samples=read_json(output / 'samples.json'), sampled_video_url='sampled.mp4')
        atomic_json(manifest_path, manifest)
        checkpoint = output / "checkpoint.json"
        state = read_json(checkpoint) if checkpoint.exists() else {
            "status": "new", "windows": [], "events": [], "registry": {}, "pending": None, "errors": []}
        if state["status"] == "complete":
            LOG.info("이미 완료된 분석입니다. 저장된 결과를 엽니다.")
            return export(output, manifest, state)
        if state.pop('tracking_only_windows', False):
            state['windows'] = []
        backend = backend or (None if tracking_only else QwenBackend(settings, output / "model_calls"))
        timer = RunTimer(output / "timing.json")
        previous_term = None
        if threading.current_thread() is threading.main_thread():
            previous_term = signal.getsignal(signal.SIGTERM)
            def terminate(signum, frame):
                raise KeyboardInterrupt("SIGTERM")
            signal.signal(signal.SIGTERM, terminate)
        windows = list(sliding_windows(source.duration, settings.window_sec, settings.stride_sec))
        state["status"] = "running"
        completed_this_run = 0
        try:
            perception = perception or make_perception(settings, output)
            state.setdefault('visual_identity', {})
            samples = prepare_samples(source, output, settings.sample_fps)
            manifest['samples'] = samples
            manifest['sampled_video_url'] = 'sampled.mp4'
            atomic_json(manifest_path, manifest)
            if hasattr(perception, 'prepare_video'):
                state['tracks'] = perception.prepare_video(source, samples, state['registry'], state['visual_identity'])
                atomic_json(checkpoint,state)
            else:
                # Track every sampled image before any window-level VLM analysis.
                for offset in range(state.get('tracked_samples', 0), len(samples), 32):
                    batch = samples[offset:offset + 32]
                    tracking_clip = sampled_clip(source, output, batch, batch[0]['time_sec'],
                        min(source.duration, batch[-1]['time_sec'] + 1 / settings.sample_fps))
                    tracked = perception.prepare(source, tracking_clip, state['registry'], state['visual_identity'])
                    state.setdefault('tracks', []).extend(tracked)
                    state['tracked_samples'] = offset + len(batch)
                    atomic_json(checkpoint, state)
                    LOG.info('전체 영상 추적 · %d/%d 샘플', state['tracked_samples'], len(samples))
            export(output, manifest, state)
            if tracking_only:
                state['windows']=tracking_windows(source,output,samples,windows,perception,state)
                state['tracking_only_windows']=True
                state['status']='tracking_ready'
            else:
                for number in range(len(state["windows"]), len(windows)):
                    if max_windows is not None and completed_this_run >= max_windows:
                        state["status"] = "paused"
                        break
                    start, end = windows[number]
                    window_id = f"window_{number:04d}"
                    folder = output / "windows" / window_id
                    folder.mkdir(parents=True, exist_ok=True)
                    clip = sampled_clip(source, output, samples, start, end)
                    if not clip.frames:
                        state['windows'].append(dict(window_id=window_id, start_sec=start, end_sec=end,
                            frames=[], observations=[], events=[], local_entities=[], warnings=['이 구간에 전역 샘플 없음']))
                        atomic_json(checkpoint, state)
                        continue
                    LOG.info("윈도 %d/%d · %.3f~%.3f초 · %d프레임", number + 1, len(windows), start, end, len(clip.frames))
                    LOG.info("윈도 내 YOLO 검출 → 사람/물체 동일성 연결")
                    detected = perception.prepare(source, clip, state['registry'], state['visual_identity'])
                    atomic_json(checkpoint, state)
                    atomic_json(folder / 'detected.json', detected)
                    visible_ids = sorted({o['entity_id'] for f in detected for o in f['observations']})
                    known = [LocalEntity(local_id=eid, label=state['registry'][eid]['label'], kind=state['registry'][eid]['kind'],
                                         appearance=state['registry'][eid]['appearance']) for eid in visible_ids]
                    detector_frames = [dict(frame_index=i, time_sec=f['time_sec'], image=f['image'],
                        boxes=[dict(local_id=o['entity_id'], **{k:v for k,v in o.items() if k in {
                            'bbox','detection_confidence','identity_method','identity_similarity','identity_uncertain','grounding_source'}})
                               for o in f['observations']]) for i, f in enumerate(detected)]
                    (folder / 'detector_prompt.txt').write_text(context_prompt(clip, known, detector_frames))
                    entity_path = folder / "entities.json"
                    if entity_path.exists():
                        discovered = EntityList.model_validate(read_json(entity_path))
                    else:
                        LOG.info("검출 ID를 전달하여 Qwen이 추가 물체·배경 확인")
                        discovered = backend.discover(clip, known_entities=known, grounded=detector_frames)
                        atomic_json(entity_path, discovered.model_dump(mode="json"))
                    grounded = []
                    for i, frame in enumerate(clip.frames):
                        path = folder / f"frame_{i:03d}.json"
                        if path.exists():
                            result = read_json(path)
                        else:
                            LOG.info("추가 물체·배경의 프레임별 bbox · %d/%d", i + 1, len(clip.frames))
                            boxes = backend.ground_frame(frame, discovered.entities, frame_index=i,
                                known_entities=known, detected_boxes=detector_frames[i]['boxes'])
                            result = {"frame_index": i, "time_sec": frame.time_sec, "image": frame.path,
                                      "boxes": [dict(b.model_dump(mode="json"), grounding_source='Qwen') for b in boxes.boxes]
                                               + detector_frames[i]['boxes'],
                                      "new_entities": [e.model_dump(mode="json") for e in boxes.new_entities],
                                      "warnings": boxes.warnings}
                            atomic_json(path, result)
                        discovered.entities.extend(LocalEntity.model_validate(e) for e in result.get("new_entities", []))
                        grounded.append(result)
                    local_entities = discovered.entities + known
                    observations = crop_observations(clip, grounded, local_entities, output, window_id, original_images=True)
                    for o in observations:
                        source_box = next(b for b in grounded[o['frame_index']]['boxes'] if b['local_id'] == o['local_id'])
                        o.update({k:v for k,v in source_box.items() if k not in {'local_id','bbox'}})
                    if state["pending"] is None:
                        state["pending"] = {"window_id": window_id, "next_frame": 0, "assignments": {}}
                    pending = state["pending"]
                    if pending["window_id"] != window_id:
                        raise ValueError("Checkpoint window mismatch")
                    for i in range(pending["next_frame"], len(clip.frames)):
                        queries = [o for o in observations if o["frame_index"] == i]
                        if queries:
                            registry = deepcopy(state["registry"])
                            tracked = [q for q in queries if q['local_id'] in visible_ids]
                            mapping = {}
                            for q in tracked:
                                eid = q['local_id']
                                mapping[q['observation_id']] = eid
                                registry[eid]['observations'].append({**q, 'entity_id': eid})
                            remaining = [q for q in queries if q['local_id'] not in visible_ids]
                            if remaining:
                                LOG.info("추가 물체·배경 crop 동일성 비교 · 프레임 %d", i)
                                # Qwen may link its supplemental entities, but cannot
                                # merge or rename detector-assigned identities.
                                protected = {eid for eid in registry if eid.startswith(('person_', 'object_'))}
                                mapping.update(resolve_frame(remaining, registry, output, settings, backend, unavailable=protected))
                            state["registry"] = registry
                            pending["assignments"].update(mapping)
                        pending["next_frame"] = i + 1
                        atomic_json(checkpoint, state)
                    pending["assignments"] = {oid: canonical(state["registry"], eid) for oid, eid in pending["assignments"].items()}
                    # Event extraction sees final global IDs for every localized
                    # entity, not window-local names that are replaced afterward.
                    for o in observations:
                        old_id = o['local_id']
                        eid = pending['assignments'][o['observation_id']]
                        box = next(b for b in grounded[o['frame_index']]['boxes'] if b['local_id'] == old_id)
                        box['local_id'] = eid
                        o.update(source_local_id=old_id, local_id=eid)
                    ids = sorted(set(pending['assignments'].values()))
                    final_entities = [LocalEntity(local_id=eid, label=state['registry'][eid]['label'],
                        kind=state['registry'][eid]['kind'], appearance=state['registry'][eid]['appearance']) for eid in ids]
                    atomic_json(folder / 'entity_context.json', entity_context(clip, final_entities, grounded))
                    (folder / 'entity_prompt.txt').write_text(context_prompt(clip, final_entities, grounded))
                    scan_path = folder / "scan.json"
                    if scan_path.exists():
                        scan = ScanResult.model_validate(read_json(scan_path)).validate_clip(len(clip.frames))
                    else:
                        LOG.info("전역 ID·프레임 관찰을 전달하여 Qwen 이벤트 분석")
                        scan = backend.scan(clip, final_entities, grounded)
                        atomic_json(scan_path, scan.model_dump(mode="json"))
                    if scan.narrative is None:
                        scan.narrative = backend.narrate(clip, final_entities, grounded)
                        atomic_json(scan_path, scan.model_dump(mode="json"))
                    events = window_events(scan, clip, observations, pending["assignments"], window_id, state["registry"])
                    for event in events:
                        if any(grounded[i].get("warnings") for i in event["frame_indices"]):
                            event["uncertain"] = True
                            event["boundary_note"] += " (구간 내 bbox 검출 실패 프레임 있음)"
                    window = {"window_id": window_id, "start_sec": start, "end_sec": end,
                              "frames": grounded, "events": events, "warnings": scan.warnings,
                              "narrative": scan.narrative.model_dump() if scan.narrative else None,
                              "observations": [{**o, "entity_id": pending["assignments"][o["observation_id"]]} for o in observations],
                              "local_entities": [e.model_dump() for e in scan.entities]}
                    atomic_json(folder / "result.json", window)
                    state["windows"].append(window)
                    state["events"].extend(events)
                    state["pending"] = None
                    completed_this_run += 1
                    atomic_json(checkpoint, state)
                    export(output, manifest, state)
                    LOG.info("윈도 완료 · 이벤트 %d개 · 누적 엔티티 %d개", len(events), len(state["registry"]))
                else:
                    state["status"] = "complete"
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            LOG.warning("중단 위치와 누적 실행 시간을 저장했습니다. 같은 명령으로 재개하세요.")
            raise
        except Exception as exc:
            state["status"] = "failed"
            state["errors"].append({"window": len(state["windows"]), "error": str(exc), "resolved": False})
            raise
        finally:
            timing = timer.close(state["status"])
            if previous_term is not None:
                signal.signal(signal.SIGTERM, previous_term)
            if state["status"] == "complete":
                for error in state["errors"]:
                    error["resolved"] = True
            atomic_json(checkpoint, state)
            payload = export(output, manifest, state, timing)
            LOG.info("상태 %s · 누적 수행 %.2f초 (%d회 실행)", state["status"], timing["total_active_sec"], len(timing["sessions"]))
        return payload
