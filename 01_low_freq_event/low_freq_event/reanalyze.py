"""Resume event extraction while preserving reviewed boxes and global identities."""
from copy import deepcopy
from pathlib import Path
import shutil

from .backend import QwenBackend
from .config import Settings
from .pipeline import code_fingerprint, export, output_lock, window_events
from .identity import canonical
from .context import entity_context, context_prompt
from .schema import LocalEntity, ScanResult
from .storage import atomic_json, file_fingerprint, read_json
from .timing import RunTimer
from .video import VideoSource, sampled_clip


def global_window_context(window, registry):
    """Adapt older window-local IDs without rerunning detection or identity."""
    result = deepcopy(window)
    mapping = {}
    for observation in result['observations']:
        local_id = observation['local_id']
        eid = canonical(registry, observation['entity_id'])
        mapping[observation['frame_index'], local_id] = eid
        if local_id != eid:
            observation.setdefault('source_local_id', local_id)
        observation.update(local_id=eid, entity_id=eid)
    for i, frame in enumerate(result['frames']):
        for box in frame['boxes']:
            box['local_id'] = mapping[i, box['local_id']]
    result['local_entities'] = [dict(local_id=eid, label=registry[eid]['label'], kind=registry[eid]['kind'],
        appearance=registry[eid]['appearance']) for eid in sorted(set(mapping.values()))]
    return result


def reanalyze_events(source_result: Path, output: Path, *, backend=None, max_windows=None):
    source_result, output = source_result.resolve(), output.resolve()
    if source_result == output:
        raise ValueError('원본을 보존하도록 다른 --output을 지정하세요.')
    original = read_json(source_result / 'timeline.json')
    if original['status'] != 'complete':
        raise ValueError('완료된 분석 결과가 필요합니다.')
    settings = Settings.model_validate(original['video']['settings'])
    from .viewer import resolve_video
    video = resolve_video(original['video']['source'], source_result, None)
    source = VideoSource(video, settings.max_pixels)
    identity = {'source_sha256':file_fingerprint(source_result/'timeline.json')['sha256'],
                'code_sha256':code_fingerprint(), 'mode':'frame_evidence_events_only'}
    with output_lock(output):
        revision = output / 'event_revision.json'
        if revision.exists():
            if read_json(revision)['identity'] != identity:
                raise ValueError('이벤트 재분석의 원본/코드가 다릅니다. 다른 --output을 지정하세요.')
        else:
            # Independent copies survive moving/archiving the earlier result.
            for name in ('frames','crops','perception'):
                if (source_result/name).is_dir():
                    shutil.copytree(source_result/name, output/name, dirs_exist_ok=True)
            for name in ('samples.json', 'sampled.mp4'):
                if (source_result/name).exists():
                    shutil.copy2(source_result/name, output/name)
            if (source_result/'timing.json').exists():
                shutil.copy2(source_result/'timing.json', output/'timing.json')
            atomic_json(revision, {'identity':identity,'source_result':str(source_result),
                'inherited_active_sec':original['timing']['total_active_sec'],
                'reused':'Saved bbox observations, final entity registry and source sample frames; only events are inferred again.'})
        manifest = deepcopy(original['video'])
        manifest['identity'] = identity
        manifest['event_extraction'] = 'frame_evidence_v1'
        manifest['source'] = file_fingerprint(video, relative_to=output)
        atomic_json(output/'manifest.json',manifest)
        checkpoint = output/'checkpoint.json'
        state = read_json(checkpoint) if checkpoint.exists() else {
            'status':'new','windows':[],'events':[], 'tracks':deepcopy(original.get('tracks', [])),
            'registry':{e['entity_id']:deepcopy(e) for e in original['entities']},'errors':[],'pending':None}
        if state['status'] == 'complete':
            return export(output,manifest,state)
        backend = backend or QwenBackend(settings,output/'model_calls')
        timer = RunTimer(output/'timing.json')
        state['status'] = 'running'
        try:
            completed = 0
            for saved_window in original['windows'][len(state['windows']):]:
                old = global_window_context(saved_window, state['registry'])
                if max_windows is not None and completed >= max_windows:
                    state['status'] = 'paused'
                    break
                folder = output/'windows'/old['window_id']
                folder.mkdir(parents=True,exist_ok=True)
                if not old['frames']:
                    state['windows'].append(old)
                    atomic_json(checkpoint, state)
                    continue
                clip = sampled_clip(source, output, old['frames'], old['start_sec'], old['end_sec'])
                if len(clip.frames) != len(old['frames']) or any(abs(f.time_sec-g['time_sec'])>1e-6 for f,g in zip(clip.frames,old['frames'])):
                    raise ValueError('저장된 bbox와 원본 샘플 시각이 일치하지 않습니다.')
                for f,g in zip(clip.frames,old['frames']):
                    f.path = g['image']
                entities = [LocalEntity.model_validate(e) for e in old['local_entities']]
                atomic_json(folder/'entity_context.json', entity_context(clip,entities,old['frames']))
                (folder/'entity_prompt.txt').write_text(context_prompt(clip,entities,old['frames']))
                scan_path = folder/'scan.json'
                if scan_path.exists():
                    scan = ScanResult.model_validate(read_json(scan_path)).validate_clip(len(clip.frames))
                else:
                    import logging
                    logging.getLogger(__name__).info('이벤트 재분석 %s · %.3f~%.3f초',old['window_id'],clip.start_sec,clip.end_sec)
                    scan = backend.scan(clip,entities,old['frames'])
                    atomic_json(scan_path,scan.model_dump(mode='json'))
                if scan.narrative is None:
                    scan.narrative = backend.narrate(clip,entities,old['frames'])
                    atomic_json(scan_path,scan.model_dump(mode='json'))
                assignments = {o['observation_id']:o['entity_id'] for o in old['observations']}
                events = window_events(scan,clip,old['observations'],assignments,old['window_id'],state['registry'])
                for event in events:
                    if any(old['frames'][i].get('warnings') for i in event['frame_indices']):
                        event['uncertain'] = True
                        event['boundary_note'] += ' (구간 내 bbox 검출 실패 프레임 있음)'
                window = dict(deepcopy(old),events=events,warnings=scan.warnings,
                    narrative=scan.narrative.model_dump() if scan.narrative else None)
                atomic_json(folder/'result.json',window)
                state['windows'].append(window)
                state['events'].extend(events)
                completed += 1
                atomic_json(checkpoint,state)
                export(output,manifest,state)
            else:
                state['status'] = 'complete'
                for error in state['errors']:
                    error['resolved'] = True
        except KeyboardInterrupt:
            state['status'] = 'interrupted'
            raise
        except Exception as exc:
            state['status'] = 'failed'
            state['errors'].append({'window':len(state['windows']),'error':str(exc),'resolved':False})
            raise
        finally:
            timing = timer.close(state['status'])
            atomic_json(checkpoint,state)
            payload = export(output,manifest,state,timing)
        return payload
