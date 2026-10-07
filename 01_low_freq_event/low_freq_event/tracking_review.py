"""Replay saved detector embeddings on CPU; never reuse events with changed identities."""
from copy import deepcopy
from pathlib import Path
import shutil
import time

from .config import Settings
from .detection import WindowPerception
from .identity import canonical
from .pipeline import output_lock, export, code_fingerprint
from .storage import read_json, atomic_json
from .video import VideoSource, sampled_clip
from .viewer import resolve_video


def review_tracking(source_result: Path, output: Path):
    source_result, output = source_result.resolve(), output.resolve()
    original = read_json(source_result / 'timeline.json')
    samples = original['video'].get('samples')
    if not samples:
        raise ValueError('전역 샘플 목록이 있는 분석 결과가 필요합니다.')
    if source_result == output or (output / 'manifest.json').exists():
        raise ValueError('추적 비교용으로 새로운 --output 폴더를 지정하세요.')
    for sample in samples:
        key = f"t{round(sample['time_sec']*1_000_000):012d}"
        if not (source_result / 'perception' / 'frames' / (key+'.json')).is_file():
            raise ValueError(f'저장된 검출 특징이 없습니다: {key}. 모델은 자동 실행하지 않습니다.')
    settings = Settings.model_validate(dict(original['video']['settings'], tracking_backend='appearance'))
    source_path = resolve_video(original['video']['source'], source_result, None)
    source = VideoSource(source_path, settings.max_pixels)
    with output_lock(output):
        started = time.monotonic()
        for name in ('frames', 'crops'):
            shutil.copytree(source_result/name, output/name, dirs_exist_ok=True)
        shutil.copytree(source_result/'perception'/'frames', output/'perception'/'frames', dirs_exist_ok=True)
        for name in ('sampled.mp4','samples.json'):
            if (source_result/name).exists(): shutil.copy2(source_result/name, output/name)
        manifest = deepcopy(original['video'])
        manifest['settings'] = settings.model_dump()
        manifest['identity'] = dict(mode='tracking_review', code_sha256=code_fingerprint())
        manifest['source_result'] = str(source_result)
        manifest['review_note'] = 'CPU로 저장된 검출 특징만 다시 연결. 이전 이벤트·서술은 새로운 ID에 재사용하지 않음.'
        atomic_json(output/'manifest.json', manifest)
        registry, visual, frames = {}, {}, []
        perception = WindowPerception(settings, output)
        for start in range(0, len(samples), 32):
            batch=samples[start:start+32]
            clip=sampled_clip(source, output, batch, batch[0]['time_sec'],
                min(source.duration,batch[-1]['time_sec']+1/settings.sample_fps))
            frames.extend(perception.prepare(source,clip,registry,visual))
        assert perception.encoder is None, 'Tracking review must never start detector inference'
        for frame in frames:
            for o in frame['observations']:
                o.update(visual['assignments'][o['detection_id']])
                eid=canonical(registry,o['entity_id'])
                o.update(entity_id=eid, local_id=eid, observation_id=o['detection_id'])
                registry[eid]['observations'].append(deepcopy(o))
        windows=[]
        for old in original['windows']:
            selected=[f for f in frames if old['start_sec'] <= f['time_sec'] < old['end_sec']]
            grounded=[]; observations=[]
            for i,f in enumerate(selected):
                grounded.append(dict(frame_index=i,time_sec=f['time_sec'],image=f['image'],
                    boxes=[dict(local_id=o['entity_id'],bbox=o['bbox']) for o in f['observations']]))
                observations.extend(dict(o,frame_index=i) for o in f['observations'])
            windows.append(dict(window_id=old['window_id'],start_sec=old['start_sec'],end_sec=old['end_sec'],
                frames=grounded,observations=observations,events=[],local_entities=[],narrative=None,warnings=[]))
        state=dict(status='tracking_review',windows=windows,events=[],registry=registry,tracks=frames,errors=[])
        summary=dict(source_result=str(source_result),
            before_person_ids=sum(e['kind']=='person' for e in original['entities']),
            after_person_ids=sum(e['kind']=='person' and not e.get('alias_of') for e in registry.values()),
            camera_continuity_links=sum(o['identity_method']=='appearance_camera_continuity' for f in frames for o in f['observations']),
            warning='ID 수 감소는 정확도 점수가 아닙니다. 교차·가림 구간의 오연결을 함께 검토해야 합니다.',
            elapsed_sec=round(time.monotonic()-started,3))
        atomic_json(output/'tracking_comparison.json',summary)
        atomic_json(output/'tracking_state.json',visual)
        payload=export(output,manifest,state,dict(total_active_sec=summary['elapsed_sec'],sessions=[]))
        return summary
