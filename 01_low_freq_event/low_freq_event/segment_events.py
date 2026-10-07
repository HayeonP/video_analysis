"""User-invoked 10 fps, non-overlapping 5 s event analysis of frozen tracks."""
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import shutil
from typing import Literal

from PIL import Image
from pydantic import Field

from .backend import QwenBackend, ModelOutputError
from .config import Settings
from .context import annotated_clip, entity_token
from .schema import StrictModel, LocalEntity
from .storage import atomic_json, read_json, file_fingerprint
from .video import Clip, Frame, fit_image, sliding_windows

LOG = logging.getLogger(__name__)


class Activity(StrictModel):
    entity_id: str
    summary: str = Field(min_length=1)
    visibility: Literal['visible', 'partially_visible', 'not_visible']
    uncertain: bool


class SceneSummary(StrictModel):
    summary: str = Field(min_length=1)
    uncertain: bool


class SceneContext(StrictModel):
    background: SceneSummary
    composition: SceneSummary


class SegmentSummary(StrictModel):
    activities: list[Activity]
    background: SceneSummary
    composition: SceneSummary


PROMPT_DIR = Path(__file__).resolve().parents[1]/'prompts'


def load_summary_prompts():
    prompts={name:(PROMPT_DIR/filename).read_text(encoding='utf-8').strip() for name,filename in
             [('person','person.txt'),('scene','scene.txt')]}
    if any(not text for text in prompts.values()):raise ValueError('프롬프트 파일이 비어 있습니다.')
    return prompts


def observation_profile(frames, eid):
    indices=[i for i,f in enumerate(frames) if any(o['entity_id']==eid for o in f['observations'])]
    runs=[]
    for i in indices:
        if runs and i==runs[-1][-1]+1: runs[-1].append(i)
        else: runs.append([i])
    return dict(entity_id=eid,observed_frames=len(indices),total_frames=len(frames),
        observed_at_window_start=0 in indices,observed_at_window_end=len(frames)-1 in indices,
        intervals=[dict(first_frame=r[0],last_frame=r[-1],first_observed_sec=frames[r[0]]['time_sec'],
                        last_observed_sec=frames[r[-1]]['time_sec']) for r in runs],
        time_basis='source_video_seconds',meaning='Tracking observations only; missing detection does not prove offscreen presence')




def load_plan(source_result):
    base = Path(source_result).resolve()
    original = read_json(base/'timeline.json')
    if original['status'] not in {'tracking_ready','complete'}:
        raise ValueError('추적이 완료된 결과 폴더를 입력하세요.')
    tracking = read_json(base/'tracking/result.json')
    if abs(tracking['tracking_fps']-10.) > 1e-6:
        raise ValueError('이 스크립트는 저장된 10fps 추적 결과가 필요합니다.')
    frames = read_json(base/'tracking/frames.json')
    duration = original['video']['duration_sec']
    if not frames or any(b['time_sec']<=a['time_sec'] for a,b in zip(frames,frames[1:])):
        raise ValueError('추적 프레임이 비었거나 시간 순서가 올바르지 않습니다.')
    if frames[0]['time_sec']>.11 or duration-frames[-1]['time_sec']>.15 or any(
            b['time_sec']-a['time_sec']>.15 for a,b in zip(frames,frames[1:])):
        raise ValueError('10fps 추적 프레임에 누락 구간이 있습니다.')
    known={e['entity_id'] for e in original['entities']}
    for frame in frames:
        if not (base/frame['image']).is_file(): raise ValueError(f"프레임 이미지 없음: {frame['image']}")
        ids=[o['entity_id'] for o in frame['observations']]
        if len(ids)!=len(set(ids)) or not set(ids)<=known:
            raise ValueError('추적 ID가 중복되거나 entity 목록에 없습니다.')
    windows=[]
    for index,(start,end) in enumerate(sliding_windows(duration,5.,5.)):
        selected=[f for f in frames if start<=f['time_sec']<end]
        if not selected: raise ValueError(f'빈 분석 구간: {start}–{end}')
        windows.append(dict(window_id=f'window_{index:04d}',start_sec=start,end_sec=end,frames=selected))
    return original,frames,windows


def event_context(clip, entities, grounded):
    """Compact bbox evidence in pixels of the resized VLM input, not source pixels."""
    width,height=clip.frames[0].image.size
    records=[]
    for frame,ground in zip(clip.frames,grounded):
        if frame.image.size!=(width,height): raise ValueError('Mixed input image sizes')
        boxes=[]
        for box in ground['boxes']:
            x1,y1,x2,y2=box['bbox']
            pixels=[round(max(0,min(1000,x1))*width/1000),round(max(0,min(1000,y1))*height/1000),
                    round(max(0,min(1000,x2))*width/1000),round(max(0,min(1000,y2))*height/1000)]
            boxes.append([entity_token(box['local_id']),*pixels,int(box.get('identity_uncertain',False))])
        records.append([ground['frame_index'],round(frame.time_sec,3),boxes])
    return dict(coordinate_system='xyxy pixels in resized input image',image_size=[width,height],
        box_columns=['token','x1','y1','x2','y2','identity_uncertain'],frame_columns=['frame_index','source_time_sec','boxes'],
        entities=[dict(id=e.local_id,token=entity_token(e.local_id),kind=e.kind,label=e.label) for e in entities],frames=records)


def event_context_prompt(clip, entities, grounded):
    return ('\n박스 좌표는 리사이즈된 입력 이미지의 픽셀 단위 xyxy입니다. '
            '번호 표시는 영상 속 물체가 아닙니다. tokens를 대응하는 원래 entity_id로 출력하세요. '
            '추적 ID를 변경하지 마세요.\n'+json.dumps(event_context(clip,entities,grounded),ensure_ascii=False,separators=(',',':')))


class EventBackend(QwenBackend):
    """Record actual inference peaks and input dimensions for reproducible tuning."""
    def _generate(self,prompt,clip,labeled_images,max_tokens,image_pixels=None):
        import time
        import torch
        self.load()
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic();status='failed'
        try:
            result=super()._generate(prompt,clip,labeled_images,max_tokens,image_pixels)
            status='complete'
            return result
        finally:
            record=dict(status=status,frames=len(clip.frames) if clip else 0,
                input_size=list(clip.frames[0].image.size) if clip else None,
                prompt_chars=len(prompt),max_new_tokens=max_tokens,elapsed_sec=round(time.monotonic()-started,3),
                peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3),
                peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,3))
            folder=self.trace_dir.parent/'memory_metrics';folder.mkdir(parents=True,exist_ok=True)
            atomic_json(folder/f'{time.time_ns()}.json',record)
            self.recover()
            LOG.info('VLM 메모리: %s · %s · 할당 %.2f / 예약 %.2f GiB',status,record['input_size'],record['peak_allocated_gib'],record['peak_reserved_gib'])


def prepare_clip(base, frames, start, end, settings, source_fps):
    images=[];grounded=[]
    for index,frame in enumerate(frames):
        with Image.open(base/frame['image']) as image:
            images.append(Frame(frame['time_sec'],fit_image(image.convert('RGB'),settings.max_pixels),frame['image']))
        grounded.append(dict(frame_index=index,time_sec=frame['time_sec'],image=frame['image'],
            boxes=[dict(local_id=o['entity_id'],bbox=o['bbox'],identity_uncertain=o.get('identity_uncertain',False))
                   for o in frame['observations']]))
    clip=Clip(start,end,source_fps,images)
    return annotated_clip(clip,grounded),grounded


def normalize_activity(activity, expected, frames):
    aliases={entity_token(eid):eid for eid in expected}
    activity.entity_id=aliases.get(activity.entity_id,activity.entity_id)
    if activity.entity_id not in expected: raise ValueError('Unknown activity entity_id')
    visible=sum(any(o['entity_id']==activity.entity_id for o in f['observations']) for f in frames)
    if not visible and activity.visibility!='not_visible':
        raise ValueError('Entity is not observed in this window; return not_visible')
    if visible and activity.visibility=='not_visible':
        activity.uncertain=True
    if 0<visible<len(frames):
        activity.uncertain=True
        if activity.visibility=='visible': activity.visibility='partially_visible'
    return activity


def summarize(backend, base, window, registry, settings, source_fps, *, prompts=None):
    prompts=prompts if prompts is not None else load_summary_prompts()
    frames=window['frames'];clip,grounded=prepare_clip(base,frames,window['start_sec'],window['end_sec'],settings,source_fps)
    seen={o['entity_id'] for f in frames for o in f['observations']}
    expected=sorted(eid for eid,e in registry.items() if e['kind']=='person')
    entities=[LocalEntity(local_id=eid,label=registry[eid]['label'],kind='person',appearance=registry[eid]['appearance']) for eid in expected]
    context=event_context_prompt(clip,entities,grounded)+f'\n요약 대상 구간: [{clip.start_sec}, {clip.end_sec})초. 총 {len(frames)}프레임.'
    profiles={eid:observation_profile(frames,eid) for eid in expected}
    context+='\n추적 관찰 구간 (각 구간의 첫/마지막 샘플 시각, 원본 영상 초 단위):\n'+json.dumps(profiles,ensure_ascii=False)
    if hasattr(backend,'trace_dir'):
        audit=event_context(clip,entities,grounded)
        audit.update(observation_profiles=profiles,analysis_strategy='full_frame_per_person_then_scene')
        atomic_json(backend.trace_dir.parent/'windows'/window['window_id']/'input_context.json',audit)
    activities=[]
    for eid in expected:
        if eid not in seen:
            activity=Activity(entity_id=eid,summary='이 구간에서는 추적으로 관찰되지 않아 행동을 판단할 수 없음',visibility='not_visible',uncertain=True)
        else:
            focused=(prompts['person']+"\n"+context+f'\n분석 대상 entity_id: {eid} (화면 표시 {entity_token(eid)}). '
                +'전체 화면과 주변 인물을 보면서 이 대상 한 명만 설명하세요. '
                +'처음/마지막 상태는 윈도 경계가 아니라 해당 인물이 실제로 관찰된 첫/마지막 상태입니다. '
                +'Activity JSON 하나: '+json.dumps(Activity.model_json_schema(),ensure_ascii=False))
            LOG.info('%s · %s 개별 행동 요약 · 전체 화면 %d프레임',window['window_id'],eid,len(frames))
            try:
                activity=backend.request(focused,Activity,clip=clip,
                    validate=lambda a:normalize_activity(a,[eid],frames),max_tokens=2000)
            except ModelOutputError:
                LOG.warning('%s 행동 판정 보류: 개별 응답 검증 실패',eid)
                activity=Activity(entity_id=eid,summary='행동 판정 보류: 모델 응답 검증 실패',visibility='partially_visible',uncertain=True)
        activities.append(activity)
    LOG.info('%s · 배경·구도 요약',window['window_id'])
    scene=backend.request(prompts['scene']+"\n"+context+'\nSceneContext JSON: '+json.dumps(SceneContext.model_json_schema(),ensure_ascii=False),
        SceneContext,clip=clip,validate=lambda value:value,max_tokens=2000)
    return SegmentSummary(activities=activities,background=scene.background,composition=scene.composition),grounded


def summary_record(description, uncertain, entity_ids, category, window, index):
    # Bounds describe the summary's window, never the onset/duration of an action.
    frames=window['frames'];start=window['start_sec'];end=window['end_sec']
    evidence=[dict(frame_index=i,time_sec=f['time_sec'],image=f['image'],window_id=window['window_id']) for i,f in enumerate(frames)]
    return dict(description=description,uncertain=uncertain,action={'activity':'인물 요약','background':'배경 요약','composition':'구도·카메라 요약'}[category],
        event_id=f"{window['window_id']}_event_{index:03d}",window_id=window['window_id'],window_ids=[window['window_id']],entity_ids=entity_ids,
        category=category,event_type='window_summary',start_sec=start,end_sec=end,duration_sec=round(end-start,6),
        start_frame=0,end_frame=len(frames)-1,
        participants=[dict(local_id=eid,label=eid,entity_ids=[eid],observation_ids=[],localization_status='located') for eid in entity_ids],
        frame_indices=list(range(len(frames))),evidence=evidence,temporal_basis='window_summary',temporal_evidence=[],time_basis='summary_window',
        boundary_note='표시된 시간은 요약 대상 윈도입니다. 개별 행동의 시작·종료 시각이나 지속시간을 뜻하지 않습니다.')


def build_window(response, saved, grounded, registry):
    window=deepcopy(saved);events=[]
    analysis=response.model_dump()
    for a in analysis['activities']:
        a['observation']=observation_profile(saved['frames'],a['entity_id'])
    for activity in response.activities:
        events.append(summary_record(activity.summary,activity.uncertain,[activity.entity_id],'activity',window,len(events)))
    for category in ('background','composition'):
        scene=getattr(response,category)
        events.append(summary_record(scene.summary,scene.uncertain,[],category,window,len(events)))
    observations=[dict(o,local_id=o['entity_id'],observation_id=o['detection_id'],frame_index=i) for i,f in enumerate(saved['frames']) for o in f['observations']]
    window.update(frames=grounded,observations=observations,events=events,local_entities=[],warnings=[a.entity_id+': '+a.summary for a in response.activities if a.summary.startswith('행동 판정 보류:')],analysis=analysis,
        narrative=dict(background=response.background.summary,composition=response.composition.summary,summary='',
            activities=[dict(entity_id=a.entity_id,description=a.summary,uncertain=a.uncertain) for a in response.activities]))
    return window


def independent_events(windows):
    """Preserve each window's separate actions and local frame indices."""
    return [deepcopy(event) for window in windows for event in window['events']]


def analyze_segments(source_result,output,*,max_windows=None,max_pixels=65536,backend=None):
    from .pipeline import output_lock, code_fingerprint
    from .timing import RunTimer
    from .viewer import generate_viewer,resolve_video
    base,output=Path(source_result).resolve(),Path(output).resolve()
    if base==output or base in output.parents or output in base.parents: raise ValueError('입력과 분리된 새 결과 폴더를 지정하세요.')
    original,frames,plan=load_plan(base)
    prompts=load_summary_prompts()
    person_ids={e['entity_id'] for e in original['entities'] if e['kind']=='person'}
    frames=[dict(f,observations=[o for o in f['observations'] if o['entity_id'] in person_ids]) for f in frames]
    for window in plan:
        window['frames']=[f for f in frames if window['start_sec']<=f['time_sec']<window['end_sec']]
    settings=Settings.model_validate(dict(original['video']['settings'],sample_fps=10.,window_sec=5.,stride_sec=5.,max_pixels=max_pixels,max_new_tokens=8192))
    identity=dict(source_timeline=file_fingerprint(base/'timeline.json')['sha256'],source_tracks=file_fingerprint(base/'tracking/frames.json')['sha256'],
        code=hashlib.sha256((code_fingerprint()+Path(__file__).read_text()).encode()).hexdigest(),settings=settings.model_dump(),prompts=prompts,mode='full_frame_per_person_summaries_v6')
    with output_lock(output):
        revision=output/'event_revision.json'
        if revision.exists():
            if read_json(revision)['identity']!=identity: raise ValueError('입력/설정/코드가 변경됐습니다. 새 --output을 사용하세요.')
        else:
            if (output/'manifest.json').exists() or (output/'checkpoint.json').exists(): raise ValueError('다른 분석이 있는 폴더입니다. 새 --output을 사용하세요.')
            atomic_json(revision,dict(identity=identity,source_result=str(base)))
        atomic_json(output/'prompts_used.json',prompts)
        for name in ('frames','tracking/frames','crops'):
            if (base/name).exists(): shutil.copytree(base/name,output/name,dirs_exist_ok=True)
        if (base/'tracking.mp4').exists() and not (output/'sampled.mp4').exists(): shutil.copy2(base/'tracking.mp4',output/'sampled.mp4')
        video=resolve_video(original['video']['source'],base,None)
        manifest=deepcopy(original['video']);manifest.update(settings=settings.model_dump(),identity=identity,
            source=file_fingerprint(video,relative_to=output),event_extraction='10fps_window_summary',analysis_strategy='full_frame_per_person_then_scene',event_scope='people_and_scene',window_policy='independent',
            samples=[dict(time_sec=f['time_sec'],playback_sec=i/10,image=f['image']) for i,f in enumerate(frames)],sampled_video_url='sampled.mp4')
        atomic_json(output/'manifest.json',manifest);atomic_json(output/'samples.json',manifest['samples'])
        checkpoint=output/'checkpoint.json'
        state=read_json(checkpoint) if checkpoint.exists() else dict(status='new',windows=[],events=[],registry={e['entity_id']:deepcopy(e) for e in original['entities'] if e['kind']=='person'},errors=[])
        def publish(timing=None):
            payload=dict(schema_version=1,video=manifest,status=state['status'],windows=state['windows'],events=state['events'],
                entities=list(state['registry'].values()),tracks=frames,entity_aliases={k:v for k,v in original.get('entity_aliases',{}).items() if v in person_ids},errors=state['errors'],
                timing=timing or (read_json(output/'timing.json') if (output/'timing.json').exists() else {}),
                event_policy='Independent 5 s summaries of people, background and composition; no action timestamps or cross-window merging')
            atomic_json(checkpoint,state);atomic_json(output/'timeline.json',payload)
            atomic_json(output/'events_by_person.json',{eid:dict(label=e['label'],events=[event for event in state['events'] if eid in event['entity_ids']],
                summaries=[dict(window_id=w['window_id'],start_sec=w['start_sec'],end_sec=w['end_sec'],**a) for w in state['windows'] for a in w['analysis']['activities'] if a['entity_id']==eid]) for eid,e in state['registry'].items()})
            atomic_json(output/'events.json',state['events']);atomic_json(output/'entities.json',payload['entities'])
            atomic_json(output/'window_summaries.json',[dict(window_id=w['window_id'],start_sec=w['start_sec'],end_sec=w['end_sec'],**w['analysis']) for w in state['windows']])
            generate_viewer(output)
            return payload
        if state['status']=='complete': return publish()
        backend=backend or EventBackend(settings,output/'model_calls')
        timer=RunTimer(output/'timing.json');state['status']='running'
        try:
            processed=0
            for saved in plan[len(state['windows']):]:
                if max_windows is not None and processed>=max_windows: state['status']='paused';break
                LOG.info('행동 분석 %s · %.1f–%.1f초 · %d프레임',saved['window_id'],saved['start_sec'],saved['end_sec'],len(saved['frames']))
                response,grounded=summarize(backend,base,saved,state['registry'],settings,manifest['source_fps'],prompts=prompts)
                window=build_window(response,saved,grounded,state['registry']);state['windows'].append(window)
                state['events']=independent_events(state['windows']);processed+=1;publish()
            else:
                state['events']=independent_events(state['windows']);state['status']='complete'
                for error in state['errors']: error['resolved']=True
        except KeyboardInterrupt:
            state['status']='interrupted';raise
        except Exception as exc:
            state['status']='failed';state['errors'].append(dict(window=len(state['windows']),error=str(exc),resolved=False));raise
        finally:
            timing=timer.close(state['status']);publish(timing)
        return read_json(output/'timeline.json')
