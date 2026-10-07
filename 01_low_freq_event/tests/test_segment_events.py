from copy import deepcopy
import json

import av
import numpy as np
from PIL import Image
import pytest

from low_freq_event.config import Settings
from low_freq_event.segment_events import (analyze_segments,load_plan,SegmentSummary,
    Activity,SceneContext,observation_profile,normalize_activity)
from low_freq_event.storage import atomic_json,read_json,file_fingerprint


@pytest.fixture
def tracked(tmp_path):
    base=tmp_path/'tracked';base.mkdir();(base/'frames').mkdir()
    video=base/'video.mp4'
    with av.open(str(video),'w') as output:
        stream=output.add_stream('libx264',rate=10);stream.width=96;stream.height=64;stream.pix_fmt='yuv420p'
        for i in range(110):
            image=Image.new('RGB',(96,64),(i,40,80));frame=av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    frames=[]
    for i in range(110):
        path=f'frames/t{i:03d}.jpg';Image.new('RGB',(96,64),(i,40,80)).save(base/path)
        frames.append(dict(time_sec=i/10,image=path,observations=[dict(entity_id='person_0001',
            detection_id=f'd{i}',bbox=[100,100,500,900],time_sec=i/10,kind='person',identity_uncertain=False)]))
    entity=dict(entity_id='person_0001',kind='person',label='인물 1',appearance='흰 옷',gallery=[],observations=[],possible_matches=[],identity_status='matched')
    manifest=dict(source=file_fingerprint(video),duration_sec=11.,source_fps=10.,width=96,height=64,settings=Settings().model_dump())
    atomic_json(base/'timeline.json',dict(status='tracking_ready',video=manifest,entities=[entity]))
    atomic_json(base/'tracking/result.json',dict(tracking_fps=10))
    atomic_json(base/'tracking/frames.json',frames)
    return base


class FakeEvents:
    def __init__(self,missing=False):self.calls=[];self.missing=missing
    def request(self,prompt,schema,*,clip,validate,**kwargs):
        self.calls.append((schema.__name__,clip.start_sec,clip.end_sec,len(clip.frames)))
        activity=dict(entity_id='person_0001',summary='서 있다가 양팔을 번갈아 흔든 뒤 몸을 돌린다',visibility='visible',uncertain=False)
        if schema is SceneContext:
            value=dict(background=dict(summary='무대 뒤 배경이 유지된다',uncertain=False),
                composition=dict(summary='인물을 중심으로 화면이 넓어진다',uncertain=True))
        elif schema is Activity:value=activity
        else:raise AssertionError('Unexpected inference request')
        assert 'start_frame' not in schema.model_json_schema().get('properties',{})
        return validate(schema.model_validate(value))


def test_window_summaries_resume_without_merging_or_action_timestamps(tracked,tmp_path):
    before=(tracked/'timeline.json').read_bytes();backend=FakeEvents();output=tmp_path/'events'
    result=analyze_segments(tracked,output,max_windows=1,backend=backend)
    assert result['status']=='paused'
    result=analyze_segments(tracked,output,backend=backend)
    assert result['status']=='complete'
    assert backend.calls==[(kind,start,end,count) for start,end,count in [(0,5,50),(5,10,50),(10,11,10)] for kind in ['Activity','SceneContext']]
    assert len(result['events'])==9
    people=[e for e in result['events'] if e['category']=='activity']
    assert [(e['start_sec'],e['end_sec']) for e in people]==[(0,5),(5,10),(10,11)]
    assert all(e['temporal_basis']=='window_summary' and e['time_basis']=='summary_window' for e in result['events'])
    assert all(len(e['window_ids'])==1 for e in result['events'])
    assert result['video']['event_scope']=='people_and_scene'
    assert (tracked/'timeline.json').read_bytes()==before
    summaries=read_json(output/'window_summaries.json')
    assert len(summaries)==3
    for summary in summaries:
        assert set(summary)=={'window_id','start_sec','end_sec','activities','background','composition'}
        assert set(summary['activities'][0])=={'entity_id','summary','visibility','uncertain','observation'}
        assert '양팔' in summary['activities'][0]['summary'] and '몸을 돌린다' in summary['activities'][0]['summary']
    by_person=read_json(output/'events_by_person.json')
    assert len(by_person['person_0001']['summaries'])==3
    calls=len(backend.calls);analyze_segments(tracked,output,backend=backend);assert len(backend.calls)==calls


def test_missing_person_is_requested_individually(tracked,tmp_path):
    backend=FakeEvents(missing=True)
    result=analyze_segments(tracked,tmp_path/'events',max_windows=1,backend=backend)
    assert [c[0] for c in backend.calls]==['Activity','SceneContext']
    assert result['windows'][0]['analysis']['activities'][0]['entity_id']=='person_0001'


def test_unknown_id_and_unobserved_claim_rejected():
    for eid in ['person_9999','person_0001']:
        a=Activity(entity_id=eid,summary='서 있다',visibility='visible',uncertain=False)
        with pytest.raises(ValueError):normalize_activity(a,['person_0001'],[{'observations':[]}])
    a=Activity(entity_id='P1',summary='일부 구간에서만 관찰됨',visibility='visible',uncertain=False)
    normalize_activity(a,['person_0001'],[{'observations':[{'entity_id':'person_0001'}]},{'observations':[]}])
    assert a.entity_id=='person_0001' and a.uncertain and a.visibility=='partially_visible'


def test_old_timed_action_schema_is_not_accepted():
    a=dict(entity_id='person_0001',summary='서 있다',visibility='visible',uncertain=False)
    for key,value in [('events',[]),('start_frame',0),('end_frame',49)]:
        with pytest.raises(ValueError):Activity.model_validate(dict(a,**{key:value}))


def test_failed_person_summary_is_explicitly_deferred(tracked,tmp_path):
    from low_freq_event.backend import ModelOutputError
    class Invalid(FakeEvents):
        def request(self,prompt,schema,**kwargs):
            if schema is Activity:raise ModelOutputError('invalid JSON')
            return super().request(prompt,schema,**kwargs)
    result=analyze_segments(tracked,tmp_path/'events',max_windows=1,backend=Invalid(missing=True))
    w=result['windows'][0]
    assert w['warnings'] and '판정 보류' in w['warnings'][0]
    assert w['analysis']['activities'][0]['uncertain']


def test_scene_summaries_do_not_create_or_retrack_entities(tracked,tmp_path):
    path=tracked/'timeline.json';data=read_json(path)
    data['entities'].append(dict(entity_id='object_0001',kind='object',label='넥타이',appearance='검정'))
    atomic_json(path,data)
    path=tracked/'tracking/frames.json';frames=read_json(path)
    for f in frames:f['observations'].append(dict(f['observations'][0],entity_id='object_0001',kind='object',detection_id='obj'))
    atomic_json(path,frames)
    class People(FakeEvents):
        def request(self,prompt,schema,**kwargs):
            assert 'object_0001' not in prompt
            return super().request(prompt,schema,**kwargs)
    result=analyze_segments(tracked,tmp_path/'events',max_windows=1,backend=People())
    assert [e['entity_id'] for e in result['entities']]==['person_0001']
    assert all(o['entity_id']=='person_0001' for f in result['tracks'] for o in f['observations'])
    assert [(e['category'],e['entity_ids']) for e in result['events']]==[('activity',['person_0001']),('background',[]),('composition',[])]


def test_plan_is_disjoint_and_keeps_short_tail(tracked):
    _,frames,plan=load_plan(tracked)
    assert [len(w['frames']) for w in plan]==[50,50,10]
    assert [(w['start_sec'],w['end_sec']) for w in plan]==[(0,5),(5,10),(10,11)]
    assert len({f['time_sec'] for w in plan for f in w['frames']})==len(frames)

def test_rejects_changed_settings_and_source_output_collision(tracked,tmp_path):
    output=tmp_path/'events';analyze_segments(tracked,output,max_windows=1,backend=FakeEvents())
    with pytest.raises(ValueError,match='변경'):analyze_segments(tracked,output,max_pixels=98304,backend=FakeEvents())
    with pytest.raises(ValueError,match='분리'):analyze_segments(tracked,tracked,backend=FakeEvents())

def test_plan_rejects_missing_tracking_frames(tracked):
    path=tracked/'tracking/frames.json';frames=read_json(path);del frames[4];atomic_json(path,frames)
    with pytest.raises(ValueError,match='누락'):load_plan(tracked)

def test_resized_pixel_boxes_match_normalized_overlay_coordinates():
    from low_freq_event.segment_events import event_context
    from low_freq_event.video import Clip,Frame
    from low_freq_event.schema import LocalEntity
    entities=[LocalEntity(local_id='person_0001',kind='person',label='인물 1',appearance='흰 옷')]
    ground=[dict(frame_index=0,boxes=[dict(local_id='person_0001',bbox=[100,200,900,800])])]
    for width,height in ((576,320),(416,224),(320,192)):
        clip=Clip(0,5,30,[Frame(0,Image.new('RGB',(width,height)))])
        context=event_context(clip,entities,ground)
        assert context['image_size']==[width,height]
        assert context['frames'][0][2][0]==['P1',round(width*.1),round(height*.2),round(width*.9),round(height*.8),0]
        assert ground[0]['boxes'][0]['bbox']==[100,200,900,800]


def test_observation_intervals_preserve_gaps_and_absolute_time():
    frames=[dict(time_sec=5+i/10,observations=[{'entity_id':'person_0001'}] if i in (2,3,6,7,8) else []) for i in range(10)]
    p=observation_profile(frames,'person_0001')
    assert p['observed_frames']==5
    assert not p['observed_at_window_start'] and not p['observed_at_window_end']
    assert [(r['first_observed_sec'],r['last_observed_sec']) for r in p['intervals']]==[(5.2,5.3),(5.6,5.8)]
    assert observation_profile(frames,'person_0002')['intervals']==[]


def test_separate_people_get_identical_full_video_and_presence_context(tracked,tmp_path):
    path=tracked/'timeline.json';d=read_json(path)
    for n in (2,3): d['entities'].append(dict(d['entities'][0],entity_id=f'person_{n:04d}'))
    atomic_json(path,d)
    path=tracked/'tracking/frames.json';frames=read_json(path)
    for i,f in enumerate(frames):
        if 15<=i<30 or 35<=i<50:f['observations'].append(dict(f['observations'][0],entity_id='person_0002',detection_id=f'p2_{i}'))
    atomic_json(path,frames)
    class Focused:
        def __init__(self):self.clips=[];self.ids=[]
        def request(self,prompt,schema,*,clip,validate,**kwargs):
            self.clips.append(clip)
            assert 'person_0001' in prompt and 'person_0002' in prompt
            assert 'first_observed_sec' in prompt and '1.5' in prompt and '3.5' in prompt
            if schema is Activity:
                eid='person_0001' if '분석 대상 entity_id: person_0001' in prompt else 'person_0002'
                self.ids.append(eid)
                assert '상호작용' in prompt and '따라 할 수' in prompt
                return validate(Activity(entity_id=eid,summary=eid+' 관찰된 자세를 유지한다',visibility='visible',uncertain=False))
            assert schema is SceneContext
            return validate(SceneContext(background={'summary':'배경 유지','uncertain':False},composition={'summary':'구도 유지','uncertain':False}))
    backend=Focused();result=analyze_segments(tracked,tmp_path/'events',max_windows=1,backend=backend)
    assert backend.ids==['person_0001','person_0002']
    assert len(backend.clips)==3 and all(c is backend.clips[0] for c in backend.clips)
    assert len(backend.clips[0].frames)==50
    assert backend.clips[0].frames[0].image.size==(96,64)
    people=result['windows'][0]['analysis']['activities']
    assert people[1]['observation']['observed_frames']==30
    assert len(people[1]['observation']['intervals'])==2
    assert people[2]['visibility']=='not_visible' and people[2]['uncertain']
    assert people[2]['observation']['observed_frames']==0


def test_edited_prompts_are_used_snapshotted_and_block_mixed_resume(tracked,tmp_path,monkeypatch):
    import low_freq_event.segment_events as module
    prompts=tmp_path/'prompts';prompts.mkdir()
    (prompts/'person.txt').write_text('PERSON CUSTOM INSTRUCTION')
    (prompts/'scene.txt').write_text('SCENE CUSTOM INSTRUCTION')
    monkeypatch.setattr(module,'PROMPT_DIR',prompts)
    class Custom(FakeEvents):
        def request(self,prompt,schema,**kwargs):
            assert ('PERSON CUSTOM INSTRUCTION' if schema is Activity else 'SCENE CUSTOM INSTRUCTION') in prompt
            return super().request(prompt,schema,**kwargs)
    out=tmp_path/'custom'
    analyze_segments(tracked,out,max_windows=1,backend=Custom())
    assert read_json(out/'prompts_used.json')['person']=='PERSON CUSTOM INSTRUCTION'
    (prompts/'person.txt').write_text('NEW INSTRUCTION')
    with pytest.raises(ValueError,match='変更|변경'):
        analyze_segments(tracked,out,backend=Custom())
    assert read_json(out/'prompts_used.json')['person']=='PERSON CUSTOM INSTRUCTION'
