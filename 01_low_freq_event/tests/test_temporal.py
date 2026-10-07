import pytest

from low_freq_event.schema import LocalEntity, TemporalEvidence
from low_freq_event.temporal import events_from_evidence


def fact(action, *ids):
    return {'action':action,'entity_ids':list(ids),'description':action,'uncertain':False}


def test_pose_and_motion_have_separate_spans_and_do_not_bridge_gaps():
    entities = [LocalEntity(local_id='p',label='인물',kind='person',appearance='옷')]
    frames = []
    for i in range(5):
        frames.append({'frame_index':i,
            'states':[fact('arms_raised' if i in {1,2,4} else 'arms_lowered','p')],
            'transitions_from_previous':[fact('raise_arm','p')] if i in {1,4} else []})
    scan = events_from_evidence(TemporalEvidence.model_validate({'frames':frames}),entities,[{'p'}]*5)
    spans = lambda action:[(e.start_frame,e.end_frame) for e in scan.events if e.action==action]
    assert spans('arms_raised') == [(1,2),(4,4)]
    assert spans('raise_arm') == [(0,1),(3,4)]
    assert spans('visible') == [(0,4)]  # Sustained states must remain possible.
    assert all(not e.continues_before and not e.continues_after for e in scan.events)
    for e in scan.events:
        if e.action == 'raise_arm':
            assert e.temporal_basis == 'sampled_transition_bracket'


def test_each_participant_must_have_evidence_for_the_whole_merged_interval():
    entities = [LocalEntity(local_id=eid,label=eid,kind='person',appearance='옷') for eid in ['p','q']]
    evidence = TemporalEvidence.model_validate({'frames':[
        {'frame_index':0,'states':[fact('arms_raised','p','q')],'transitions_from_previous':[]},
        {'frame_index':1,'states':[fact('arms_raised','p')],'transitions_from_previous':[]},
        {'frame_index':2,'states':[fact('arms_raised','p','q')],'transitions_from_previous':[fact('raise_arm','q')]},
    ]})
    result = events_from_evidence(evidence,entities,[{'p','q'},{'p'},{'p','q'}])
    raised = [(e.entity_ids,e.start_frame,e.end_frame) for e in result.events if e.action=='arms_raised']
    assert (['p'],0,2) in raised
    assert (['q'],0,0) in raised and (['q'],2,2) in raised
    assert not any(e.action=='raise_arm' for e in result.events)  # Previous q was not visible.


def test_unknown_pose_is_preserved_as_presence_without_inventing_an_action():
    entity = LocalEntity(local_id='p',label='인물',kind='person',appearance='가려짐')
    evidence = TemporalEvidence.model_validate({'frames':[
        {'frame_index':0,'states':[dict(fact('unknown_pose','p'),uncertain=True)],'transitions_from_previous':[]}]})
    result = events_from_evidence(evidence,[entity],[{'p'}])
    assert [e.action for e in result.events] == ['visible']


def test_unchanged_pose_or_opposite_direction_does_not_support_arm_movement():
    entity=LocalEntity(local_id='p',label='인물',kind='person',appearance='옷')
    for poses, movement in [(['hand_on_hip','hand_on_hip'],'lower_arm'),(['arms_lowered','arms_raised'],'lower_arm')]:
        evidence=TemporalEvidence.model_validate({'frames':[
            {'frame_index':0,'states':[fact(poses[0],'p')],'transitions_from_previous':[]},
            {'frame_index':1,'states':[fact(poses[1],'p')],'transitions_from_previous':[fact(movement,'p')]}]})
        result=events_from_evidence(evidence,[entity],[{'p'},{'p'}])
        assert not any(e.action==movement for e in result.events)
        assert evidence.frames[1].warnings


def test_evidence_uses_final_identity_not_window_local_label():
    from PIL import Image
    from low_freq_event.pipeline import window_events
    from low_freq_event.video import Clip, Frame
    entity = LocalEntity(local_id='local_1',label='임시 인물',kind='person',appearance='옷')
    evidence = TemporalEvidence.model_validate({'frames':[
        {'frame_index':0,'states':[fact('arms_raised','local_1')],'transitions_from_previous':[]}]})
    scan = events_from_evidence(evidence,[entity],[{'local_1'}])
    clip = Clip(0,1,30,[Frame(0,Image.new('RGB',(32,32)),'f.jpg')])
    obs = [{'observation_id':'o1','local_id':'local_1','frame_index':0}]
    exported = window_events(scan,clip,obs,{'o1':'person_0002'},'w')
    assert all(e['entity_ids']==['person_0002'] for e in exported)
    assert all(f['entity_id']=='person_0002' and f['local_id']=='local_1' for e in exported for f in e['temporal_evidence'])


def test_window_candidates_are_verified_with_adjacent_frames(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from low_freq_event.backend import QwenBackend
    from low_freq_event.config import Settings
    from low_freq_event.video import Clip, Frame
    entity=LocalEntity(local_id='person_0002',label='인물 2',kind='person',appearance='옷')
    clip=Clip(10,12,30,[Frame(10,Image.new('RGB',(32,32))),Frame(10+2/3,Image.new('RGB',(32,32)),'f.jpg')])
    backend=QwenBackend(Settings(perception='person_reid',json_retries=0),tmp_path/'calls')
    seen=[]
    def generate(prompt, video, images, *args):
        assert prompt.index('</entity_context>') < prompt.index('OUTPUT TASK')
        seen.append((video is not None,[label for label,_ in images]))
        if video is not None:
            return json.dumps({'actions':['raise_arm','turn']})
        has_before=len(images)==2
        return json.dumps({'frame_index':1,'states':[{'entity_id':'P2','action':'arms_raised' if has_before else 'arms_lowered','uncertain':False}],
            'transitions_from_previous':[{'entity_id':'P2','action':'raise_arm','uncertain':False}] if has_before else []})
    monkeypatch.setattr(backend,'_generate',generate)
    result=backend.scan(clip,[entity],[{'boxes':[{'local_id':'person_0002','bbox':[0,0,500,900]}]}]*2)
    assert seen==[(True,[]),(False,['NOW (NO BEFORE FRAME)']),(False,['BEFORE','NOW'])]
    assert result.action_candidates==['raise_arm','turn']
    assert not any(e.action=='turn' for e in result.events)  # Unverified window candidate is not an event.
    event=next(e for e in result.events if e.action=='raise_arm')
    assert event.entity_ids==['person_0002'] and (event.start_frame,event.end_frame)==(0,1)
    assert [f['frame_index'] for f in result.frame_facts]==[0,1]


def test_empty_and_background_frames_keep_presence_without_requesting_poses(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from low_freq_event.backend import QwenBackend
    from low_freq_event.config import Settings
    from low_freq_event.video import Clip, Frame

    entity = LocalEntity(local_id='mountain', label='설산', kind='environment', appearance='눈')
    clip = Clip(0, 2, 24, [Frame(i * 2 / 3, Image.new('RGB', (32, 32))) for i in range(3)])
    backend = QwenBackend(Settings(json_retries=0), tmp_path / 'calls')
    calls = []

    def generate(prompt, video, images, *args):
        assert video is not None, 'No subject is available for a pose comparison'
        calls.append(prompt)
        return json.dumps({'actions': []})

    monkeypatch.setattr(backend, '_generate', generate)
    grounded = [{'boxes': []}] + [{'boxes': [{'local_id': 'mountain', 'bbox': [0, 0, 1000, 1000]}]}] * 2
    result = backend.scan(clip, [entity], grounded)
    assert calls == []  # Background presence needs no foreground-action hints.
    assert [(e.action, e.start_frame, e.end_frame, e.entity_ids) for e in result.events] == [
        ('visible', 1, 2, ['mountain'])]
    assert [f['frame_index'] for f in result.frame_facts] == [0, 1, 2]
    assert all(not f['states'] and not f['transitions_from_previous'] for f in result.frame_facts)


@pytest.mark.parametrize('wrong_reply', ['truncated', 'input_metadata'])
def test_invalid_optional_candidates_keep_independent_frame_evidence(tmp_path, monkeypatch, wrong_reply):
    import json
    from pathlib import Path
    from PIL import Image
    from low_freq_event.backend import QwenBackend
    from low_freq_event.config import Settings
    from low_freq_event.video import Clip, Frame
    bad = ((Path(__file__).parent/'fixtures/truncated_action_context.txt').read_text() if wrong_reply=='truncated'
           else json.dumps({'entities': [], 'frames': []}))
    backend=QwenBackend(Settings(json_retries=2),tmp_path/'calls')
    entity=LocalEntity(local_id='person_0002',label='인물',kind='person',appearance='옷')
    clip=Clip(0,2,30,[Frame(i*2/3,Image.new('RGB',(32,32))) for i in range(2)])
    calls=[]
    def generate(prompt, video, images, *args):
        calls.append(video is not None)
        assert prompt.index('</entity_context>') < prompt.index('OUTPUT TASK')
        if video is not None:
            return bad
        after=len(images)==2
        return json.dumps({'frame_index':1,'states':{'P2':['arms_raised' if after else 'arms_lowered']},
                           'transitions_from_previous':{'P2':['raise_arm']} if after else {}})
    monkeypatch.setattr(backend,'_generate',generate)
    scan=backend.scan(clip,[entity],[{'boxes':[{'local_id':'person_0002','bbox':[0,0,500,900]}]}]*2)
    assert calls==[True,True,True,False,False]
    assert scan.action_candidates==[] and len(scan.warnings)==1
    assert '후보 없이' in scan.warnings[0]
    movement=[e for e in scan.events if e.action=='raise_arm']
    assert len(movement)==1 and (movement[0].start_frame,movement[0].end_frame)==(0,1)
    assert not movement[0].uncertain  # Independent valid evidence remains valid.
    assert len(list((tmp_path/'calls').glob('*.json')))==5


def test_runtime_failure_is_not_hidden_as_optional_candidate_failure(tmp_path, monkeypatch):
    from PIL import Image
    from low_freq_event.backend import QwenBackend
    from low_freq_event.config import Settings
    from low_freq_event.video import Clip, Frame
    backend=QwenBackend(Settings(),tmp_path/'calls')
    def generate(*args): raise RuntimeError('CUDA failure')
    monkeypatch.setattr(backend,'_generate',generate)
    entity=LocalEntity(local_id='person_0001',label='인물',kind='person',appearance='옷')
    clip=Clip(0,1,24,[Frame(0,Image.new('RGB',(32,32)))])
    with pytest.raises(RuntimeError,match='CUDA failure'):
        backend.scan(clip,[entity],[{'boxes':[{'local_id':'person_0001','bbox':[0,0,500,900]}]}])
