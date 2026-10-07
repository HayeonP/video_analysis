"""Build event intervals only from consecutive, explicit sampled-frame evidence."""
from collections import defaultdict
import json
import logging

from .backend import ModelOutputError
from .prompts import PAIR_TEMPORAL, WINDOW_ACTIONS
from .schema import ActionVocabulary, Candidate, FrameFact, FrameFacts, ScanResult, TemporalEvidence
from .context import annotated_clip, context_prompt, entity_token

LOG = logging.getLogger(__name__)


def temporal_prompt(clip, entities, grounded, instruction):
    return ('The following block is reference INPUT, not an output example. Do not repeat it.\n'
            '<entity_context>' + context_prompt(clip, entities, grounded) + '\n</entity_context>\n\n'
            'OUTPUT TASK (return only the JSON requested below):\n' + instruction)

ACTION_LABELS = {
    'visible':'대상이 보임','stand':'서 있는 자세','sit':'앉아 있음','hold':'물건을 들고 있음',
    'arms_raised':'팔이 올라가 있음','arms_lowered':'팔이 내려가 있음','arms_extended':'팔을 옆으로 뻗고 있음',
    'hand_on_hip':'손을 허리에 대고 있음','hands_on_hips':'양손을 허리에 대고 있음',
    'arms_bent':'팔을 굽히고 있음','raise_arm':'팔을 올림','lower_arm':'팔을 내림',
    'turn':'몸 방향을 바꿈','walk':'걸음','dance':'춤 동작이 이어짐','jump':'뜀',
    'bend':'몸을 숙임','clap':'손뼉을 침','wave':'손을 흔듦','step':'발을 옮김',
    'facing_front':'정면을 향함','facing_back':'뒤를 향함','facing_side':'옆을 향함',
    'crouching':'몸을 낮춘 자세','kneeling':'무릎을 꿇은 자세',
}


def extract_frame_events(backend, clip, entities, grounded):
    visible = [{box['local_id'] for box in frame['boxes']} for frame in grounded]
    annotated = annotated_clip(clip, grounded)
    names = {e.local_id:entity_token(e.local_id) for e in entities}
    reverse = {token:eid for eid,token in names.items()}
    objects = {names[e.local_id]:e.label for e in entities if e.kind not in {'person','character','environment'}}
    people = {e.local_id for e in entities if e.kind in {'person','character'}}
    subjects = {e.local_id for e in entities if e.kind != 'environment'}
    candidates = ActionVocabulary(actions=[])
    warnings = []
    # Coarse candidates are hints only. Their formatting failure must not
    # prevent the independent, required frame-evidence pass.
    if any(ids & subjects for ids in visible):
        try:
            candidates = backend.request(temporal_prompt(clip, entities, grounded, WINDOW_ACTIONS),
                ActionVocabulary, clip=annotated, max_tokens=384)
        except ModelOutputError as exc:
            warning = '윈도 행동 후보의 JSON 검증 실패: 후보 없이 프레임별 상태·변화 분석을 계속했습니다. 원문은 model_calls에 보존됩니다.'
            warnings.append(warning)
            LOG.warning('%s (%s)', warning, exc)
    frames=[]
    for i, frame in enumerate(annotated.frames):
        current = {names[eid] for eid in visible[i] & subjects}
        previous = {names[eid] for eid in visible[i-1] & subjects} if i else set()
        if not current:
            # No localized foreground subject can support a pose or transition.
            # Background presence is still retained by events_from_evidence.
            frames.append(FrameFacts(frame_index=i, states=[], transitions_from_previous=[]))
            continue
        prompt = PAIR_TEMPORAL.format(actions=json.dumps(sorted(set(candidates.actions))),
            current_ids=json.dumps(sorted(current)),previous_ids=json.dumps(sorted(previous)),
            objects=json.dumps(objects,ensure_ascii=False))
        pair_indices = [i-1, i] if i else [i]
        from .video import Clip
        pair = Clip(clip.start_sec, clip.end_sec, clip.source_fps, [clip.frames[j] for j in pair_indices])
        prompt = temporal_prompt(pair, entities, [grounded[j] for j in pair_indices], prompt)
        images = [('BEFORE',annotated.frames[i-1].image)] if i else []
        images.append(('NOW' if i else 'NOW (NO BEFORE FRAME)',frame.image))

        def validate(result):
            if result.frame_index != 1:
                raise ValueError('frame_index must be 1, the NOW image; not a time in seconds')
            if i == 0 and result.transitions_from_previous:
                raise ValueError('No BEFORE image: transitions must be empty')
            described = {eid for f in result.states for eid in f.entity_ids}
            required = {names[eid] for eid in people & visible[i]}
            for missing in sorted(required-described):
                # Missing pose output is unknown, never evidence that the
                # person stood still or shared another person's action.
                result.states.append(FrameFact(entity_ids=[missing],action='unknown_pose',
                    description='해당 인물의 자세 판정이 모델 응답에서 누락됨',uncertain=True))
            for kind, facts in (('state',result.states),('transition',result.transitions_from_previous)):
                allowed = current if kind=='state' else current & previous
                seen=set()
                for fact in facts:
                    if not set(fact.entity_ids)<=set(reverse) or len(fact.entity_ids)!=len(set(fact.entity_ids)):
                        raise ValueError(f'{kind}: IDs must be unique supplied IDs: {sorted(reverse)}')
                    if unsupported := set(fact.entity_ids)-allowed:
                        result.warnings.append(f'{kind}: bbox 근거가 없는 ID의 판정은 이벤트로 사용하지 않음: {sorted(unsupported)}')
                    for eid in fact.entity_ids:
                        if (eid,fact.action) in seen:
                            raise ValueError('Duplicate entity/action; combine or remove duplicate observations')
                        seen.add((eid,fact.action))
                    if kind=='state' and fact.action in {'raise_arm','lower_arm','turn','walk','dance'}:
                        raise ValueError(f'{fact.action} is a CHANGE. Use a pose code for states, e.g. arms_raised or facing_back')
            return result

        result = backend.request(prompt,FrameFacts,images=images,validate=validate,
            max_tokens=1536,image_pixels=backend.settings.max_pixels).model_copy(deep=True)
        result.frame_index=i
        for fact in [*result.states,*result.transitions_from_previous]:
            fact.entity_ids=[reverse[eid] for eid in fact.entity_ids]
        frames.append(result)
    evidence=TemporalEvidence(frames=frames)
    result=events_from_evidence(evidence,entities,visible)
    result.action_candidates=candidates.actions
    result.frame_facts=evidence.model_dump(mode='json')['frames']
    result.warnings=warnings
    return result


def events_from_evidence(evidence, entities, visible):
    """Never bridge a missing/negative sample or different participants' spans."""
    by_id = {e.local_id:e for e in entities}
    poses = [{eid:{f.action for f in frame.states if eid in f.entity_ids}
              for eid in visible[i]} for i,frame in enumerate(evidence.frames)]
    arm_poses={'hand_on_hip','hands_on_hips','arms_lowered','arms_raised','arms_extended','arms_bent'}
    orientations={'facing_front','facing_back','facing_side'}
    facts = defaultdict(dict)
    for frame in evidence.frames:
        i = frame.frame_index
        # Presence is derived from actual detections, not forced action guesses.
        for eid in visible[i]:
            facts[('state',eid,'visible')][i] = {
                'frame_index':i,'description':'bbox에서 대상이 관찰됨','uncertain':False,'source':'bbox'}
        for kind, items in (('state',frame.states), ('transition',frame.transitions_from_previous)):
            for fact in items:
                if fact.action in {'visible', 'unknown_pose', 'none', 'no_change'}:
                    continue
                for eid in fact.entity_ids:
                    # A transition requires both endpoints. Never turn a
                    # missing/occluded observation into evidence of motion.
                    if eid not in visible[i] or (kind == 'transition' and (i == 0 or eid not in visible[i-1])):
                        continue
                    before, after = set(), set()
                    uncertain = fact.uncertain
                    if kind == 'transition':
                        if fact.action in arm_poses | orientations | {'stand','sit','hold','crouching'}:
                            frame.warnings.append(f'{eid}: {fact.action}는 변화가 아닌 자세이므로 동작에서 제외')
                            continue
                        relevant = arm_poses if fact.action in {'raise_arm','lower_arm'} else orientations if fact.action=='turn' else set()
                        before = poses[i-1].get(eid,set()) & relevant
                        after = poses[i].get(eid,set()) & relevant
                        if relevant and before and after and before==after:
                            frame.warnings.append(f'{eid}: {fact.action} 판정과 달리 두 샘플의 자세 코드가 같아 변화 미확정')
                            continue
                        if relevant and (not before or not after):
                            uncertain=True
                        if fact.action=='raise_arm' and before=={'arms_raised'} and after=={'arms_lowered'} or fact.action=='lower_arm' and before=={'arms_lowered'} and after=={'arms_raised'}:
                            frame.warnings.append(f'{eid}: {fact.action}와 두 샘플의 팔 높이 변화 방향이 충돌')
                            continue
                    facts[(kind,eid,fact.action)][i] = {
                        'frame_index':i,'description':fact.description or ACTION_LABELS.get(fact.action,fact.action),
                        'uncertain':uncertain,'source':'qwen_frame_comparison',
                        'state_before':sorted(before),'state_after':sorted(after)}
    grouped = defaultdict(list)
    for (kind,eid,action), samples in facts.items():
        runs = []
        for i in sorted(samples):
            if not runs or i != runs[-1][-1]+1:
                runs.append([])
            runs[-1].append(i)
        for run in runs:
            start = run[0] - (kind == 'transition')
            end = run[-1]
            grouped[(kind,action,start,end)].append((eid,[samples[i] for i in run]))
    events=[]
    for (kind,action,start,end), participants in sorted(grouped.items(), key=lambda item:(item[0][2],item[0][3],item[0][0],item[0][1])):
        ids = sorted(eid for eid,_ in participants)
        items = [dict(sample, entity_id=eid) for eid,samples in participants for sample in samples]
        description = '대상이 보임' if action == 'visible' else participants[0][1][0]['description']
        basis = 'sampled_transition_bracket' if kind == 'transition' else 'sampled_state_chain'
        note = ('인접 샘플 사이의 변화가 관찰된 구간이며 정확한 동작 시작·종료 시각은 미확정입니다.' if kind == 'transition'
                else '같은 상태가 확인된 연속 샘플 구간입니다. 샘플 사이 또는 윈도 밖으로 지속 시간을 연장하지 않습니다.')
        if start == end:
            note = '한 샘플에서만 확인한 상태입니다. 실제 지속 시간은 미확정입니다.'
        events.append(Candidate(event_type='action' if kind == 'transition' else 'ongoing', action=action,
            description=', '.join(by_id[eid].label for eid in ids)+' · '+description,
            entity_ids=ids,start_frame=start,end_frame=end,continues_before=False,continues_after=False,
            uncertain=any(s['uncertain'] for s in items),boundary_note=note,
            temporal_basis=basis,temporal_evidence=items))
    return ScanResult(entities=entities,events=events).validate_clip(len(evidence.frames))
