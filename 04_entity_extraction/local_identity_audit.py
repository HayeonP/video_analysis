"""Bounded visual audit of large, simultaneous regions merged into one local ID.

Geometry only proposes questions. A physical split requires explicit visual
confirmation in at least two original source frames; body parts keep their owner.
"""
from itertools import combinations
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel,Field

VERSION=3
AUDIT_REASON_PREFIX='Focused visual audit v3 of original frames '


class SubjectReview(BaseModel):
    role:Literal['whole_subject','body_part','handled_object','worn_item','background','uncertain']
    kind:Literal['person','animal','object','unknown']
    description:str=Field(min_length=3)


class PairReview(BaseModel):
    decision:Literal['same','different','uncertain']
    left:SubjectReview
    right:SubjectReview
    evidence_frames:list[int]
    visible_evidence:str=Field(min_length=8)
    count_visible_bodies:Literal[1,2,'unknown']='unknown'
    same_basis:Literal['duplicate_region','visibly_connected_body_parts','reflection_or_display']|None=None
    visible_connection:str=''


def audit_candidates(result,shot,aliases,max_pairs=None):
    """Candidates are questions, never geometry-derived physical constraints."""
    tracks={aliases[t['track_id']]:t for t in shot['tracks']}
    area_frame=max(1,shot.get('width',1)*shot.get('height',1));candidates=[]
    known={tuple(sorted((r.left,r.right))) for r in result.relations
           if r.relation in ('same_individual','different_individual','part_of') and r.reason.startswith(AUDIT_REASON_PREFIX)}
    for entity in result.entities:
        for left,right in combinations(entity.track_ids,2):
            if tuple(sorted((left,right))) in known:continue
            # Explicitly segmented ownership must be checked in its actual span.
            def owned_rows(alias):
                spans=[s for s in entity.segments if s.track_id==alias]
                return {r['frame']:r for r in tracks[alias]['observations'] if
                    (not spans or any(s.first_frame<=r['frame']<=s.last_frame for s in spans)) and
                    r.get('evidence','observed_detection')=='observed_detection' and
                    r.get('tracker_confirmed') is not False and r.get('confidence',0)>=.24}
            a,b=owned_rows(left),owned_rows(right);support=[]
            for frame in a.keys()&b.keys():
                x,y=a[frame]['bbox'],b[frame]['bbox']
                ax=(x[2]-x[0])*(x[3]-x[1]);ay=(y[2]-y[0])*(y[3]-y[1])
                if min(ax,ay)/area_frame<.015:continue
                intersection=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
                iou=intersection/max(ax+ay-intersection,1e-9);containment=intersection/max(min(ax,ay),1e-9)
                if iou<.30 and containment<.85:
                    support.append(dict(frame=frame,left_observation=a[frame],right_observation=b[frame],
                                        iou=iou,minimum_area=min(ax,ay)))
            if len(support)<3:continue
            support.sort(key=lambda r:(r['iou'],-r['minimum_area'],r['frame']))
            candidates.append(dict(left=left,right=right,local_id=entity.local_id,support=support,
                rank=(max(r['minimum_area'] for r in support)*len(support))))
    candidates.sort(key=lambda r:r['rank'],reverse=True)
    return candidates if max_pairs is None else candidates[:max_pairs]


def validate_pair(review,frames,overlaps=None):
    if not set(review.evidence_frames)<=set(frames):raise ValueError('Pair evidence must use supplied original frames.')
    if review.decision in ('same','different') and len(set(review.evidence_frames))<2:
        raise ValueError('A same/different decision requires visible evidence in at least two supplied frames.')
    if review.decision=='different' and any(s.role in ('background','uncertain') for s in (review.left,review.right)):
        raise ValueError('Uncertain/background regions cannot establish two distinct physical subjects.')
    if (review.decision=='different' and all(s.role=='whole_subject' for s in (review.left,review.right))
            and review.count_visible_bodies!=2):
        raise ValueError('Two different whole subjects require two independently visible bodies; a single visible body or hidden fragment is insufficient.')
    if review.decision=='same':
        if review.same_basis is None or len(review.visible_connection.strip())<12:
            raise ValueError('Same owner requires a visible physical connection: duplicate region, connected anatomy, or an actual reflection/display. Clothing similarity is insufficient.')
        if review.count_visible_bodies==2 and review.same_basis!='reflection_or_display':
            raise ValueError('Two bodies simultaneously visible in one camera frame cannot be one person unless an actual reflection/display is visible.')
        if review.count_visible_bodies=='unknown':
            raise ValueError('Uncertain body count cannot support same; inspect source anatomy or return uncertain.')
        both_whole=all(s.role=='whole_subject' for s in (review.left,review.right))
        if review.same_basis=='duplicate_region' and both_whole and overlaps is not None and max(overlaps,default=0)<.30:
            raise ValueError('Low-overlap complete bodies are not duplicate image regions. The crops are magnifications of the SAME instant, not alternate camera views.')
        if review.same_basis=='visibly_connected_body_parts' and both_whole:
            raise ValueError('Connected body parts require an actual body-part role and visible anatomical connection; two whole subjects are not a part/owner pair.')
    return review


def audit_local_identity(result,bundle,shot,aliases,output,backend,state,*,max_calls=4):
    from story_semantics import BudgetExhausted,TrackRelation
    from low_freq_event.backend import ModelOutputError
    from local_consistency import separate_conflicting_subjects
    from story_relationship_images import _source_frames,_box,_font,COLORS
    from PIL import ImageDraw
    output=Path(output);folder=output/'local_identity_audit'/shot['shot_id'];folder.mkdir(parents=True,exist_ok=True)
    result=result.model_copy(deep=True);candidates=audit_candidates(result,shot,aliases);records=[];subject_updates={}
    reverse={v:k for k,v in aliases.items()}
    for number,candidate in enumerate(candidates):
        left,right=candidate['left'],candidate['right'];items=[]
        if number>=4:
            records.append(dict(left=left,right=right,status='pending',reason='Per-shot comparison cap of four pairs'))
            for entity in result.entities:
                if left in entity.track_ids or right in entity.track_ids:entity.uncertain=True
            continue
        for support in candidate['support'][:3]:
            items.append(dict(support,left=reverse[left],right=reverse[right],shared_frames=len(candidate['support']),shared_mask_frames=0))
        frames=_source_frames(bundle,shot,output,items);images=[]
        for item in items:
            if item['frame'] not in frames:continue
            # Exactly one camera image per input. Repeated crops in a montage
            # were empirically mistaken for alternate views of one person.
            panel=frames[item['frame']].copy();draw=ImageDraw.Draw(panel)
            font=_font(max(18,round(panel.width/70)))
            for alias,color,key in zip((left,right),COLORS,('left_observation','right_observation')):
                box=_box(item[key],panel.size)
                draw.rectangle(box,outline=color,width=max(3,round(panel.width/450)))
                draw.text((box[0],max(0,box[1]-font.size-5)),alias,fill=color,font=font,
                          stroke_width=1,stroke_fill='black')
            draw.text((12,10),f"원본 프레임 {item['frame']} / 청록 {left} / 주황 {right}",
                      font=font,fill='white',stroke_width=1,stroke_fill='black')
            panel.save(folder/f'pair_{number:02d}_f{item["frame"]:08d}.jpg',quality=95)
            images.append((f'원본 프레임 {item["frame"]}: 청록 {left}, 주황 {right}',panel))
        visible=sorted({item['frame'] for item in items if item['frame'] in frames})
        record=dict(left=left,right=right,candidate_frames=sorted(r['frame'] for r in candidate['support']),shown_frames=visible,status='pending')
        if len(visible)<2:
            record['reason']='insufficient original frame evidence';records.append(record);continue
        prompt=(f'동시 등장 대상 검수 VERSION {VERSION}\n'
            '각 이미지는 카메라 한 대가 찍은 원본 프레임 한 장입니다. 같은 이미지 안의 청록색 상자와 주황색 상자는 같은 시각에 존재합니다. 복제된 crop이나 다른 카메라 각도가 아닙니다. '
            '먼저 두 색 상자가 가리키는 주된 대상의 독립적인 몸이 총 몇 개인지 세고, 각각의 위치와 머리·얼굴·몸의 외형을 따로 설명하세요. '
            '서로 다른 머리와 몸 두 개가 동시에 보이면 서로 다른 대상입니다. 앞모습 한 명과 뒷모습 한 명도 동시에 보이면 두 명입니다. 옷이나 체형이 비슷하다는 이유로 한 명이라고 판단하지 마세요. '
            '동일 몸의 중복 검출, 한 몸에 실제로 연결된 신체 부분, 눈에 보이는 거울/화면의 반사라면 same이 가능합니다. 실제 연결 부위를 설명해야 합니다. '
            '최소 두 원본 프레임에서 확인한 근거로 same/different를 결정하고 근거가 부족하면 uncertain으로 남기세요. 영화 이름이나 인물 이름은 추측하지 마세요. '
            'JSON 하나만 반환하세요. 필드: count_visible_bodies(관찰한 몸 개수 정수 1 또는 2, 판단 불가시 "unknown"), '
            'left/right(각각 role,kind,description을 가진 객체), decision(same/different/uncertain), evidence_frames(확인한 원본 프레임 번호 2개 이상), '
            'visible_evidence(양쪽 대상의 위치와 외형 및 동일/다른 판정 근거), '
            'same_basis(same일 때만 duplicate_region/visibly_connected_body_parts/reflection_or_display 중 하나, 아니면 null), '
            'visible_connection(same일 때 실제 물리적 연결 설명, 아니면 빈 문자열). '
            'role은 whole_subject/body_part/handled_object/worn_item/background/uncertain, kind는 person/animal/object/unknown입니다. '
            'left는 청록 상자, right는 주황 상자입니다.\n'+
            json.dumps(dict(shot_id=shot['shot_id'],left=left,right=right,frames=visible),ensure_ascii=False,sort_keys=True))
        before=backend.ledger['generations'];phase=backend.phase;reserve=backend.reserve
        backend.phase=f'local_identity_audit:{shot["shot_id"]}'
        backend.reserve=max(reserve,backend.remaining-max(0,max_calls-state.get('generations',0)))
        try:
            review=backend.request(prompt,PairReview,images=images,max_tokens=1800,image_pixels=1048576,
                                   validate=lambda r:validate_pair(r,visible,[item['iou'] for item in items]))
            record.update(status=review.decision,review=review.model_dump())
            if review.decision in ('same','different'):
                relation='same_individual' if review.decision=='same' else 'different_individual'
                result.relations.append(TrackRelation(left=left,right=right,relation=relation,
                    reason=f'{AUDIT_REASON_PREFIX}{review.evidence_frames}: {review.visible_evidence}'))
                for alias,subject in ((left,review.left),(right,review.right)):
                    result.track_roles[alias]=subject.role;result.track_descriptions[alias]=subject.description
                    subject_updates[alias]=(subject,review.evidence_frames,review.visible_evidence)
            else:
                for entity in result.entities:
                    if left in entity.track_ids or right in entity.track_ids:entity.uncertain=True
        except (ModelOutputError,BudgetExhausted) as exc:
            record.update(status='pending',reason=str(exc))
            for entity in result.entities:
                if left in entity.track_ids or right in entity.track_ids:entity.uncertain=True
        finally:
            state['generations']=state.get('generations',0)+backend.ledger['generations']-before
            backend.phase=phase;backend.reserve=reserve
        records.append(record)
    unresolved=[(r['left'],r['right']) for r in records if r['status'] in ('uncertain','pending')]
    result=separate_conflicting_subjects(result,shot,aliases,provisional_pairs=unresolved)
    # A review observes two specific tracks, not every track in their original
    # owner groups. Never spread one track's corrected species over its siblings.
    from types import SimpleNamespace
    from local_classification_audit import apply_classification
    classification_proposals=[]
    for alias,(subject,frames,evidence) in subject_updates.items():
        owner=next((e for e in result.entities if alias in e.track_ids),None)
        if owner and subject.kind!=owner.kind:
            # Two colored targets can be confused by the model. A category
            # change needs its own single-target source-image confirmation.
            classification_proposals.append(dict(alias=alias,previous_kind=owner.kind,proposed_kind=subject.kind))
            owner.uncertain=True
            continue
        classified=SimpleNamespace(**subject.model_dump(),evidence_frames=frames,visible_evidence=evidence)
        result=apply_classification(result,alias,classified,reason_prefix=f'Focused physical audit v{VERSION}')
    report=dict(version=VERSION,shot_id=shot['shot_id'],max_generation_calls_per_run=max_calls,
                generations_used_in_run=state.get('generations',0),pairs=records,
                classification_proposals=classification_proposals,
                provisional_fragments=[list(pair) for pair in unresolved],
                provisional_policy='Unresolved pairs are separate hypotheses, not confirmed different identities; global review may reunify them.',
                status='needs_review' if any(r['status'] in ('uncertain','pending') for r in records) else 'complete')
    (folder/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    shot['local_identity_audit']=report
    return result
