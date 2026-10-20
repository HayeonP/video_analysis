"""Small visual reviews of strong detector/semantic category disagreements.

Detector votes only nominate a question. Only source-image VLM evidence may
change a classification, and no identity annotation is read by this module.
"""
import json
from pathlib import Path
from statistics import mean, median
from typing import Literal
from pydantic import BaseModel, Field, field_validator

VERSION=2
REVIEW_PROTOCOL_VERSION=2
ANIMAL_CATEGORIES=frozenset({'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe'})


class ClassificationReview(BaseModel):
    kind:Literal['person','animal','object','unknown']
    role:Literal['whole_subject','body_part','handled_object','worn_item','background','uncertain']
    description:str=Field(min_length=8)
    evidence_frames:list[int]
    visible_evidence:str=Field(min_length=12)

    @field_validator('visible_evidence',mode='before')
    @classmethod
    def join_evidence_phrases(cls,value):
        # Formatting repair only: retain every supplied phrase verbatim and
        # refuse objects, numbers, empty lists or empty/non-string entries.
        if isinstance(value,list) and value and all(isinstance(x,str) and x.strip() for x in value):
            return '; '.join(value)
        return value


def classification_candidates(result,shot,aliases):
    """Recompute evidence inside each segment; never trust parent-track votes."""
    owners={t:e for e in result.entities for t in e.track_ids}
    proposals={r['alias']:r for r in shot.get('local_identity_audit',{}).get('classification_proposals',[])}
    candidates=[]
    for track in shot['tracks']:
        alias=aliases[track['track_id']];owner=owners.get(alias)
        # Explicit semantic intervals need interval-specific evidence; do not
        # overwrite several owners with one whole-track category decision.
        if any(s.track_id==alias for e in result.entities for s in e.segments):continue
        rows=[r for r in track['observations'] if r.get('evidence')=='observed_detection'
              and r.get('tracker_confirmed') is not False and r.get('confidence',0)>=.5]
        people=[r for r in rows if r.get('category')=='person']
        detector_kind='animal' if owner and owner.kind=='person' else 'person'
        support=([r for r in rows if r.get('category') in ANIMAL_CATEGORIES] if detector_kind=='animal' else people)
        proposed=alias in proposals
        if proposed:support=rows
        if proposed:
            if len({r['frame'] for r in support})<2:continue
        elif len(support)<6 or len(support)/max(1,len(rows))<.85:continue
        confidence=mean(r['confidence'] for r in support)
        area=median((r['bbox'][2]-r['bbox'][0])*(r['bbox'][3]-r['bbox'][1]) for r in support)/max(1,shot['width']*shot['height'])
        # Small animals need less image area than a person, but the same repeated
        # observed-detection support. Classification still requires visual review.
        if not proposed and (confidence<.65 or area<(.01 if detector_kind=='animal' else .04)):continue
        support.sort(key=lambda r:r['frame'])
        candidates.append(dict(alias=alias,track_id=track['track_id'],previous_kind=owner.kind if owner else 'rejected',
            observations=support,detector_kind=detector_kind,support_observations=len(support),
            trigger='paired_review_disagreement' if proposed else 'detector_semantic_disagreement',
            person_observations=len(people),eligible_observations=len(rows),
            mean_confidence=confidence,median_area_fraction=area,rank=area*len(support)*confidence))
    return sorted(candidates,key=lambda r:r['rank'],reverse=True)


def validate_classification(review,frames):
    if not set(review.evidence_frames)<=set(frames):raise ValueError('Classification evidence must cite supplied source frames.')
    if review.kind!='unknown' and review.role!='uncertain' and len(set(review.evidence_frames))<2:
        raise ValueError('Confirmed classification needs two supplied source frames.')
    return review


def apply_classification(result,alias,review,*,reason_prefix=None):
    """Reclassify only the reviewed track, preserving every other assignment."""
    from story_semantics import LocalEntity
    result=result.model_copy(deep=True)
    if review.kind=='unknown' or review.role in ('uncertain','background','worn_item'):
        for entity in result.entities:
            if alias in entity.track_ids:entity.uncertain=True
        return result
    owner=next((e for e in result.entities if alias in e.track_ids),None)
    result.track_roles[alias]=review.role;result.track_descriptions[alias]=review.description
    reason=(reason_prefix or f'Focused classification audit v{VERSION}')+f', source frames {review.evidence_frames}: {review.visible_evidence}'
    if owner and len(owner.track_ids)==1:
        owner.kind=review.kind;owner.label=review.description;owner.appearance=review.description
        owner.reason=reason;owner.uncertain=False
    elif owner and owner.kind==review.kind:
        # The owner grouping stays a separate question from species/role.
        owner.appearance='; '.join(result.track_descriptions.get(t,'') for t in owner.track_ids)
    else:
        if owner:
            owner.track_ids.remove(alias)
            owner.segments=[s for s in owner.segments if s.track_id!=alias]
            owner.appearance='; '.join(result.track_descriptions.get(t,'') for t in owner.track_ids)
        used={e.local_id for e in result.entities};serial=1
        while f'L{serial:02d}' in used:serial+=1
        local_id=f'L{serial:02d}'
        result.entities.append(LocalEntity(local_id=local_id,track_ids=[alias],kind=review.kind,
            label=review.description,appearance=review.description,reason=reason))
        result.track_assignments[alias]=local_id
        result.rejected_tracks=[t for t in result.rejected_tracks if t!=alias]
        result.rejection_reasons.pop(alias,None)
    # A stale semantic same/part-of claim cannot contradict the newly observed
    # category. Preserve it as uncertainty, rather than inventing a different ID.
    owners={t:e for e in result.entities for t in e.track_ids}
    for relation in result.relations:
        if alias in (relation.left,relation.right) and relation.relation in ('same_individual','part_of'):
            a,b=owners.get(relation.left),owners.get(relation.right)
            if a and b and a.kind!=b.kind:
                relation.relation='uncertain';relation.reason='Category disagreement after source-image classification review; identity unresolved'
    return result


def audit_local_classification(result,bundle,shot,aliases,output,backend,state,*,max_calls=4):
    from PIL import Image,ImageDraw,ImageOps
    from story_semantics import BudgetExhausted
    from low_freq_event.backend import ModelOutputError
    from story_relationship_images import _source_frames,_box,_font
    output=Path(output);folder=output/'local_classification_audit'/shot['shot_id'];folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for number,candidate in enumerate(classification_candidates(result,shot,aliases)):
        record={k:v for k,v in candidate.items() if k not in ('observations','rank')};record['status']='pending'
        if number>=4:
            record['reason']='Per-shot candidate cap';records.append(record);continue
        rows=candidate['observations'];selected=[rows[i] for i in sorted({len(rows)//4,len(rows)//2,3*len(rows)//4})]
        items=[dict(frame=r['frame'],left_observation=r,right_observation=r) for r in selected]
        source=_source_frames(bundle,shot,output,items);images=[]
        for row in selected:
            if row['frame'] not in source:continue
            original=source[row['frame']].copy();box=_box(row,original.size)
            crop=ImageOps.contain(original.crop(box),(720,520))
            ImageDraw.Draw(original).rectangle(box,outline='#00c8ed',width=max(3,round(original.width/400)))
            context=ImageOps.contain(original,(300,200))
            panel=Image.new('RGB',(1040,590),'#e3e8f0');draw=ImageDraw.Draw(panel)
            panel.paste(crop,((740-crop.width)//2,55+(520-crop.height)//2))
            panel.paste(context,(740,165))
            draw.text((15,12),f"TARGET CROP / SOURCE FRAME {row['frame']}",font=_font(25),fill='#102a46')
            draw.text((745,120),'SCENE CONTEXT',font=_font(20),fill='#102a46')
            panel.save(folder/f'track_{number:02d}_f{row["frame"]:08d}.jpg',quality=95)
            images.append((f'TARGET CROP / 원본 프레임 {row["frame"]}',panel))
        frames=[r['frame'] for r in selected if r['frame'] in source];record['shown_frames']=frames
        if len(frames)<2:
            record['reason']='Insufficient source evidence';records.append(record);continue
        prompt=(f'단일 대상 시각 분류 검수 VERSION {REVIEW_PROTOCOL_VERSION}\n'
            '왼쪽의 큰 TARGET CROP에 실제 보이는 대상만 분류하세요. 각 crop은 서로 다른 원본 프레임의 대상 영역입니다. '
            '오른쪽 작은 SCENE CONTEXT는 위치 참고용이며, 그 안의 다른 대상이나 눈에 띄는 얼굴을 대신 분류하면 안 됩니다. '
            '앞선 검출기나 모델의 라벨은 제공하지 않으니 TARGET CROP의 외형으로 판단하세요. '
            '사람 얼굴/몸이면 person, 동물/상상생물과 그 신체 부분이면 animal입니다. '
            '얼굴과 상반신처럼 인물 자체를 식별할 수 있으면 whole_subject, 손/날개/다리만 보이면 body_part입니다. '
            '보이는 머리·얼굴·피부·깃털·비늘·의복 등 실제 근거를 간단히 적으세요. 식별이 어려우면 unknown/uncertain을 사용하세요. '
            'JSON 필드: kind(person/animal/object/unknown), role(whole_subject/body_part/handled_object/worn_item/background/uncertain), '
            'description(주 대상 외형), evidence_frames(확인한 원본 번호 최소2개), visible_evidence(직접 보이는 분류 근거). '
            '인물 이름이나 영화 줄거리는 추측하지 마세요.\n'+json.dumps(dict(shot_id=shot['shot_id'],track=candidate['alias'],frames=frames),sort_keys=True))
        before=backend.ledger['generations'];phase=backend.phase;reserve=backend.reserve
        backend.phase=f'local_classification_audit:{shot["shot_id"]}'
        backend.reserve=max(reserve,backend.remaining-max(0,max_calls-state.get('generations',0)))
        try:
            review=backend.request(prompt,ClassificationReview,images=images,max_tokens=1000,image_pixels=1048576,
                validate=lambda r:validate_classification(r,frames))
            result=apply_classification(result,candidate['alias'],review)
            record.update(status='uncertain' if review.kind=='unknown' or review.role=='uncertain' else 'reviewed',review=review.model_dump())
        except (ModelOutputError,BudgetExhausted) as exc:record.update(status='pending',reason=str(exc))
        finally:
            state['generations']=state.get('generations',0)+backend.ledger['generations']-before
            backend.phase=phase;backend.reserve=reserve
        records.append(record)
    report=dict(version=VERSION,shot_id=shot['shot_id'],candidates=records,max_generation_calls_per_run=max_calls,
        generations_used_in_run=state.get('generations',0),status='needs_review' if any(r['status']!='reviewed' for r in records) else 'complete')
    (folder/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    shot['local_classification_audit']=report
    return result
