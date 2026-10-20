"""Reassign ambiguous local observations against explicit visual identity anchors.

Catalog labels and shot histories are hypotheses. Actual target crops, scenes and
co-visible distinct bodies constrain every reassignment. No film-specific rules.
"""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from pydantic import BaseModel,Field
from low_freq_event.storage import atomic_json
from shot_memory import MemoryShot,is_korean,korean_catalog_narrative

VERSION=9

class ContextAssignment(BaseModel):
    assignments:dict[str,str]
    reasons:dict[str,str]
    shots:list[MemoryShot]=Field(default_factory=list)
    labels:dict[str,str]=Field(default_factory=dict)


def reassign(bundle,local,catalog,output,backend,review_settings,context=None):
    import gc,torch
    from compact_identity import visual_board
    from story_semantics import local_lookup,cannot_links,scene_sheet,samples_evenly
    root=Path(output)/'context_reassignment';root.mkdir(parents=True,exist_ok=True)
    entries=local_lookup(local);entities={e['entity_id']:e for e in catalog['entities']};mapping=dict(catalog['mapping'])
    tracks={t['track_id']:t for s in bundle['shots'] for t in s['tracks']}
    shotmap={s['shot_id']:s for s in bundle['shots']};memory=[];decisions=[]
    forbidden=cannot_links(bundle,local)
    from boundary_identity import boundary_candidates
    boundaries=[b for b in boundary_candidates(bundle,entries,output,allow_unconfirmed=True) if b['bbox_iou']>=.9 and b['scene_similarity']>=.98 and b['crop_similarity']>=.97]
    original=backend.backend.settings
    backend.backend.model=None;backend.backend.processor=None;gc.collect();torch.cuda.empty_cache()
    backend.backend.settings=original.model_copy(update=review_settings)
    try:
        for phase,kind in enumerate(('person','animal')):
            refs={eid:e for eid,e in entities.items() if e['kind']==kind}
            grouped=defaultdict(list)
            for lid,e in entries.items():
                if mapping[lid] in refs:
                    # Human fragments already physically consolidated per shot.
                    # Animal locals stay separate: a tight crop can contain a
                    # different animal or an adult limb beside a carried young.
                    key=(e['shot_id'],mapping[lid]) if kind=='person' else (e['shot_id'],lid)
                    grouped[key].append(lid)
            units={f'U{i+1:03d}':dict(local_ids=lids,kind=kind) for i,lids in enumerate(grouped.values())}
            if not units and kind=='animal':continue
            if len(units)>32:raise ValueError('Context reassignment exceeds 32 targets; split input rather than truncate')
            def unit_area(u):return max(((o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]) for lid in u['local_ids'] for tid in entries[lid]['track_ids'] for o in tracks[tid]['observations']),default=0)
            if kind=='person':
                def low_detail(u):
                    shot=shotmap[entries[u['local_ids'][0]]['shot_id']]
                    width=shot.get('width') or bundle.get('width')
                    height=shot.get('height') or bundle.get('height')
                    if not width or not height:
                        from PIL import Image
                        frame=next(iter(shot['selected_images'].values()))
                        with Image.open(frame) as im:width,height=im.size
                    return unit_area(u)/(width*height)<.15
                linked={b[side] for b in boundaries if mapping[b['left_local_id']]!=mapping[b['right_local_id']] for side in ('left_local_id','right_local_id')}
                units={uid:u for uid,u in units.items() if low_detail(u) or bool(set(u['local_ids'])&linked)}
            if kind=='animal':
                backend.backend.model=None;backend.backend.processor=None;gc.collect();torch.cuda.empty_cache()
                backend.backend.settings=original
                animal_partition(bundle,local,entities,mapping,units,entries,catalog,output,backend,memory,context,decisions)
                continue
            images=[];sources={}
            for eid,e in refs.items():
                # Show chronological wardrobe/anatomy variants at equal size.
                # A largest-box thumbnail alone loses recurring travel clothes.
                from PIL import Image,ImageDraw,ImageOps
                scores={lid:unit_area(dict(local_ids=[lid])) for lid in e['local_ids']}
                clear=[lid for lid in e['local_ids'] if scores[lid]>=max(scores.values(),default=1)*.3]
                chosen=samples_evenly(clear,6)
                board=Image.new('RGB',(600,270*((len(chosen)+1)//2)),(242,245,249))
                draw=ImageDraw.Draw(board)
                for index,lid in enumerate(chosen):
                    ims,_,src=visual_board(bundle,entries,{eid:dict(e,local_ids=[lid])},[eid],output)
                    board.paste(ims[0][1].crop((0,0,300,260)),((index%2)*300,(index//2)*270))
                    draw.text(((index%2)*300+5,(index//2)*270+249),entries[lid]['shot_id'],fill='black')
                    sources[eid+':'+str(index)]=src
                images.append((eid+' REFERENCE: chronological clear observed views of this candidate',board))
            for uid,u in units.items():
                ims,_,src=visual_board(bundle,entries,units,[uid],output)
                images.append((uid+' TARGET to classify, not another reference',ims[0][1].crop((0,0,300,360))));sources.update(src)
            # Full-scene context for smallest targets; large clear crops already
            # have enough detail, and image count/area must remain bounded.
            def area(uid):return max(((o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]) for lid in units[uid]['local_ids'] for tid in entries[lid]['track_ids'] for o in tracks[tid]['observations']),default=0)
            contextual=list(shotmap) if not memory else list(dict.fromkeys(entries[units[uid]['local_ids'][0]]['shot_id'] for uid in sorted(units,key=area)))[:3]
            for sid in contextual:
                shot=deepcopy(shotmap[sid]);shot['scene_frames']=samples_evenly(shot['scene_frames'][1:-1] or shot['scene_frames'],3)
                aliases={tid:uid for uid,u in units.items() for lid in u['local_ids'] if entries[lid]['shot_id']==sid for tid in entries[lid]['track_ids']}
                images.append((sid+' TIMELINE: three different moments of this shot',scene_sheet(bundle,shot,output,aliases)))
            for index,(label,im) in enumerate(images):im.save(root/f'{kind}_{index:03d}.jpg',quality=92)
            expected_shots=[s['shot_id'] for s in local['shots']] if not memory else []
            unit_input={uid:dict(shot=entries[u['local_ids'][0]]['shot_id']) for uid,u in units.items()}
            inverse={lid:uid for uid,u in units.items() for lid in u['local_ids']}
            boundary_evidence=[]
            for b in boundaries:
                left=inverse.get(b['left_local_id'],mapping[b['left_local_id']]);right=inverse.get(b['right_local_id'],mapping[b['right_local_id']])
                if not (left.startswith('U') or right.startswith('U')):continue
                boundary_evidence.append(dict(left=left,right=right,time_gap_sec=b['right_time_sec']-b['left_time_sec'],bbox_iou=b['bbox_iou'],scene_similarity=b['scene_similarity'],crop_similarity=b['crop_similarity']))
                from PIL import Image,ImageOps
                board=Image.new('RGB',(640,240),'white')
                for i,key in enumerate(('left_crop','right_crop')):
                    with Image.open(b[key]) as im:
                        thumb=ImageOps.contain(im.convert('RGB'),(310,230));board.paste(thumb,(i*320,0))
                images.append((left+' → '+right+' ADJACENT BOUNDARY: inspect near-identical observed body across one frame',board))
            conflicts=sorted({tuple(sorted((inverse[a],inverse[b]))) for a,b in forbidden if a in inverse and b in inverse and inverse[a]!=inverse[b]})
            blocked={uid:sorted({mapping[b] for a,b in forbidden if a in u['local_ids'] and b not in inverse and mapping[a]!=mapping[b]}|{mapping[a] for a,b in forbidden if b in u['local_ids'] and a not in inverse and mapping[a]!=mapping[b]}) for uid,u in units.items()}
            occurrence={eid:len({entries[lid]['shot_id'] for lid in e['local_ids'] if unit_area(dict(local_ids=[lid]))>=.15*shotmap[entries[lid]['shot_id']]['width']*shotmap[entries[lid]['shot_id']]['height']}) for eid,e in refs.items()}
            total=sum(occurrence.values());lead=max(occurrence,key=occurrence.get) if occurrence else None
            lead=lead if lead and occurrence[lead]>=3 and occurrence[lead]/max(total,1)>=.6 else None
            for uid,u in units.items():
                dimensions=[min(o['bbox'][2]-o['bbox'][0],o['bbox'][3]-o['bbox'][1]) for lid in u['local_ids'] for tid in entries[lid]['track_ids'] for o in tracks[tid]['observations']]
                if max(dimensions,default=0)<64:
                    unit_input[uid].update(face_detail='too small to resolve age, hairline or beard',recurrence_candidate=lead)
            inventory=dict(clear_shot_recurrence=occurrence,dominant_recurring_subject_hypothesis=lead,adjacent_boundary_evidence=boundary_evidence,forbidden_reference_for_target=blocked,video_context=context or {},targets=unit_input,references={eid:dict(kind=e['kind'],high_detail_support=sum(unit_area(dict(local_ids=[lid]))>=max((unit_area(dict(local_ids=[x])) for x in e['local_ids']),default=1)*.3 for lid in e['local_ids'])) for eid,e in refs.items()},
                cannot_share_identity=conflicts,chronological_shot_history=memory or [dict(shot_id=s['shot_id'],prior_summary=__import__('re').sub(r'shot_[0-9]+(?::[A-Za-z0-9]+)*','',s['summary']),clear_references=sorted({mapping[e['local_id']] for e in s['entities'] if e['local_id'] not in inverse})) for s in local['shots']],summarize_shots=expected_shots)
            prompt='''You are analyzing a chronological video. First understand the complete TIMELINE of shots. All montage panels show DIFFERENT TIMES, not simultaneous copies of a body. Every G reference is a provisional identity, represented by clear chronological visual examples. Every U target is a low-detail or ambiguously classified observation requiring reassignment.
Adjacent_boundary_evidence identifies nearly identical observed crops at adjacent shot endpoints, usually one frame apart. Inspect those pairs together: a fading image of the same body should inherit the same identity, not acquire an imaginary face from another reference. These are strong continuity proposals, not identity names.
For a target marked face_detail too small, do NOT invent baldness, beard, age or facial similarity. Its recurrence_candidate is the empirically dominant clear subject: prefer that candidate when the traveler/body/gear is compatible and there is no positive visible contradiction. A rare character requires distinctive positive evidence, not a guessed face in a few pixels. Keep these assignments marked as contextual inference in the reasons.
Match every U to the visually and narratively best supported G, or uncertain. Use outfit construction, equipment, build, anatomy and continuing interaction, not just color or an old species name. Frequency/high_detail_support is a useful prior: a recurring central traveler often reappears in an ending montage with a hidden face. Compare that candidate FIRST, then check for positive contradictory evidence. A tiny or obscured face does not prove age, baldness or a different person. A background/lighting change alone does not prove a different identity. A clear different face or incompatible anatomy must stay separate. Do not force a match from frequency alone.
For animals, compare skin versus feathers, wing construction, relative size and the same interaction in preceding/following shots. Belly and back views may differ in color. A large body part and a nearby small creature remain different targets. Previously grouped locals can be separated; visually identical reference candidates can be duplicates, in which case consistently prefer the reference with more clear observations. Respect cannot_share_identity and forbidden_reference_for_target. Do not use named-film knowledge.
IMPORTANT: summarize_shots is the COMPLETE required list, including shots without U targets. Return exactly one short Korean summary and transition for EVERY shot in that list, in order. Prior summaries are fallible but useful descriptions: translate or revise using TIMELINE images, correcting clear contradictions; no guessed identities, flying humans, intentions or unseen actions. Transitions describe relation to the preceding shot: continuous, scene_change, montage, unknown. Quote literal on-screen text if used. If summarize_shots is empty return shots: []. Reasons and optional labels must be Korean.
Return ONLY JSON {"assignments":{"U001":"G0001"},"reasons":{"U001":"한국어 근거"},"shots":[{"shot_id":"...","summary":"한국어 요약","transition":"unknown","transition_evidence":"한국어 근거"}],"labels":{}}. Every U exactly once, every requested shot exactly once. No extra keys.
INPUT: '''+json.dumps(inventory,ensure_ascii=False)
            def validate(r):
                # Canonicalize only unambiguous zero padding of explicit U IDs.
                import re
                for field in ('assignments','reasons'):
                    values=getattr(r,field);normalized={}
                    for key,value in values.items():
                        canonical=f'U{int(key[1:]):03d}' if re.fullmatch(r'U[0-9]+',key) else key
                        if canonical in normalized or canonical not in units:raise ValueError('Unknown/duplicate target ID')
                        normalized[canonical]=value
                    setattr(r,field,normalized)
                if set(r.assignments)!=set(units) or set(r.reasons)!=set(units):raise ValueError('Assign and explain every U exactly once')
                if not set(r.assignments.values())<=set(refs)|{'uncertain'}:raise ValueError('Use existing reference IDs or uncertain only')
                if any(r.assignments[a]==r.assignments[b] and r.assignments[a]!='uncertain' for a,b in conflicts):raise ValueError('Separate observed bodies cannot share an identity')
                if any(gid in blocked[uid] for uid,gid in r.assignments.items()):raise ValueError('Target cannot merge with a separate fixed body')
                if expected_shots and [s.shot_id for s in r.shots]!=expected_shots:raise ValueError('Summarize all requested shots in order')
                if not expected_shots:r.shots=[]
                if any(not is_korean(s.summary) or not is_korean(s.transition_evidence) for s in r.shots):raise ValueError('All summaries and transition explanations must be Korean')
                if not set(r.labels)<=set(refs) or any(not is_korean(v) for v in r.labels.values()):raise ValueError('Labels must be Korean and use reference IDs')
                return r
            backend.phase=f'context_reassign:{phase+1:03d}'
            response=backend.request(prompt,ContextAssignment,images=images,max_tokens=5500 if expected_shots else 2200,image_pixels=786432,validate=validate,
                generation_options=dict(do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,seed=0,thinking_token_budget=2048),repair_root_closure=True)
            atomic_json(root/f'{kind}.json',dict(input=inventory,sources=sources,response=response.model_dump()))
            memory.extend(s.model_dump() for s in response.shots)
            for uid,eid in response.assignments.items():
                for lid in units[uid]['local_ids']:
                    chosen=mapping[lid] if eid=='uncertain' else eid
                    decisions.append(dict(local_id=lid,before=mapping[lid],after=chosen,uncertain=eid=='uncertain',reason=response.reasons[uid]));mapping[lid]=chosen
            for eid,label in response.labels.items():entities[eid]['label']=label
    finally:
        backend.backend.model=None;backend.backend.processor=None;gc.collect();torch.cuda.empty_cache();backend.backend.settings=original
    regrouped=defaultdict(list)
    for lid,eid in mapping.items():regrouped[eid].append(lid)
    for eid,lids in regrouped.items():
        if entities[eid]['kind']=='animal' and set(lids)<set(entities[eid]['local_ids']):
            entities[eid].update(label='미확정 동물 관측',uncertain=True,label_note='다른 개체로 재배정된 구성원을 제외한 잔여 관측; 기존 종 이름을 확정하지 않음')
    catalog['entities']=[dict(entities[eid],local_ids=lids) for eid,lids in sorted(regrouped.items())];catalog['mapping']=mapping
    track_owner={tid:mapping[e['local_id']] for s in local['shots'] for e in s['entities'] for tid in e['track_ids']}
    for proof in catalog.get('physical_groups',[]):
        ids={track_owner[m['track_id']] for m in proof['members']}
        if len(ids)!=1:raise ValueError('Context assignment split an already verified physical body')
        proof['entity_id']=next(iter(ids))
    catalog['context_reassignments']=decisions
    saved=dict(video_context=context or {},shots=memory,policy='Joint chronological summaries and anchored reassignment; not cast-specific rules.')
    atomic_json(Path(output)/'shot_memory/memory.json',saved)
    return korean_catalog_narrative(bundle,local,catalog,saved)


class AnimalPairs(BaseModel):
    compatible:dict[str,bool]
    reasons:dict[str,str]


def animal_partition(bundle,local,entities,mapping,units,entries,catalog,output,backend,memory,context,decisions):
    """Resolve low-support animal fragments against retrieved individual observations."""
    import numpy as np
    from PIL import Image
    from compact_identity import visual_board
    from story_semantics import cannot_links
    tracks={t['track_id']:t for s in bundle['shots'] for t in s['tracks']}
    support={eid:len({entries[lid]['shot_id'] for lid in e['local_ids']}) for eid,e in entities.items()}
    vectors={}
    for uid,u in units.items():
        values=[o['feature'] for lid in u['local_ids'] for tid in entries[lid]['track_ids'] for o in tracks[tid].get('representatives',[]) if o.get('feature')]
        if values:
            mean=np.mean(values,axis=0);vectors[uid]=mean/max(float(np.linalg.norm(mean)),1e-9)
    forbidden={tuple(sorted(p)) for p in cannot_links(bundle,local)}
    pairs={};images=[];sources={}
    for uid,u in units.items():
        lid=u['local_ids'][0];old=mapping[lid]
        # Only sparse fragments need the additional contextual identity check.
        # A long established group is not repartitioned from weak global names.
        if support[old]>1 or uid not in vectors:continue
        candidates=[(float(vectors[uid]@vectors[rid]),rid) for rid in vectors if mapping[units[rid]['local_ids'][0]]!=old and support[mapping[units[rid]['local_ids'][0]]]>support[old]]
        if not candidates:continue
        ranked=sorted(candidates,reverse=True);score,rid=ranked[0]
        reverse=sorted([(float(vectors[rid]@vectors[other]),other) for other in vectors if mapping[units[other]['local_ids'][0]]!=mapping[units[rid]['local_ids'][0]]],reverse=True)
        margin=score-ranked[1][0] if len(ranked)>1 else 1.
        reverse_margin=reverse[0][0]-reverse[1][0] if len(reverse)>1 else 1.
        if score<.75 or margin<.035 or reverse[0][1]!=uid or reverse_margin<.035:continue
        ref=units[rid]['local_ids'][0];gid=mapping[ref]
        if any(tuple(sorted((lid,other))) in forbidden for other in entities[gid]['local_ids']):continue
        key=f'P{len(pairs)+1:03d}'
        pairs[key]=dict(target=uid,reference=rid,target_local_id=lid,reference_local_id=ref,destination=gid,similarity=score,margin=margin,reverse_margin=reverse_margin)
        # Retrieve matching viewpoints, not simply the largest wing/body box.
        # The cached DINO descriptors select images; they never approve a merge.
        from PIL import ImageOps,ImageDraw
        root=Path(bundle.get('asset_root',output))
        def views(which):
            return [o for x in units[which]['local_ids'] for tid in entries[x]['track_ids'] for o in tracks[tid]['observations']
                    if o.get('feature') and o.get('crop') and (root/o['crop']).is_file()]
        choices=[]
        for a in views(rid):
            for b in views(uid):
                av=np.asarray(a['feature']);bv=np.asarray(b['feature'])
                cosine=float(av@bv)/max(float(np.linalg.norm(av)*np.linalg.norm(bv)),1e-9)
                choices.append((cosine,a,b))
        picked=[];seen_a=set();seen_b=set()
        for cosine,a,b in sorted(choices,key=lambda x:-x[0]):
            if a['frame'] in seen_a or b['frame'] in seen_b:continue
            picked.append((a,b));seen_a.add(a['frame']);seen_b.add(b['frame'])
            if len(picked)==3:break
        if not picked:
            raise ValueError('Animal pair lacks observed crop evidence')
        board=Image.new('RGB',(640,250*len(picked)),'#eef2f6');draw=ImageDraw.Draw(board)
        for row,observations in enumerate(picked):
            for col,o in enumerate(observations):
                with Image.open(root/o['crop']) as im:thumb=ImageOps.contain(im.convert('RGB'),(310,220))
                board.paste(thumb,(col*320+(310-thumb.width)//2,row*250+25))
                draw.text((col*320+6,row*250+5),('REFERENCE' if col==0 else 'TARGET')+' f'+str(o['frame']),fill='black')
        sources[key]=[dict(reference=dict(frame=a['frame'],crop=a['crop']),target=dict(frame=b['frame'],crop=b['crop'])) for a,b in picked]
        images.append((key+': compare LEFT reference and RIGHT target across matched viewpoints, one pair per row',board))
    if not pairs:return
    if len(pairs)>12:raise ValueError('Animal fragment review exceeds 12 pairs; split input')
    history={s['shot_id']:s for s in memory}
    inventory=dict(video_context=context or {},pairs={key:dict(retrieval_cosine=p['similarity'],reference_shot=entries[p['reference_local_id']]['shot_id'],target_shot=entries[p['target_local_id']]['shot_id'],chronological_context=[{k:history[sid][k] for k in ('shot_id','transition','transition_evidence')} for sid in sorted({entries[p['reference_local_id']]['shot_id'],entries[p['target_local_id']]['shot_id']}) if sid in history]) for key,p in pairs.items()})
    prompt='''Do the LEFT and RIGHT animals in each P image have compatible underlying creature anatomy/design, allowing different camera angles, pose, expression and lighting? Each row compares two actual video observations. Inspect the visible eyes, snout, skull structures, skin, ears/horns and limbs. Do not invent structures from blur. Background differences or an open versus closed mouth do not make a different creature design. A body part from a large animal is not compatible with a complete small animal.
Return true for compatible intrinsic anatomy, false for a clear intrinsic contradiction or insufficient detail. This does not by itself prove identity: the caller separately requires a strong unique mutual visual-feature match and excludes distinct simultaneous bodies. Old species names have deliberately been withheld because they can be wrong. Brief Korean reasons. ONLY JSON {"compatible":{"P001":true},"reasons":{"P001":"눈·주둥이·머리와 날개 구조가 시점 차이를 고려하면 일치한다."}}. Every P exactly once.
INPUT: '''+json.dumps(inventory,ensure_ascii=False)
    def validate(r):
        if set(r.compatible)!=set(pairs) or set(r.reasons)!=set(pairs):raise ValueError('Review every pair exactly once')
        return r
    backend.phase='context_reassign:002'
    response=backend.request(prompt,AnimalPairs,images=images,max_tokens=1500,image_pixels=786432,validate=validate,
        generation_options=dict(do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,seed=0,thinking_token_budget=2048),repair_root_closure=True)
    root=Path(output)/'context_reassignment';root.mkdir(parents=True,exist_ok=True)
    for index,(_,im) in enumerate(images):im.save(root/f'animal_pair_{index:03d}.jpg',quality=92)
    atomic_json(root/'animal.json',dict(input=inventory,pairs=pairs,sources=sources,response=response.model_dump()))
    proposed=dict(mapping)
    for key,p in pairs.items():
        if response.compatible[key]:proposed[p['target_local_id']]=p['destination']
    owner={tid:proposed[lid] for lid,e in entries.items() for tid in e['track_ids']}
    for proof in catalog.get('physical_groups',[]):
        if len({owner[m['track_id']] for m in proof['members']})>1:raise ValueError('Animal reassignment would split a verified body')
    for key,p in pairs.items():
        lid=p['target_local_id'];decisions.append(dict(local_id=lid,before=mapping[lid],after=proposed[lid],uncertain=not response.compatible[key],reason=response.reasons[key],method='mutual_visual_retrieval_plus_anatomical_compatibility',similarity=p['similarity'],margin=p['margin'],reverse_margin=p['reverse_margin']));mapping[lid]=proposed[lid]
