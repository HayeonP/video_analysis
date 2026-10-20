"""Surgical repair of an existing identity catalog, using selected images only.

Keep established cross-shot associations. Review labels and physical collisions,
not every identity pair again. No evaluation annotations or video names are read.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib
import json
from typing import Literal
from PIL import Image, ImageDraw, ImageOps, ImageFont
from pydantic import BaseModel, Field
from low_freq_event.storage import atomic_json

VERSION = 1

class Body(BaseModel):
    members: list[str] = Field(min_length=1)
    identity: str
    reason: str = Field(min_length=1)

class FrameDecision(BaseModel):
    bodies: list[Body]


class PhysicalBody(BaseModel):
    members: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class NonSubject(BaseModel):
    kind: Literal['cast_shadow','reflection','background_artifact','text']
    reason: str = Field(min_length=1)


class PhysicalPartition(BaseModel):
    bodies: list[PhysicalBody]
    excluded: dict[str,NonSubject] = Field(default_factory=dict)


def reconcile_physical_partition(response, partition, count, allowed):
    """Identity uncertainty cannot create an extra physical body or a shadow ID."""
    previous={m:b.identity for b in response.bodies for m in b.members}
    bodies=[]
    for body in partition.bodies:
        known={previous[m] for m in body.members if m in previous and previous[m]!='new'}
        if len(known)>1:raise ValueError('Physical partition combines conflicting known identities; keep them separate or review their identities independently.')
        bodies.append(Body(members=body.members,identity=next(iter(known)) if known else 'new',reason=body.reason))
    if any(not item.reason.strip() for item in partition.excluded.values()):raise ValueError('Every excluded detection needs visible non-entity evidence.')
    return validate_frame(FrameDecision(bodies=bodies),count,allowed,excluded=partition.excluded)


def needs_physical_count(response, rows, entities):
    # Human anatomy review must not reinterpret separate flying animals as
    # detached wings. The ordinary frame review still handles every entity kind.
    return (any(b.identity=='new' for b in response.bodies)
            and all(entities[gid].get('kind')=='person' for key,gid,track,o in rows))


class FrameAssignments(BaseModel):
    assignments: dict[str, str]
    evidence: dict[str, str]

    def bodies(self, count, allowed):
        import re
        expected={f'D{i+1}' for i in range(count)}
        if set(self.assignments)!=expected or set(self.evidence)!=expected:
            raise ValueError('assignments and evidence must each contain exactly '+str(sorted(expected)))
        grouped=defaultdict(list)
        for member,eid in self.assignments.items():
            if eid not in allowed and not re.fullmatch(r'new[1-9][0-9]*',eid):
                raise ValueError('Each value must be a supplied G identity or new1/new2 for a separate unknown body')
            grouped[eid].append(member)
        result=FrameDecision(bodies=[Body(members=members,identity=eid if eid in allowed else 'new',
            reason='; '.join(self.evidence[m] for m in members)) for eid,members in grouped.items()])
        return validate_frame(result,count,allowed)

class Description(BaseModel):
    id: str
    label: str
    appearance: str
    inconsistent: bool = False

class Descriptions(BaseModel):
    profiles: list[Description]


def read(path):return json.loads(Path(path).read_text())


def observation_key(shot, observation):
    # File bytes/coordinates, not fragile track IDs, bind reused evidence.
    return (shot, int(observation['frame']), tuple(observation['bbox']))


def prepare(bundle, local, catalog, profile_root=None):
    """Retain prior links, split only independently observed trait conflicts."""
    from compact_identity import canonical, incompatible
    bundle,local,catalog=deepcopy(bundle),deepcopy(local),deepcopy(catalog)
    entries={e['local_id']:e for s in local['shots'] for e in s['entities']}
    owners={t:e['local_id'] for e in entries.values() for t in e['track_ids']}
    observations={observation_key(s['shot_id'],o):(catalog['mapping'][owners[t['track_id']]],t,o)
                  for s in bundle['shots'] for t in s['tracks'] if t['track_id'] in owners for o in t['observations']}
    profiles=defaultdict(list); assignments={};prototypes={};sources=[]
    if profile_root:
        profile_root=Path(profile_root);pb=read(profile_root/'reviewed_tracking.json');pl=read(profile_root/'local_entities.json')
        pts={t['track_id']:t for s in pb['shots'] for t in s['tracks']}
        cached=read(profile_root/'compact_identity/profiles.json')
        for shot in pl['shots']:
            for e in shot['entities']:
                p=e.get('target_profile',cached.get(e['local_id'],{}))
                if not p.get('verified',True) or p.get('mixed') or p.get('kind')!='person':continue
                if sum(p.get(k,'unknown')!='unknown' for k in ('hair','presentation','facial_hair'))<2:continue
                rows=[o for tid in e['track_ids'] for o in pts[tid]['observations']]
                matches=[(observation_key(shot['shot_id'],o),o) for o in rows if observation_key(shot['shot_id'],o) in observations]
                gids={observations[k][0] for k,o in matches}
                if len(gids)!=1:continue
                gid=next(iter(gids));sample=next((o for k,o in matches if o['frame']==p['frame']),None)
                if sample is None:continue
                original=observations[observation_key(shot['shot_id'],sample)][2]
                # Cached metadata alone is insufficient: verify the cited image bytes.
                a=Path(bundle['asset_root'])/(original.get('crop') or original['image']);b=Path(pb.get('asset_root',profile_root))/(sample.get('crop') or sample['image'])
                if hashlib.sha256(a.read_bytes()).digest()!=hashlib.sha256(b.read_bytes()).digest():raise ValueError('Profile source pixels differ')
                profiles[gid].append(dict(profile=p,shot=shot['shot_id'],rows=[k for k,o in matches],sample=original))
    entities={e['entity_id']:e for e in catalog['entities']};next_id=max(int(k[1:]) for k in entities)+1
    for gid,records in profiles.items():
        def support(record):
            p=record['profile']
            compatible=[r for r in records if not incompatible(p,r['profile'])]
            return len({r['shot'] for r in compatible}),len(compatible),sum(p.get(k,'unknown')!='unknown' for k in ('hair','presentation','facial_hair'))
        primary=max(records,key=support);partitions=[(gid,primary,[])]
        for record in sorted(records,key=lambda r:support(r),reverse=True):
            # Only target-specific profiles may split a previously accepted identity.
            conflicts=bool(incompatible(record['profile'],primary['profile']))
            if conflicts and not record['profile'].get('target_verified'):continue
            target=next((x for x in partitions if not incompatible(x[1]['profile'],record['profile'])),None)
            if target is None:
                key=f'G{next_id:04d}';next_id+=1
                entities[key]=dict(entities[gid],entity_id=key,local_ids=[],reason='Split after target-specific visible trait contradiction.')
                target=(key,record,[]);partitions.append(target)
            target[2].append(record)
        for key,representative,accepted in partitions:
            p=representative['profile'];entities[key].update(canonical(dict(local_ids=['p'],kind=p['kind']),{'p':p}))
            prototypes[key]=representative['sample']
            sources.append(dict(entity_id=key,source_group=gid,profile=p,shot_id=representative['shot'],frame=p['frame']))
            for record in accepted:
                if record['profile'].get('target_verified'):
                    for obskey in record['rows']:assignments[obskey]=key
    # Every prior identity gets a representative for collision checks, including nonpeople.
    from view_quality import face_detail
    for gid,e in entities.items():
        if gid in prototypes:continue
        options=[o for key,(g,t,o) in observations.items() if g==gid]
        if not options:continue
        def quality(o):
            face=face_detail(bundle,o,Path(bundle['asset_root'])) if e['kind']=='person' else None
            return (bool(face),face['quality'] if face else 0,(o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]))
        prototypes[gid]=max(options,key=quality)
    return bundle,local,entities,observations,assignments,prototypes,sources


def board(bundle, observations, prototypes, identities, *, zoom_context=False):
    root=Path(bundle['asset_root']);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    with Image.open(root/observations[0]['image']) as im:scene=im.convert('RGB')
    pen=ImageDraw.Draw(scene)
    colors=['#ffdd00','#00ffff','#ff55aa','#55ff55','#ff8800','#8888ff']
    if not zoom_context:
        for i,o in enumerate(observations):
            pen.rectangle(o['bbox'],outline=colors[i%len(colors)],width=5)
            pen.text((o['bbox'][0],max(0,o['bbox'][1]-25)),f'D{i+1}',fill=colors[i%len(colors)],font=font,stroke_width=2,stroke_fill='black')
    if zoom_context:
        boxes=[o['bbox'] for o in observations]
        x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes)
        x1=max(b[2] for b in boxes);y1=max(b[3] for b in boxes)
        margin=max(40,.25*max(x1-x0,y1-y0))
        bounds=(max(0,int(x0-margin)),max(0,int(y0-margin)),
                min(scene.width,int(x1+margin)),min(scene.height,int(y1+margin)))
        scene=ImageOps.contain(scene.crop(bounds),(1400,700))
        sx=scene.width/(bounds[2]-bounds[0]);sy=scene.height/(bounds[3]-bounds[1])
        pen=ImageDraw.Draw(scene)
        for i,o in enumerate(observations):
            b=o['bbox'];box=((b[0]-bounds[0])*sx,(b[1]-bounds[1])*sy,(b[2]-bounds[0])*sx,(b[3]-bounds[1])*sy)
            pen.rectangle(box,outline=colors[i%len(colors)],width=3)
            pen.text((box[0],max(0,box[1]-25)),f'D{i+1}',fill=colors[i%len(colors)],font=font,stroke_width=2,stroke_fill='black')
    width=1400;scene=ImageOps.contain(scene,(width,700));cards=len(observations)+len(identities)
    result=Image.new('RGB',(width,scene.height+320*((cards+4)//5)), '#edf1f7');result.paste(scene,((width-scene.width)//2,0));draw=ImageDraw.Draw(result)
    for i,(label,o) in enumerate([(f'D{j+1}',o) for j,o in enumerate(observations)]+[(k,prototypes[k]) for k in identities]):
        x=(i%5)*280;y=scene.height+(i//5)*320
        draw.text((x+6,y+4),label,fill='black',font=font)
        with Image.open(root/(o.get('crop') or o['image'])) as im:
            pixels=im.convert('RGB') if o.get('crop') else im.convert('RGB').crop(o['bbox'])
            crop=ImageOps.contain(pixels,(270,278))
        result.paste(crop,(x+(280-crop.width)//2,y+36))
    return result


def validate_frame(response, count, allowed, *, excluded=()):
    import re
    members=[m for b in response.bodies for m in b.members]+list(excluded)
    if len(members)!=len(set(members)) or set(members)!={f'D{i+1}' for i in range(count)}:raise ValueError('Every detection must belong to exactly one physical body. Missing='+str(sorted({f'D{i+1}' for i in range(count)}-set(members)))+'; duplicate/unknown='+str([m for m in members if members.count(m)>1 or m not in {f'D{i+1}' for i in range(count)}]))
    ids=[b.identity for b in response.bodies if b.identity!='new']
    if len(ids)!=len(set(ids)) or not set(ids)<=set(allowed):raise ValueError('One identity per physical body/frame; only supplied prototypes or new.')
    for body in response.bodies:
        mentioned=set(re.findall(r'\bG\d{4}\b',body.reason)) & set(allowed)
        if len(body.members)>1 and mentioned-{body.identity}:
            raise ValueError('Your reason assigns these detections to different reference identities '+str(sorted(mentioned))+', but your bodies array merges them. Return separate bodies with the respective members/identity, consistent with your visual explanation.')
    return response


def needs_second_view(response, rows, assignments, roles):
    """Escalate changed identity and weak-overlap whole-body unions, not all frames."""
    from itertools import combinations
    if response is None:return True
    for body in response.bodies:
        selected=[rows[int(member[1:])-1] for member in body.members]
        if any(body.identity!=assignments.get(key,gid) for key,gid,track,o in selected):return True
        for a,b in combinations(selected,2):
            if not all(roles.get(row[2]['track_id'])=='whole_subject' for row in (a,b)):continue
            x,y=a[3]['bbox'],b[3]['bbox']
            intersection=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
            area=min((x[2]-x[0])*(x[3]-x[1]),(y[2]-y[0])*(y[3]-y[1]))
            if intersection/max(area,1)<.5:return True
    return False


def request_assignments(backend, prompt, image, count, allowed):
    """One bounded format recovery; never drop invalid or missing detections."""
    from low_freq_event.backend import ModelOutputError
    kwargs=dict(images=[('Main targets, not peripheral faces',image)],max_tokens=1000,
                image_pixels=1572864,validate=lambda r:(r.bodies(count,allowed) and r))
    try:
        return backend.request(prompt,FrameAssignments,**kwargs)
    except ModelOutputError:
        keys=[f'D{i+1}' for i in range(count)]
        correction=('\nSTRICT DETECTION INVENTORY: '+json.dumps(keys)+
            '. These are the ONLY keys allowed in assignments AND evidence. '+
            'G cards are reference photographs from OTHER frames, NEVER detections. '+
            'Do not add rows for any reference photograph. '+
            'Fill exactly this template using visual evidence: '+json.dumps(dict(
                assignments={k:'<G reference ID or new1>' for k in keys},
                evidence={k:'<short visible support>' for k in keys})))
        return backend.request(prompt+correction,FrameAssignments,**kwargs)


FRAME_PROMPT='''Repair detections in ONE source frame (top image). Colored D boxes and D crops are detections in that SAME frame. G cards below are reference identities from other frames.
Partition ALL D detections by PHYSICAL BODY: overlapping/truncated boxes, a face/arm/torso belonging to one person, and duplicate detections belong to ONE body. Different people/creatures visible together belong to DIFFERENT bodies. Ignore reflections. Do not create multiple bodies just because the boxes differ. Conversely, never combine adjacent people merely because their clothing is similar.
Match each body to a supplied G reference identity from visible anatomy, face, hair shape, gear and supported continuity. References are appearance examples; lighting, pose, mask and outfit may change. For an incompatible or genuinely unidentifiable body use identity "new". A small attached part can use the visible owner's identity. A carried independent animal remains a separate body. Do not copy a nearby person's attributes. Every D must appear exactly once, and no G may be assigned to two bodies.
Return concise JSON {"bodies":[{"members":["D1","D2"],"identity":"G0001","reason":"visible physical connection and matching identity"}]}. No prose.
'''


def repair(bundle,local,catalog,output,backend,profile_root=None,*,batch_review=None):
    from story_semantics import BudgetExhausted
    from low_freq_event.backend import ModelOutputError
    from compact_identity import same_region
    output=Path(output);dest=output/'catalog_repair';dest.mkdir(parents=True,exist_ok=True)
    bundle,local,entities,observations,assignments,prototypes,sources=prepare(bundle,local,catalog,profile_root)
    by_frame=defaultdict(list)
    for key,(gid,track,o) in observations.items():by_frame[(key[0],key[1])].append((key,gid,track,o))
    proofs=[];reviews=[];failures=[];exclusions={};next_id=max(int(k[1:]) for k in entities)+1
    variants=defaultdict(set)
    for s in sources:variants[s['source_group']].add(s['entity_id'])
    contaminated_shots={(s['shot_id'],s['source_group']) for s in sources if s['entity_id']!=s['source_group']}
    # Labels are corrected independently from membership. Prior associations remain.
    atomic_json(dest/'profile_sources.json',sources)
    selected=[]
    roles={tid:role for s in local['shots'] for tid,role in s.get('track_roles',{}).items()}
    for (sid,frame),rows in by_frame.items():
        conflict=any(assignments.get(a[0],a[1])==assignments.get(b[0],b[1]) and not same_region(a[3]['bbox'],b[3]['bbox']) for i,a in enumerate(rows) for b in rows[i+1:])
        contaminated=any((sid,g) in contaminated_shots for k,g,t,o in rows)
        if conflict or contaminated:selected.append((sid,frame,rows))
    batched=batch_review(bundle,selected,entities,assignments,prototypes,variants,output) if batch_review else {}
    pending=[]
    from tqdm import tqdm
    for index,(sid,frame,rows) in enumerate(tqdm(sorted(selected),desc='충돌 프레임 검수',unit='frame'),1):
        allowed=sorted({x for k,g,t,o in rows for x in ({g}|variants[g]) if x in prototypes})
        folder=dest/f'{index:03d}';folder.mkdir(exist_ok=True)
        im=board(bundle,[o for k,g,t,o in rows],prototypes,allowed);im.save(folder/'evidence.jpg',quality=92)
        inventory=dict(shot_id=sid,frame=frame,detections={f'D{i+1}':dict(track_id=t['track_id'],bbox=o['bbox'],prior_identity=g) for i,(k,g,t,o) in enumerate(rows)},allowed=allowed)
        atomic_json(folder/'input.json',inventory);backend.phase=f'catalog_repair:{index:03d}'
        if batch_review:
            response,allowed=batched[(sid,frame)]
            pending.append((index,sid,frame,rows,allowed,folder,im,response))
            continue
        try:
            response=backend.request(FRAME_PROMPT+'\nINPUT: '+json.dumps(dict(ids=list(inventory['detections']),references=allowed)),FrameDecision,
                images=[('One source frame, its detections, and reference identities',im)],max_tokens=900,image_pixels=1572864,
                validate=lambda r:validate_frame(r,len(rows),allowed))
        except ModelOutputError as exc:
            atomic_json(folder/'first_pass_failure.json',dict(error=str(exc)))
            response=None
        except BudgetExhausted as exc:
            failures.append(dict(shot_id=sid,frame=frame,error=str(exc)));atomic_json(folder/'failure.json',failures[-1]);continue
        pending.append((index,sid,frame,rows,allowed,folder,im,response))
    # Batch escalations after the cheap pass: at most one model switch, and the
    # expensive model is never loaded when no changed/ambiguous union exists.
    stronger=None
    for index,sid,frame,rows,allowed,folder,im,response in tqdm(pending,desc='변경·충돌 추가 검수',unit='frame'):
        if response is not None:atomic_json(folder/'initial_decision.json',response.model_dump())
        if not batch_review and needs_second_view(response,rows,assignments,roles):
            if stronger is None:
                import gc,torch
                from story_semantics import StoryBackend
                from story_pipeline import default_global_model,THINKING_REVISION
                backend.model=None;backend.processor=None;gc.collect();torch.cuda.empty_cache()
                review_settings=getattr(backend,'repair_settings',None) or backend.settings.model_copy(update=dict(model_id=default_global_model(),model_revision=THINKING_REVISION))
                stronger=StoryBackend(review_settings,output,backend.limit)
                stronger.identity_thinking_budget=1024
            stronger.phase=f'global_focused:repair_{index:03d}'
            check_prompt=FRAME_PROMPT+"\nINDEPENDENT CHECK: A prior review may have confused a BACKGROUND person's face at the EDGE of another person's box with the MAIN FOREGROUND body filling that box. Inspect each D crop separately. Identify the MAIN body/torso occupying its CENTER, not any recognizable face at its edge. Two overlapping boxes can enclose different people, including a foreground person's back and an opponent behind them. Conversely, connected limbs and duplicate crops of one body must stay together. Count main bodies from the source frame before matching references. Prior track associations below are fallible continuity hints, not proof: "+json.dumps({f'D{i+1}':assignments.get(k,g) for i,(k,g,t,o) in enumerate(rows)})+'\nReferences: '+json.dumps(allowed)
            try:
                reserve=stronger.reserve;stronger.reserve=stronger.remaining
                try:
                    response=stronger.request(check_prompt,FrameDecision,images=[('Main targets, not peripheral faces',im)],max_tokens=1000,image_pixels=1572864,
                        validate=lambda r:validate_frame(r,len(rows),allowed))
                except BudgetExhausted:
                    response=None
                finally:
                    stronger.reserve=reserve
                if response is None:
                    flat_prompt=check_prompt+'''\nOUTPUT FORMAT OVERRIDE: use a flat per-detection map, NOT a bodies/members array. Return {"assignments":{"D1":"reference ID or new1",...},"evidence":{"D1":"brief visible target-specific support",...}}. Include EVERY supplied D exactly once in each dictionary. Two detections on the same physical body get the same value. Distinct unknown bodies get new1, new2, etc. G reference IDs are VALUES, never detection keys. Keep evidence short and consistent with the assignment.'''
                    flat=request_assignments(stronger,flat_prompt,im,len(rows),allowed)
                    response=flat.bodies(len(rows),allowed)
            except (ModelOutputError,BudgetExhausted) as exc:
                failures.append(dict(shot_id=sid,frame=frame,error=str(exc)));atomic_json(folder/'failure.json',failures[-1]);continue
        excluded={}
        if not batch_review and needs_physical_count(response,rows,entities):
            # Remove reference portraits: this question is solely about bodies
            # present in this exposure, not whether their identity is recognizable.
            physical_board=board(bundle,[o for k,g,t,o in rows],{},[],zoom_context=True)
            physical_board.save(folder/'physical_only.jpg',quality=94)
            reviewer=stronger or backend
            reviewer.phase=f'global_focused:repair_physical_{index:03d}'
            physical_prompt='''Count HUMAN PHYSICAL BODIES in this ONE source exposure, independently of identity. The top image is a magnified context around the detections; D cards below enlarge detections from that same moment, NOT additional people or other moments. Inspect where each box lies in the shared context. Multiple overlapping boxes on one torso, face, or attached limbs belong to ONE body even if one crop lacks recognizable facial details. Attached body parts must be grouped with their owner, NEVER excluded merely because they are not independent people. Lack of identity detail is NOT evidence of a second body. A flat CAST SHADOW extending from feet along the ground is not a person or a body part: exclude that shadow detection. A reflection, text, or background artifact is also not an independent physical subject. Another physically separate person must remain a separate body. For separate bodies explain their separate physical locations. Partition EVERY supplied D exactly once across bodies and excluded. Do not use reference IDs or character names.
Return {"bodies":[{"members":["D1","D2"],"reason":"visible physical connection"}],"excluded":{"D3":{"kind":"cast_shadow","reason":"flat cast shadow on the ground, not another body"}}}.
ONLY detection IDs: '''+json.dumps([f'D{i+1}' for i in range(len(rows))])
            try:
                partition=reviewer.request(physical_prompt,PhysicalPartition,
                    images=[('ONE frame and its detection enlargements; NO identity reference photos',physical_board)],
                    max_tokens=900,image_pixels=1572864,
                    validate=lambda p:(reconcile_physical_partition(response,p,len(rows),allowed) and p))
                response=reconcile_physical_partition(response,partition,len(rows),allowed)
                excluded={m:f'{item.kind}: {item.reason}' for m,item in partition.excluded.items()}
                atomic_json(folder/'physical_partition.json',partition.model_dump())
            except (ModelOutputError,BudgetExhausted) as exc:
                failures.append(dict(shot_id=sid,frame=frame,error=str(exc)));atomic_json(folder/'failure.json',failures[-1]);continue
        atomic_json(folder/'decision.json',dict(response.model_dump(),excluded=excluded))
        (folder/'failure.json').unlink(missing_ok=True)
        for member,reason in excluded.items():
            key,g,t,o=rows[int(member[1:])-1]
            assignments[key]='__excluded__';exclusions[key]=dict(reason=reason,review=str(folder.relative_to(output)))
        for body in response.bodies:
            eid=body.identity
            if eid=='new':
                eid=f'G{next_id:04d}';next_id+=1
                original=rows[int(body.members[0][1:])-1][1]
                entities[eid]=dict(entities[original],entity_id=eid,local_ids=[],label='식별 미확정',appearance='',uncertain=True,reason=body.reason)
            keys=[]
            for member in body.members:
                key,g,t,o=rows[int(member[1:])-1];assignments[key]=eid
                keys.append(dict(track_id=t['track_id'],bbox=o['bbox']))
            if len(keys)>1:proofs.append(dict(shot_id=sid,frame=frame,entity_id=eid,members=keys,reason=body.reason,review=str(folder.relative_to(output))))
        reviews.append(dict(shot_id=sid,frame=frame,review=str(folder.relative_to(output))))
    if stronger is not None:
        backend.ledger=stronger.ledger
        stronger.model=None;stronger.processor=None;gc.collect();torch.cuda.empty_cache()
    if failures:
        atomic_json(dest/'audit.json',dict(failures=failures,reviewed=len(reviews),requested=len(selected)))
        raise RuntimeError('Unresolved frame reviews; prior output remains intact. Resume to use cached replies.')
    # Materialize observation-level corrections as tracks/local IDs, retaining all detections.
    repaired_local=deepcopy(local);new_mapping={};track_alias={};before_count=sum(len(t['observations']) for shot in bundle['shots'] for t in shot['tracks'] if any(t['track_id'] in e['track_ids'] for row in local['shots'] for e in row['entities']))
    for shot in bundle['shots']:
        row=next(s for s in repaired_local['shots'] if s['shot_id']==shot['shot_id'])
        old_owners={tid:e for e in row['entities'] for tid in e['track_ids']};out_tracks=[];out_entities={}
        for track in shot['tracks']:
            if track['track_id'] not in old_owners:out_tracks.append(track);continue
            owner=old_owners[track['track_id']];prior=catalog['mapping'][owner['local_id']];parts=defaultdict(list)
            for o in track['observations']:parts[assignments.get(observation_key(shot['shot_id'],o),prior)].append(o)
            for eid,obs in parts.items():
                tid=track['track_id'] if len(parts)==1 else track['track_id']+':R'+eid
                child=deepcopy(track);child.update(track_id=tid,observations=deepcopy(obs),representatives=deepcopy(obs),first_frame=min(o['frame'] for o in obs),last_frame=max(o['frame'] for o in obs),observed_frames=len(obs))
                for o in child['observations']:
                    o['track_id']=tid;track_alias[(shot['shot_id'],o['frame'],track['track_id'])]=tid
                out_tracks.append(child)
                if eid=='__excluded__':
                    row.setdefault('rejected_tracks',[]).append(tid)
                    evidence=[exclusions[observation_key(shot['shot_id'],o)] for o in obs]
                    row.setdefault('rejection_reasons',{})[tid]='Physical-only review: '+ '; '.join(sorted({e['reason'] for e in evidence}))
                    child['physical_exclusion_evidence']=evidence
                    continue
                lid=owner['local_id']+':R'+eid
                if lid not in out_entities:out_entities[lid]=dict(deepcopy(owner),local_id=lid,track_ids=[],segments=[])
                out_entities[lid]['track_ids'].append(tid);new_mapping[lid]=eid
                if track['track_id'] in row.get('track_roles',{}):row['track_roles'][tid]=row['track_roles'][track['track_id']]
        shot['tracks']=out_tracks;row['entities']=list(out_entities.values());row['events']=[]
    for proof in proofs:
        for member in proof['members']:member['track_id']=track_alias[(proof['shot_id'],proof['frame'],member['track_id'])]
    for eid,e in entities.items():e['local_ids']=[lid for lid,gid in new_mapping.items() if gid==eid]
    final=deepcopy(catalog);final.update(entities=[e for e in entities.values() if e['local_ids']],mapping=new_mapping,shots=[],
        strategy='repair',one_entity_per_frame=True,physical_groups=proofs,cannot_link=[],
        repair_provenance=dict(version=VERSION,prior_entities=len(catalog['entities']),preserved_source_observations=before_count,reviewed_frames=len(reviews),profile_sources=sources))
    from story_semantics import global_input_fingerprint
    bundle['fingerprint']=hashlib.sha256(json.dumps([bundle.get('fingerprint'),assignments.__repr__()],sort_keys=True).encode()).hexdigest()
    final['global_input_fingerprint']=global_input_fingerprint(bundle,repaired_local)
    audit=dict(version=VERSION,requested_frames=len(selected),reviewed_frames=len(reviews),failures=failures,final_entities=len(final['entities']),physical_groups=len(proofs),generations=sum(e.get('phase','').startswith(('catalog_repair:','global_focused:repair_','batch_frames:')) for e in backend.ledger['events']),strong_generations=sum(e.get('phase','').startswith('global_focused:repair_') for e in backend.ledger['events']),policy='Preserve prior cross-shot links; target-specific trait splits and exact-frame physical-body review only.')
    atomic_json(dest/'audit.json',audit)
    return bundle,repaired_local,final


def build_profile_cache(bundle,local,output,backend):
    """Probe only suspicious within-track changes, without splitting the catalog."""
    from compact_identity import prepare_temporal
    from temporal_profile_review import review_temporal_profiles
    folder=Path(output)/'repair_profile_cache';folder.mkdir(exist_ok=True)
    prepared,entries,audit=prepare_temporal(bundle,local,output)
    entries=review_temporal_profiles(prepared,entries,output,backend,max_calls=4)
    atomic_json(folder/'reviewed_tracking.json',prepared)
    atomic_json(folder/'local_entities.json',entries)
    profiles={e['local_id']:e['target_profile'] for s in entries['shots'] for e in s['entities'] if e.get('target_profile')}
    atomic_json(folder/'compact_identity/profiles.json',profiles)
    atomic_json(folder/'candidates.json',audit)
    return folder
