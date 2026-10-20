"""Small-budget catalog review with hard co-visibility and attribute constraints."""
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
from typing import Literal
from PIL import Image, ImageDraw, ImageOps, ImageFont
from pydantic import BaseModel, Field, field_validator
from low_freq_event.storage import atomic_json
from low_freq_event.backend import ModelOutputError

VERSION = 9


def prepare_temporal(bundle, local, output, threshold=.50):
    """Split abrupt descriptor changes even when crossing people overlap.

    Only existing selected observations/descriptors are used. These are review
    candidates, not assertions that a pose change means a different person.
    """
    import hashlib
    from temporal_track_repair import _features_for_track, _unit
    bundle, local = deepcopy(bundle), deepcopy(local)
    root = Path(bundle.get('asset_root', output))
    audit = []
    for shot in bundle['shots']:
        row = next(s for s in local['shots'] if s['shot_id'] == shot['shot_id'])
        assigned = {tid for e in row['entities'] for tid in e['track_ids']}
        children, replacement, cache = {}, [], {}
        for track in shot['tracks']:
            tid = track['track_id']
            observations = sorted(track['observations'], key=lambda o: o['frame'])
            features = _features_for_track(track, shot, root, cache) if tid in assigned else {}
            cuts = [b['frame'] for a,b in zip(observations, observations[1:])
                    if a['frame'] in features and b['frame'] in features
                    and float(_unit(features[a['frame']]) @ _unit(features[b['frame']])) < threshold]
            if not cuts:
                children[tid] = [tid]; replacement.append(track); continue
            parts = [[] for _ in range(len(cuts)+1)]
            for observation in observations:
                parts[sum(observation['frame'] >= cut for cut in cuts)].append(observation)
            children[tid] = []
            for index, observations in enumerate(parts):
                cid = f'{tid}:D{index+1:03d}'
                child = deepcopy(track)
                for observation in observations:
                    observation.setdefault('source_temporal_track_id', tid)
                    observation['track_id'] = cid
                representatives = []
                for observation in observations:
                    if observation.get('crop'):
                        representative = deepcopy(observation)
                        if observation['frame'] in features:
                            representative['feature'] = list(features[observation['frame']])
                        representatives.append(representative)
                if not representatives:
                    raise ValueError(f'No saved selected-frame crops for {cid}')
                child.update(track_id=cid, observations=observations,
                    representatives=representatives, first_frame=observations[0]['frame'],
                    last_frame=observations[-1]['frame'], observed_frames=len(observations),
                    observed_duration_sec=len(observations)/shot['fps'],
                    span_duration_sec=(observations[-1]['frame']-observations[0]['frame']+1)/shot['fps'],
                    observation_coverage=len(observations)/(observations[-1]['frame']-observations[0]['frame']+1),
                    mean_confidence=sum(o.get('confidence',0) for o in observations)/len(observations),
                    occlusion_gaps=[], uncertain=True,
                    temporal_candidate=dict(parent_track_id=tid, same_identity_allowed=True))
                replacement.append(child); children[tid].append(cid)
                if tid in row.get('track_roles',{}): row['track_roles'][cid] = row['track_roles'][tid]
            audit.append(dict(track_id=tid, cuts=cuts, children=children[tid], threshold=threshold))
        shot['tracks'] = replacement
        candidate_tracks={t['track_id'] for t in replacement if t.get('temporal_candidate')}
        entities, used = [], {e['local_id'] for e in row['entities']}
        for entity in row['entities']:
            if not any(len(children[t]) > 1 for t in entity['track_ids']) and not (len(entity['track_ids'])>1 and any(t in candidate_tracks for t in entity['track_ids'])):
                entities.append(entity); continue
            # A previous local union must not immediately undo the split.
            audit.append(dict(local_id=entity['local_id'],previous_tracks=entity['track_ids'],reason='Reopen temporal candidates for global visual review.'))
            pieces = [cid for tid in entity['track_ids'] for cid in children[tid]]
            for index, cid in enumerate(pieces):
                child = deepcopy(entity)
                if index:
                    suffix = index+1
                    while f"{entity['local_id']}.t{suffix}" in used: suffix += 1
                    child['local_id'] = f"{entity['local_id']}.t{suffix}"
                    used.add(child['local_id'])
                child.update(track_ids=[cid], segments=[], label='재검토 대상', appearance='',
                    uncertain=True, reason='Abrupt appearance change; separate temporal candidate pending visual review.')
                entities.append(child)
            row['events'] = []
        row['entities'] = entities
    if audit:
        bundle['fingerprint'] = hashlib.sha256(json.dumps([bundle.get('fingerprint'), audit],sort_keys=True).encode()).hexdigest()
    return bundle, local, audit


def same_region(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return intersection/max(aa+bb-intersection,1e-9)>=.70 or intersection/max(min(aa,bb),1e-9)>=.95


def prepare_local(bundle,local):
    """Split conflicting local track assignments; never hide separated bodies."""
    local=deepcopy(local);audit=[]
    for shot in bundle['shots']:
        row=next(s for s in local['shots'] if s['shot_id']==shot['shot_id'])
        tracks={t['track_id']:t for t in shot['tracks']}
        boxes={tid:{o['frame']:o['bbox'] for o in t['observations']} for tid,t in tracks.items()}
        used={e['local_id'] for e in row['entities']};out=[]
        for entity in row['entities']:
            conflicts={tuple(sorted((a,b))) for a,b in combinations(entity['track_ids'],2)
                if any(not same_region(boxes[a][f],boxes[b][f]) for f in boxes[a].keys()&boxes[b].keys())}
            groups=[]
            for tid in sorted(entity['track_ids'],key=lambda t:(-sum(t in p for p in conflicts),t)):
                group=next((g for g in groups if all(tuple(sorted((tid,x))) not in conflicts for x in g)),None)
                if group is None:groups.append([tid])
                else:group.append(tid)
            for index,group in enumerate(groups):
                child=deepcopy(entity);child['track_ids']=group
                if index:
                    suffix=index+1;key=f"{entity['local_id']}.u{suffix}"
                    while key in used:suffix+=1;key=f"{entity['local_id']}.u{suffix}"
                    child['local_id']=key;used.add(key)
                if len(groups)>1:
                    child.update(uncertain=True,label='재검토 대상',appearance='',reason='Separated simultaneous regions; visual identity requires new review.')
                    child['segments']=[s for s in child.get('segments',[]) if s.get('track_id') in group]
                out.append(child)
            if len(groups)>1:
                audit.append(dict(original=entity['local_id'],groups=groups,conflicts=[list(p) for p in sorted(conflicts)]))
                row['events']=[]
        row['entities']=out
    return local,audit


def co_visibility(bundle,local):
    tracks={t['track_id']:t for s in bundle['shots'] for t in s['tracks']}
    forbidden=[]
    for shot in local['shots']:
        frames={e['local_id']:{o['frame'] for tid in e['track_ids'] for o in tracks[tid]['observations']} for e in shot['entities']}
        for a,b in combinations(frames,2):
            if not frames[a]&frames[b]:continue
            ea=next(e for e in shot['entities'] if e['local_id']==a);eb=next(e for e in shot['entities'] if e['local_id']==b)
            if any(x['frame']==y['frame'] and not same_region(x['bbox'],y['bbox']) for ta in ea['track_ids'] for tb in eb['track_ids'] for x in tracks[ta]['observations'] for y in tracks[tb]['observations']):forbidden.append(tuple(sorted((a,b))))
    return sorted(set(forbidden))


class Profile(BaseModel):
    id: str
    kind: Literal['person','animal','object','unknown']
    label: str
    appearance: str
    hair: str = Field(default='unknown',min_length=1,max_length=40)

    @field_validator('hair')
    @classmethod
    def normalize_hair(cls,value):
        value=value.strip().lower()
        return {'grey':'gray','blonde':'blond','n/a':'unknown','unclear':'unknown'}.get(value,value) or 'unknown'
    facial_hair: Literal['yes','no','unknown']='unknown'
    presentation: Literal['female','male','unknown']='unknown'
    evidence: str = Field(min_length=1)
    frame: int | None
    verified: bool = True
    mixed: bool = False


class Group(BaseModel):
    members: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class CatalogReview(BaseModel):
    profiles: list[Profile]
    groups: list[Group] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)


def validate_catalog(result,keys,shown):
    pids=[p.id for p in result.profiles]
    gids=[k for g in result.groups for k in g.members]
    if len(pids)!=len(set(pids)) or set(pids)!=set(keys):
        result.validation_issues.append('Missing/duplicate/unknown profile IDs; only uniquely bound records accepted.')
        unique={p.id:p for p in result.profiles if pids.count(p.id)==1 and p.id in keys}
        result.profiles=[unique.get(k) or Profile(id=k,kind='unknown',label='미검토 개체',appearance='',evidence='Missing or duplicate profile; no identity evidence accepted.',frame=None,verified=False) for k in keys]
    if len(gids)!=len(set(gids)) or not set(gids)<=set(keys):
        raise ValueError('Groups must not duplicate IDs or invent IDs.')
    for p in result.profiles:
        if p.frame not in shown[p.id]:
            p.frame=None
            p.verified=False
            p.kind='unknown'
            p.label='근거 위치 미확인'
            p.evidence += ' [Invalid frame citation; retained unresolved, never joined.]'
    # An omitted grouping is NOT positive match evidence: keep it separate.
    for key in keys:
        if key not in gids:
            result.groups.append(Group(members=[key],reason='No grouping proposed; retained as an independent identity.'))
    return result


PROMPT='''Review the labeled target crops to build a consistent physical-identity catalog.
The image is a grid of labeled CARDS. Read every bold candidate ID and inspect EACH card individually. Do not repeat a generic description across IDs.
Each card has up to 3 chronological/varied views of a candidate track. A tracking error MAY switch people across views or include two people. Set mixed=true when the marked target is inconsistent or ambiguous across views, and NEVER merge that card. Otherwise mixed=false.
Identify only the marked candidate, not a nearby rider/carrier or background. Existing names are deliberately withheld.
kind MUST be exactly person, animal, object, or unknown. Animals and fictional creatures use animal; never use creature or nonperson.
First describe EACH supplied ID from visible evidence. For people, hair/baldness, facial hair and facial presentation are stable discriminators. If hidden, tiny, silhouetted or uncertain, use unknown, never guess. For nonpeople these attributes MUST be unknown.
Then partition ALL IDs into groups. Join ONLY when face geometry or distinctive anatomy AND additional stable visual detail positively agree. A shared color, clothing, species, scene, role or vague silhouette alone is NOT sufficient. Do not force uncertain candidates into a group.
Contradictory visible hair coverage, hair color, facial hair, face geometry or anatomy must prevent a merge. Never assume a protagonist or expected cast size.
Respect every cannot_link pair: simultaneous distinct identities cannot be the same. Mirrors are not supported.
A group must be consistent with EVERY member, not just one matching anchor. When uncertain, use singletons.
Return concise JSON: {"profiles":[{"id":"C0001","kind":"person","label":"short Korean visual label","appearance":"short visible description","hair":"red|blond|brown|black|gray|white|bald|unknown","facial_hair":"yes|no|unknown","presentation":"female|male|unknown","evidence":"visible details supporting the profile","frame":0,"mixed":false}],"groups":[{"members":["C0001","C0002"],"reason":"specific visible agreement across all members"}]}.
Profiles and groups must each cover every supplied ID exactly once. Cite one supplied frame in each profile. Do not copy anatomy/attributes from nearby subjects. JSON only.
'''


def incompatible(a,b):
    if a.get('mixed') or b.get('mixed'):return 'mixed_track'
    if a['kind']=='unknown' or b['kind']=='unknown':return 'unresolved_kind'
    if a['kind']!='unknown' and b['kind']!='unknown' and a['kind']!=b['kind']:return 'kind'
    if a['kind']==b['kind']=='person':
        for key in ('hair','facial_hair','presentation'):
            if a.get(key,'unknown')!='unknown' and b.get(key,'unknown')!='unknown' and a[key]!=b[key]:
                if key!='hair' or 'bald' in (a[key],b[key]):return key
    return None


def merge_allowed(a,b,forbidden,profiles):
    if any(tuple(sorted((x,y))) in forbidden for x in a['local_ids'] for y in b['local_ids']):return 'co_visible'
    for x in a['local_ids']:
        for y in b['local_ids']:
            if x in profiles and y in profiles:
                reason=incompatible(profiles[x],profiles[y])
                if reason:return 'incompatible_'+reason
    return None


def visual_board(bundle,entries,groups,keys,output):
    tracks={t['track_id']:t for s in bundle['shots'] for t in s['tracks']}
    root=Path(bundle.get('asset_root',output));pages=[];shown={};sources={}
    columns=6;cell_w,cell_h=300,360
    board=Image.new('RGB',(columns*cell_w,cell_h*((len(keys)+columns-1)//columns)),(242,245,249))
    draw=ImageDraw.Draw(board)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title_font=ImageFont.truetype(font_path,23) if Path(font_path).is_file() else ImageFont.load_default(size=23)
    small_font=ImageFont.truetype(font_path,13) if Path(font_path).is_file() else ImageFont.load_default(size=13)
    for index,key in enumerate(keys):
        options=[]
        for lid in groups[key]['local_ids']:
            for tid in entries[lid]['track_ids']:
                t=tracks[tid]
                for o in t.get('representatives',[]) or t['observations']:
                    if o.get('crop') and (root/o['crop']).is_file():options.append((lid,o))
        options.sort(key=lambda x:((x[1]['bbox'][2]-x[1]['bbox'][0])*(x[1]['bbox'][3]-x[1]['bbox'][1])),reverse=True)
        selected=[];seen=set()
        # Show distinct local members before taking another view of one member.
        for lid,o in options:
            if lid not in seen:selected.append((lid,o));seen.add(lid)
            if len(selected)==3:break
        for lid,o in options:
            if len(selected)==3:break
            if (lid,o['frame']) not in {(x,y['frame']) for x,y in selected}:selected.append((lid,o))
        if not selected:raise ValueError(f'No visual evidence for {key}')
        shown[key]=[o['frame'] for _,o in selected];sources[key]=[dict(local_id=lid,frame=o['frame'],crop=o['crop']) for lid,o in selected]
        x=(index%columns)*cell_w;y=(index//columns)*cell_h
        draw.rectangle((x+1,y+1,x+cell_w-2,y+cell_h-2),outline=(100,120,140),width=2)
        draw.text((x+8,y+4),key,fill='black',font=title_font)
        for n,(lid,o) in enumerate(selected):
            size=(282,210) if n==0 else (138,84)
            with Image.open(root/o['crop']) as im:thumb=ImageOps.contain(im.convert('RGB'),size)
            px=x+9 if n==0 else x+9+(n-1)*144
            py=y+32 if n==0 else y+264
            board.paste(thumb,(px+(size[0]-thumb.width)//2,py))
            draw.text((px,py-15 if n else y+244),f"f{o['frame']}",fill='black',font=small_font)
    pages.append(('Candidate catalog: each bold ID labels one card, read all cards.',board))
    return pages,shown,sources


def canonical(group,profiles):
    rows=[profiles[x] for x in group['local_ids'] if x in profiles]
    if not rows:return dict(kind=group['kind'],label='미확정 개체',appearance='',uncertain=True)
    # Evidence completeness, then deterministic ID order; never description length.
    best=max(rows,key=lambda r:sum(r.get(k,'unknown')!='unknown' for k in ('hair','facial_hair','presentation')))
    kind=best['kind'];label=best['label']
    if any(r.get('mixed') for r in rows):return dict(kind='unknown',label='대상 혼합 의심 관측',appearance=best['appearance'],uncertain=True,reason='Mixed/ambiguous target views; no identity join permitted.')
    if kind=='person':
        hair={'red':'붉은 머리','blond':'금발','brown':'갈색 머리','black':'검은 머리','gray':'회색 머리','white':'흰 머리','bald':'대머리','pink':'분홍 머리','blue':'파란 머리','green':'초록 머리','purple':'보라 머리','orange':'주황 머리'}.get(best['hair'],'')
        face={'female':'여성','male':'남성','unknown':'인물'}[best['presentation']]
        label=' '.join(x for x in (hair,'수염 있는' if best['facial_hair']=='yes' else '',face) if x)
    return dict(kind=kind,label=label,appearance=best['appearance'],reason=best.get('evidence','Visual profile; identity joins recorded separately.'),uncertain=(kind=='unknown' or (kind=='person' and all(r['hair']=='unknown' and r['presentation']=='unknown' for r in rows))))


def select_anchors(bundle,entries,groups,profiles,fresh,count):
    """Retrieve previously reviewed groups; similarity only selects images."""
    import numpy as np
    tracks={t['track_id']:t for shot in bundle['shots'] for t in shot['tracks']}
    def vector(group):
        vectors=[o['feature'] for lid in group['local_ids'] for tid in entries[lid]['track_ids']
                 for o in tracks[tid].get('representatives',[]) if o.get('feature')]
        if not vectors:return None
        value=np.mean(vectors,axis=0);return value/max(float(np.linalg.norm(value)),1e-9)
    targets=[v for k in fresh if (v:=vector(groups[k])) is not None]
    ranked=[]
    for key,group in groups.items():
        if key in fresh or not any(lid in profiles for lid in group['local_ids']):continue
        if any(profiles.get(lid,{}).get('mixed') for lid in group['local_ids']):continue
        value=vector(group)
        score=max((float(value@target) for target in targets),default=-1) if value is not None else -1
        ranked.append((score,len(group['local_ids']),key))
    return [row[2] for row in sorted(ranked,reverse=True)[:count]]


def reconcile_compact(bundle,local,provisional,output,backend,*,max_calls=5,context=None):
    from story_semantics import local_lookup,BudgetExhausted,GlobalDecision
    entries=local_lookup(local);forbidden=set(co_visibility(bundle,local))
    from story_semantics import cannot_links
    forbidden.update(tuple(sorted(p)) for p in cannot_links(bundle,local))
    groups={f'C{i+1:04d}':dict(e.model_dump()) for i,e in enumerate(provisional.entities)}
    profiles={lid:deepcopy(e['target_profile']) for lid,e in entries.items() if e.get('target_profile',{}).get('target_verified')};reviews=[];errors=[];blocked=[];initial=backend.ledger['generations'];original_reserve=backend.reserve
    dest=Path(output)/'compact_identity';dest.mkdir(parents=True,exist_ok=True)
    atomic_json(dest/'snapshot_000.json',dict(groups=groups,profiles=profiles))
    # The first pass reviews each local candidate once; extra calls compare their groups.
    keys=sorted(groups,key=lambda k:(({'person':0,'animal':1,'object':2,'unknown':3}.get(groups[k]['kind'],3)),k));queue=[keys[:18]];pending=keys[18:]
    def consumed():return sum(e.get('phase','').startswith('global_catalog:') for e in backend.ledger['events'])
    requests=0;previous_signature=None
    while queue or pending:
        if not queue:
            fresh,pending=pending[:14],pending[14:]
            anchors=select_anchors(bundle,entries,groups,profiles,fresh,4)
            queue=[anchors+fresh]
        keys=[k for k in queue.pop(0) if k in groups]
        if not keys:continue
        requests+=1;batch=dest/f'{requests:03d}';batch.mkdir(exist_ok=True)
        images,shown,sources=visual_board(bundle,entries,groups,keys,output)
        cannot=[[a,b] for a,b in combinations(keys,2) if merge_allowed(groups[a],groups[b],forbidden,profiles)]
        inventory=dict(ids=keys,shown_frames=shown,cannot_link=cannot)
        inventory['targeted_local_observations']={k:[{f:profiles[lid][f] for f in ('kind','hair','facial_hair','presentation','evidence')} for lid in groups[k]['local_ids'] if profiles.get(lid,{}).get('target_verified')] for k in keys}
        inventory['targeted_local_observations']={k:v for k,v in inventory['targeted_local_observations'].items() if v}
        atomic_json(batch/'input.json',dict(inventory,sources=sources))
        for n,(_,im) in enumerate(images):im.save(batch/f'evidence_{n:02d}.jpg',quality=90)
        def validate(result):return validate_catalog(result,keys,shown)
        backend.phase=f'global_catalog:{requests:03d}'
        backend.reserve=max(original_reserve,backend.remaining-max(0,max_calls-consumed()))
        before=deepcopy(groups)
        try:
            options={}
            if 'thinking' in str(backend.settings.model_id).lower():
                options['generation_options']=dict(do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,seed=0,thinking_token_budget=2048)
            response=backend.request(f'Protocol {VERSION}\n'+PROMPT+'\nINPUT: '+json.dumps(inventory),CatalogReview,images=images,max_tokens=6000,image_pixels=1572864,validate=validate,repair_json_closures=True,**options)
            atomic_json(batch/'decision.json',response.model_dump())
        except (ModelOutputError,BudgetExhausted) as exc:
            errors.append(dict(batch=requests,error=str(exc)));atomic_json(batch/'failure.json',errors[-1]);continue
        for p in response.profiles:
            row=p.model_dump()
            if row['kind']!='person':row.update(hair='unknown',facial_hair='unknown',presentation='unknown')
            # Preserve original per-local visual profiles across aggregate reviews.
            for lid in groups[p.id]['local_ids']:
                if lid not in profiles:profiles[lid]=dict(row)
        joins=[]
        for proposal in response.groups:
            anchor=proposal.members[0]
            for key in proposal.members[1:]:
                reason=merge_allowed(groups[anchor],groups[key],forbidden,profiles)
                if reason:blocked.append(dict(batch=requests,left=anchor,right=key,reason=reason));continue
                groups[anchor]['local_ids']+=groups.pop(key)['local_ids']
                joins.append(dict(target=anchor,merged=key,reason=proposal.reason))
        for group in groups.values():group.update(canonical(group,profiles))
        reviews.append(dict(batch=requests,joins=joins))
        atomic_json(batch/'changes.json',dict(joins=joins,blocked=[b for b in blocked if b['batch']==requests]))
        calls_through_batch=sum(e.get('phase','').startswith('global_catalog:') and int(e['phase'].split(':')[-1])<=requests for e in backend.ledger['events'])
        atomic_json(dest/f'snapshot_{calls_through_batch:03d}.json',dict(groups=groups,profiles=profiles))
        if not queue and not pending:
            signature=sorted(sorted(g['local_ids']) for g in groups.values())
            if previous_signature==signature:break
            previous_signature=signature
            # A broad second pass sees all remaining identities when they fit;
            # larger catalogs stay bounded and visibly unreviewed beyond the budget.
            if consumed()>=max_calls and not any(e.get('phase')==f'global_catalog:{requests+1:03d}' for e in backend.ledger['events']):break
            keys=sorted(groups,key=lambda k:(({'person':0,'animal':1,'object':2,'unknown':3}.get(groups[k]['kind'],3)),k))
            queue=[keys[:18]];pending=keys[18:]
    backend.reserve=original_reserve
    for group in groups.values():group.update(canonical(group,profiles))
    unreviewed=sorted((set(entries)-set(profiles))|{lid for lid,p in profiles.items() if not p.get('verified',True)})
    result=GlobalDecision(entities=list(groups.values()),rejected_local_ids=provisional.rejected_local_ids,rejection_reasons=provisional.rejection_reasons)
    audit=dict(version=VERSION,strategy='compact',generations=consumed(),new_generations=backend.ledger['generations']-initial,max_generations=max_calls,
        completed_requests=len(reviews),initial_entities=len(provisional.entities),final_entities=len(groups),
        cannot_link=[list(p) for p in sorted(forbidden)],reviews=reviews,errors=errors,blocked_merges=blocked,
        unreviewed_local_ids=unreviewed,status='needs_review' if errors or unreviewed or any(e.uncertain for e in result.entities) else 'awaiting_evaluation',
        note='No forced joins. Call limit includes JSON retries. Unknown traits are not positive identity evidence.')
    atomic_json(dest/'audit.json',audit);atomic_json(dest/'profiles.json',profiles)
    return result,audit


def finalize_compact(value,bundle,local,output):
    """Quarantine unresolved temporal switches; never let a gallery undo them.

    A candidate interval is not a confirmed identity. Until a target-specific
    check resolves it, keep it separate and avoid a confident personal label.
    """
    import hashlib
    from low_freq_event.backend import parse_json_response
    from story_semantics import local_lookup
    value=deepcopy(value);output=Path(output);entries=local_lookup(local)
    tracks={t['track_id']:t for shot in bundle['shots'] for t in shot['tracks']}
    held={lid for lid,e in entries.items() if any(tracks[t].get('temporal_candidate') for t in e['track_ids']) and not e.get('target_profile',{}).get('target_verified')}
    path=output/'compact_identity/profiles.json'
    profiles=json.loads(path.read_text()) if path.is_file() else {}
    recovered=[]
    ledger=json.loads((output/'qwen_budget.json').read_text()) if (output/'qwen_budget.json').exists() else {'events':[]}
    # Preserve failed batches. Recover only individually valid, singleton-bound
    # profiles from them, NEVER their proposed identity joins.
    for event in ledger['events']:
        if not event.get('phase','').startswith('global_catalog:'):continue
        batch=event['phase'].split(':')[-1];folder=output/'compact_identity'/batch
        trace_path=output/'review_traces'/f"{event['number']:06d}.json"
        if not (folder/'failure.json').exists() or not trace_path.exists():continue
        try:
            trace=json.loads(trace_path.read_text());inventory=json.loads((folder/'input.json').read_text())
            data=parse_json_response(trace['response'],repair_json_closures=True)
            raw=[p for p in data.get('profiles',[]) if isinstance(p,dict)]
            counts={p.get('id'):sum(q.get('id')==p.get('id') for q in raw) for p in raw}
            for item in raw:
                p=Profile.model_validate(item)
                lids={row['local_id'] for row in inventory['sources'].get(p.id,[])}
                if counts[p.id]!=1 or len(lids)!=1 or p.frame not in inventory['shown_frames'].get(p.id,[]):continue
                lid=next(iter(lids))
                # Complete source membership is required; old inventories only
                # support singleton candidates with this original C identifier.
                initial=json.loads((output/'compact_identity/snapshot_000.json').read_text())['groups']
                if inventory.get('members',{}).get(p.id,initial.get(p.id,{}).get('local_ids'))!=[lid]:continue
                if lid in profiles:continue
                profiles[lid]=p.model_dump()
                recovered.append(dict(local_id=lid,trace=str(trace_path.relative_to(output)),sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),policy='Profile only; no joins recovered.'))
        except (ValueError,KeyError,TypeError,OSError):continue
    if recovered:atomic_json(output/'compact_identity/recovered_profiles.json',dict(profiles=profiles,recovered=recovered))
    # Always derive final validation from the preserved model checkpoint, so
    # replaying validation cannot irreversibly erase accepted stable links.
    snapshots=sorted((output/'compact_identity').glob('snapshot_*.json'))
    if snapshots:
        checkpoint=json.loads(snapshots[-1].read_text())
        saved=list(checkpoint['groups'].values())
        members=[lid for group in saved for lid in group['local_ids']]
        if len(members)!=len(set(members)) or set(members)!=set(value['mapping']):
            raise ValueError('Compact checkpoint membership does not match the current catalog')
        value['entities']=saved
    groups=[]
    for entity in value['entities']:
        members=entity['local_ids']
        # Remove unresolved intervals while preserving accepted links among
        # the remaining members; their all-member trait constraints still hold.
        stable=[lid for lid in members if lid not in held]
        partitions=([stable] if stable else [])+[[lid] for lid in members if lid in held]
        for members in partitions:
            row=deepcopy(entity);row['local_ids']=members
            row.update(canonical(row,profiles))
            if held.intersection(members):
                row.update(label='식별 미확정 · 추적 전환 후보',appearance='',uncertain=True,
                    reason='Temporal identity switch candidate; held separate pending target-specific visual confirmation.',
                    identity_status='unresolved')
            groups.append(row)
    candidates={str(i):e for i,e in enumerate(groups)}
    face_joins=[];forbidden=set(co_visibility(bundle,local))
    forbidden.update(tuple(sorted(pair)) for pair in value.get('cannot_link',[]))
    for proof in same_source_face_pairs(bundle,local,output):
        owners={lid:key for key,e in candidates.items() for lid in e['local_ids']}
        a,b=owners[proof['left']],owners[proof['right']]
        if a==b or held.intersection(candidates[a]['local_ids']+candidates[b]['local_ids']):continue
        if merge_allowed(candidates[a],candidates[b],forbidden,profiles):continue
        candidates[a]['local_ids']+=candidates.pop(b)['local_ids']
        candidates[a].update(canonical(candidates[a],profiles));face_joins.append(proof)
    if not face_joins:face_joins=value.get('compact_validation',{}).get('source_face_joins',[])
    groups=list(candidates.values())
    order={lid:i for i,lid in enumerate(entries)}
    groups.sort(key=lambda e:min(order[lid] for lid in e['local_ids']))
    mapping={}
    for i,entity in enumerate(groups,1):
        entity['entity_id']=f'G{i:04d}'
        for lid in entity['local_ids']:mapping[lid]=entity['entity_id']
    value.update(entities=groups,mapping=mapping,compact_validation=dict(
        quarantined_local_ids=sorted(held),recovered_profiles=recovered,source_face_joins=face_joins,
        policy='Temporal candidates without a verified target-specific profile remain separate; no new model calls and no discarded observations.'))
    atomic_json(output/'compact_identity/final_validation.json',value['compact_validation'])
    return value


def same_source_face_pairs(bundle,local,output):
    """Positive duplicate evidence: the SAME face at the SAME source position.

    This is geometric deduplication of verified person targets, not face
    recognition across frames. Distinct people cannot share these landmarks.
    """
    import numpy as np
    from view_quality import face_detail
    from shot_tracking import box_iou
    tracks={t['track_id']:t for s in bundle['shots'] for t in s['tracks']}
    root=Path(bundle.get('asset_root',output));pairs=[]
    def source_face(obs):
        detail=face_detail(bundle,obs,output)
        if not detail or detail['confidence']<.85:return None
        with Image.open(root/obs['crop']) as im:width,height=im.size
        x,y,x2,y2=obs['bbox'];sx=(x2-x)/width;sy=(y2-y)/height
        f=detail['detection'];box=[x+f[0]*sx,y+f[1]*sy,x+(f[0]+f[2])*sx,y+(f[1]+f[3])*sy]
        points=np.asarray([[x+f[i]*sx,y+f[i+1]*sy] for i in range(4,14,2)])
        return box,points
    for shot in local['shots']:
        candidates=[e for e in shot['entities'] if e.get('target_profile',{}).get('target_verified') and e['kind']=='person']
        for a,b in combinations(candidates,2):
            found=None
            for ta in a['track_ids']:
                for tb in b['track_ids']:
                    for x in tracks[ta]['observations']:
                        for y in tracks[tb]['observations']:
                            if x['frame']!=y['frame'] or not same_region(x['bbox'],y['bbox']):continue
                            fa,fb=source_face(x),source_face(y)
                            if fa is None or fb is None:continue
                            scale=min(fa[0][2]-fa[0][0],fa[0][3]-fa[0][1],fb[0][2]-fb[0][0],fb[0][3]-fb[0][1])
                            distance=float(np.linalg.norm(fa[1]-fb[1],axis=1).mean())
                            if box_iou(fa[0],fb[0])>=.85 and distance<=.08*scale:
                                found=dict(left=a['local_id'],right=b['local_id'],shot_id=shot['shot_id'],frame=x['frame'],left_face=fa[0],right_face=fb[0],landmark_distance=distance)
            if found:pairs.append(found)
    return pairs
