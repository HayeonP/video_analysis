"""Shot memory, independent reverse matching, and evidence-backed reconciliation.

No movie-specific IDs or manual merge rules. Forward/reverse IDs are provisional;
only the reconciler produces the IDs used by the exported catalog and video.
"""
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import json
import logging
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field

from low_freq_event.backend import ModelOutputError
from low_freq_event.detection import associate_frame, reconcile_tracklets
from low_freq_event.identity import canonical
from low_freq_event.storage import atomic_json
from low_freq_event.video import Clip, Frame
from entity_models import Detection, MergeReview, iou
from entity_output import font, representative_score
from context_identity import POLICY, adjudicate_pair

VERSION = 2
LOG = logging.getLogger('entities')


class Addition(Detection):
    frame_key: str
    local_id: str


class ShotEntity(BaseModel):
    ids: list[str] = Field(min_length=1)
    label: str = Field(min_length=1, max_length=60)
    appearance: str = Field(min_length=1, max_length=350)
    kind: Literal['person', 'animal', 'object']
    decision: Literal['same', 'new', 'uncertain']
    match_id: str | None = None
    reason: str


class ShortLabel(BaseModel):
    label: str = Field(min_length=1,max_length=24)


class ShotReview(BaseModel):
    scene_summary: str = ''
    entities: list[ShotEntity]
    additions: list[Addition] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)


class MemoryMatch(BaseModel):
    local_id: str
    decision: Literal['same', 'new', 'uncertain']
    match_id: str | None = None
    reason: str


class MemoryReview(BaseModel):
    scene_summary: str = ''
    matches: list[MemoryMatch]


def references(observed, output, limit=3):
    ranked = sorted(observed, key=lambda o: representative_score(o, output), reverse=True)
    selected = []
    # Retain clear old views as new views arrive; prefer different shots first.
    for o in ranked:
        if not selected or o['shot_id'] not in {s['shot_id'] for s in selected}:
            selected.append(o)
        if len(selected) == limit:
            return selected
    for o in ranked:
        if o not in selected and all(abs(o['full_time_sec']-s['full_time_sec']) > .15 for s in selected):
            selected.append(o)
        if len(selected) == limit:
            break
    return selected


def gallery(entries, output, folder, prefix):
    """Page every entity, never silently discard old IDs with a top-k cutoff."""
    images = []; metadata = []
    keys = list(entries)
    for start in range(0, len(keys), 6):
        page = keys[start:start+6]
        sheet = Image.new('RGB', (960, 250*((len(page)+2)//3)), '#202020')
        draw = ImageDraw.Draw(sheet)
        for i, eid in enumerate(page):
            entry = entries[eid]; refs = references(entry['observations'], output)
            x = i % 3 * 320; y = i // 3 * 250
            draw.text((x+4, y+3), eid, fill='white', font=font(18))
            for j, obs in enumerate(refs):
                with Image.open(output/obs['crop']) as source:
                    crop = ImageOps.contain(source.convert('RGB'), (103, 174))
                sheet.paste(crop, (x+j*106+(103-crop.width)//2, y+28+(174-crop.height)//2))
            draw.text((x+4, y+209), entry['label'][:17], fill='white', font=font(16))
            metadata.append(dict(id=eid, label=entry['label'], kind=entry['kind'],
                appearance=entry['appearance'], history=entry.get('history', []), shots=sorted({o['shot_id'] for o in entry['observations']}),
                references=[dict(frame_key=o['frame_key'], crop=o['crop'],
                    original_time_sec=o['original_time_sec']) for o in refs]))
        sheet.save(folder/f'{prefix}_{start//6:03d}.jpg', quality=94)
        images.append((f'{prefix} IDs {page}', sheet))
    return images, metadata


def scene_clip(frames, entries, output, fps):
    by_frame = defaultdict(list)
    for eid, e in entries.items():
        for o in e['observations']:
            by_frame[o['frame_key']].append((eid, o))
    video = []
    for f in frames:
        with Image.open(output/f['image']) as source:
            im = source.convert('RGB').resize((768, 432))
        draw = ImageDraw.Draw(im)
        for eid, o in by_frame[f['frame_key']]:
            xy = [v*s for v,s in zip(o['bbox'], [.768,.432,.768,.432])]
            draw.rectangle(xy, outline='#ffff00', width=2)
            draw.text((xy[0], max(24, xy[1]-18)), eid, fill='#ffff00', font=font(14))
        draw.rectangle((0,0,768,24), fill='black')
        draw.text((3,3), f"{f['frame_key']} | original {f['original_time_sec']:.3f}s", fill='white', font=font(14))
        video.append(Frame(f['full_time_sec'], im))
    return Clip(video[0].time_sec, frames[-1]['full_time_sec']+frames[-1]['hold_frames']/fps, fps, video)


def local_seeds(observed, settings, shot_index):
    registry = {}; visual = {}
    by_frame = defaultdict(list)
    for o in observed:
        by_frame[o['frame_key']].append(o)
    for batch in by_frame.values():
        associate_frame(batch, registry, visual, settings)
    if observed:
        reconcile_tracklets(registry, visual, settings)
    pools = defaultdict(list)
    for o in observed:
        eid = canonical(registry, visual['assignments'][o['detection_id']]['entity_id'])
        pools[eid].append(o)
    entries = {}
    for i, pool in enumerate(pools.values(), 1):
        first = pool[0]
        entries[f'L{shot_index:03d}_{i:03d}'] = dict(label=first['label'],
            kind=first['entity_kind'], appearance=first['appearance'], observations=pool)
    return entries


def visible_frames(entry):
    return {o['frame_key'] for o in entry['observations']}


def snapshot(memory):
    return {eid: dict(label=e['label'], kind=e['kind'], appearance=e['appearance'],
        members=e['members'], history=e.get('history', []), shots=sorted({o['shot_id'] for o in e['observations']}),
        observations=[o['detection_id'] for o in e['observations']]) for eid,e in memory.items()}


def update_memory(memory, group_id, entry, decision, match_id, prefix):
    target = match_id if decision == 'same' else None
    blocked = bool(target and visible_frames(memory[target]) & visible_frames(entry))
    if blocked:
        target = None
    if target is None:
        target = f'{prefix}{len(memory)+1:04d}'
        memory[target] = dict(label=entry['label'], appearance=entry['appearance'],
            kind=entry['kind'], members=[], observations=[], history=[])
    memory[target]['members'].append(group_id)
    memory[target]['observations'].extend(entry['observations'])
    memory[target]['history'].append(dict(shot_id=entry['shot_id'], label=entry['label'],
        appearance=entry['appearance'], scene_summary=entry.get('scene_summary', ''),
        match_reason=entry.get('match_reason', ''), decision=decision))
    return target, blocked


def review_shot(shot_id, frames, current, memory, proposals, output, backend, fps, folder, history):
    images, current_meta = gallery(current, output, folder, 'current')
    memory_images, memory_meta = gallery(memory, output, folder, 'memory')
    frame_map = {f['frame_key']: f for f in frames}
    forbidden = [list(p) for p in combinations(current, 2)
        if visible_frames(current[p[0]]) & visible_frames(current[p[1]])]
    prompt = POLICY + '''Review this ONE current shot and update a persistent entity memory.
The video contains EVERY adaptive sample of this shot, in playback order. Yellow boxes mark preliminary local IDs. Current crop pages contain multiple views of each local ID. MEMORY pages contain earlier shots' distinct individuals, with their IDs and multiple views. All names/classifications are provisional visual descriptions, not ground truth.
1. Inspect the complete video and YOLO hints. Add clearly visible foreground people, animals/fantasy creatures or salient objects missing from the boxes. For each added observation provide its exact supplied frame_key and tight bbox [x1,y1,x2,y2] normalized 0..1000. Use M1, M2 etc for new local IDs (reuse the same M ID across frames for the same individual). An addition to an existing local ID may use that ID only on a frame with no existing box for it. Never guess visibility between samples. No duplicate boxes, scenery, shadows, text, body parts or clothing as separate entities.
2. Consolidate fragments of the SAME physical individual within the current shot. Return every current local ID and added M ID exactly once, either in entities[].ids or rejected_ids. Reject only a clear false positive across its observations; darkness/uncertainty alone is not grounds for deletion. Forbidden pairs are distinct simultaneously visible subjects and must not be grouped.
3. Give each resulting entity a short Korean visual label (roughly 2-5 words), kind and stable appearance description. Do not use movie knowledge or invented proper names.
4. Compare EACH current entity against ALL MEMORY IDs, including long-absent people. Same individual may have different pose, lighting, clothing or apparent size; compare visible face, hair, anatomy and context. Carrier and passenger are different. Species, narrative role or a similar label alone do not establish identity. If clearly same, decision=same and match_id=exact MEMORY ID; otherwise new or uncertain with match_id=null. A shot can have no entities. Do not copy absent MEMORY subjects into the shot.
Return ONLY JSON: {"scene_summary":"brief observed scene", "entities":[{"ids":["local ID"],"label":"짧은 머리 여성","appearance":"visible stable traits","kind":"person","decision":"same|new|uncertain","match_id":null,"reason":"specific comparison evidence"}],"additions":[{"local_id":"M1","frame_key":"exact key","kind":"animal","category":"dragon","label":"작은 용","appearance":"visible features","bbox":[1,2,100,200],"uncertain":false}],"rejected_ids":[]}.
'''
    inventory = dict(version=VERSION, shot=shot_id, current=current_meta, memory=memory_meta,
        forbidden_pairs=forbidden, shot_history=history,
        frames=[dict(frame_key=f['frame_key'], time=f['original_time_sec'],
            yolo=proposals[f['frame_key']]) for f in frames])

    def validate(result):
        additions = defaultdict(list)
        for a in result.additions:
            if a.frame_key not in frame_map or (a.local_id not in current and not a.local_id.startswith('M')):
                raise ValueError('Additions require supplied frame_key and existing local ID or M-prefixed new ID.')
            existing = [o for e in current.values() for o in e['observations'] if o['frame_key'] == a.frame_key]
            if any(iou(a.bbox,o['bbox']) > .8 for o in existing):
                raise ValueError('Added box duplicates a current detection; group existing IDs instead.')
            if any(a.frame_key == b.frame_key and iou(a.bbox,b.bbox) > .8 for values in additions.values() for b in values):
                raise ValueError('Duplicate added boxes.')
            additions[a.local_id].append(a)
        allowed = set(current) | set(additions)
        used = list(result.rejected_ids)
        normalized=[]
        for entity in result.entities:
            if not set(entity.ids)<=allowed:
                raise ValueError('MEMORY IDs must NEVER appear in entities[].ids. Only current IDs: '+str(sorted(allowed)))
            used.extend(entity.ids)
            if entity.decision == 'same':
                if entity.match_id not in memory:
                    # A local consolidation is sometimes incorrectly called a
                    # memory match. Never invent its target or discard good boxes.
                    entity.reason += f' [Invalid memory reference {entity.match_id!r}; retained as uncertain.]'
                    entity.decision='uncertain';entity.match_id=None
            elif entity.match_id is not None:
                entity.reason += ' [Unconfirmed memory reference cleared.]'
                entity.match_id=None
            occupied = []
            for eid in entity.ids:
                occupied.extend(o['frame_key'] for o in current.get(eid,{}).get('observations',[]))
                occupied.extend(a.frame_key for a in additions[eid])
            if len(occupied) != len(set(occupied)):
                # Do not let one impossible proposed merge discard other valid
                # entities. Preserve its original local IDs independently.
                for eid in entity.ids:
                    entry=current.get(eid)
                    if entry is None:
                        raise ValueError('Added identity has two boxes in one frame.')
                    normalized.append(ShotEntity(ids=[eid],label=entry['label'],appearance=entry['appearance'],
                        kind=entry['kind'],decision='uncertain',match_id=None,
                        reason='Rejected co-visible group; preserved this original local identity.'))
            else:
                normalized.append(entity)
        if set(used) != allowed or len(used) != len(set(used)):
            raise ValueError(f'Partition each local ID exactly once: {sorted(allowed)}')
        result.entities=normalized
        return result

    atomic_json(folder/'input.json', inventory)
    try:
        ending='\nOUTPUT SCOPE: CURRENT IDs are '+json.dumps(list(current))+'. MEMORY IDs may appear ONLY in match_id, NEVER in ids. Output only current visible entities, not the memory catalog. If no current entities and no visible missed subject with an explicit new box, return entities=[], additions=[], rejected_ids=[]. Do not reproduce input metadata.'
        result = backend.request(prompt+json.dumps(inventory,ensure_ascii=False)+ending, ShotReview,
            clip=scene_clip(frames,current,output,fps), images=images+memory_images,
            validate=validate, image_pixels=491520, max_tokens=4096)
        atomic_json(folder/'result.json', dict(status='complete', **result.model_dump()))
        return result
    except ModelOutputError as exc:
        # Retain detections as uncertain, rather than interpreting an invalid answer as absence.
        result = ShotReview(entities=[ShotEntity(ids=[eid],label=e['label'],appearance=e['appearance'],
            kind=e['kind'],decision='uncertain',reason='Structured-output failure; retained without merge') for eid,e in current.items()])
        atomic_json(folder/'result.json', dict(status='uncertain',error=str(exc),**result.model_dump()))
        return result


def materialize_additions(result, current, frames, output):
    fmap = {f['frame_key']:f for f in frames}
    for index, a in enumerate(result.additions):
        f = fmap[a.frame_key]; path = f'crops/{a.frame_key}_supplement_{index:03d}.jpg'
        (output/'crops').mkdir(exist_ok=True)
        with Image.open(output/f['image']) as source:
            image = source.convert('RGB'); w,h = image.size
            x1,y1,x2,y2 = a.bbox
            image.crop((int(x1*w/1000),int(y1*h/1000),max(int(x1*w/1000)+1,int(np.ceil(x2*w/1000))),
                max(int(y1*h/1000)+1,int(np.ceil(y2*h/1000))))).save(output/path,quality=95)
        obs = dict(f, detection_id=f'{a.frame_key}_supplement_{index:03d}',
            time_sec=f['original_time_sec'], bbox=a.bbox, crop=path, source_size=[w,h],
            kind='person' if a.kind=='person' else 'object', entity_kind=a.kind,
            detector_class=a.category,label=a.label,appearance=a.appearance,
            detection_confidence=None,grounding_source='Qwen3-VL shot-memory supplement',detection_uncertain=a.uncertain)
        current.setdefault(a.local_id,dict(label=a.label,appearance=a.appearance,kind=a.kind,observations=[]))['observations'].append(obs)


def reverse_pass(groups, shots, frames, output, backend, fps, root):
    memory = {}; assignments = {}; edges = []; events = []; history = []
    for shot in reversed(shots):
        sid = shot['id']; folder = root/'reverse'/sid; folder.mkdir(parents=True,exist_ok=True)
        current = {k:e for k,e in groups.items() if e['shot_id']==sid}
        images, cm = gallery(current,output,folder,'current')
        mi, mm = gallery(memory,output,folder,'memory')
        inventory = dict(version=VERSION,direction='reverse',current=cm,memory=mm,shot_history=history)
        prompt = POLICY + '''Independently match this shot's local individuals to memory accumulated from LATER shots.
This is a reverse traversal of shot order, NOT reversed playback inside a shot. No forward-pass assignments are supplied. Compare all current IDs with all memory IDs using actual boxed subjects, faces/anatomy and multiple views. Labels are only hints. Long gaps, pose, lighting or clothes can change. Same species/role alone is insufficient. Carrier and passenger must remain separate. If uncertain keep a new provisional identity; do not force a match.
Return every current local_id exactly once. same requires an exact memory ID; new/uncertain requires null.
ONLY JSON: {"scene_summary":"brief observed shot, actions and interactions", "matches":[{"local_id":"ID","decision":"same|new|uncertain","match_id":null,"reason":"specific evidence"}]}.
'''
        def validate(result):
            ids = [m.local_id for m in result.matches]
            if set(ids)!=set(current) or len(ids)!=len(set(ids)):
                raise ValueError('Return every current ID exactly once.')
            for m in result.matches:
                if (m.decision=='same' and m.match_id not in memory) or (m.decision!='same' and m.match_id is not None):
                    raise ValueError('same requires memory ID; otherwise null.')
            return result
        atomic_json(folder/'input.json', inventory)
        LOG.info('Reverse shot memory %s: %d local / %d previous entities',sid,len(current),len(memory))
        try:
            result = backend.request(prompt+json.dumps(inventory,ensure_ascii=False)+'\nReturn matches ONLY for these current local IDs: '+json.dumps(list(current))+'. Memory IDs may appear only in match_id. Never list memory IDs as local_id.',MemoryReview,
                clip=scene_clip([f for f in frames if f['shot_id']==sid],current,output,fps),
                images=images+mi,validate=validate,image_pixels=491520,max_tokens=2400) 
            status = 'complete'
        except ModelOutputError as exc:
            result = MemoryReview(matches=[MemoryMatch(local_id=e,decision='uncertain',reason=str(exc)) for e in current])
            status = 'uncertain'
        atomic_json(folder/'result.json',dict(status=status,**result.model_dump()))
        for m in result.matches:
            prior = list(memory[m.match_id]['members']) if m.decision=='same' else []
            entry = dict(current[m.local_id], scene_summary=result.scene_summary, match_reason=m.reason)
            target, blocked = update_memory(memory,m.local_id,entry,m.decision,m.match_id,'R')
            assignments[m.local_id] = target
            if prior and not blocked:
                edges.append(tuple(sorted([m.local_id,prior[-1]])))
            events.append(dict(shot_id=sid,**m.model_dump(),assigned_id=target,co_visible_blocked=blocked))
        history.append(dict(shot_id=sid, scene_summary=result.scene_summary,
            entities=[dict(local_id=m.local_id, memory_id=assignments[m.local_id], reason=m.reason) for m in result.matches]))
        atomic_json(folder/'memory_after.json',snapshot(memory))
    atomic_json(root/'reverse_history.json',history)
    atomic_json(root/'reverse.json',dict(assignments=assignments,memory=snapshot(memory),events=events))
    return assignments, edges


def partition_pairs(mapping):
    return {tuple(sorted([a,b])) for a,b in combinations(mapping,2) if mapping[a]==mapping[b]}


def reconcile(groups, forward, reverse, edges, frames, output, backend, fps, root, history):
    folder = root/'final'; folder.mkdir(parents=True,exist_ok=True)
    parents = {k:k for k in groups}; negative = set(); decisions = []; joins = []
    def find(k):
        while parents[k]!=k:
            k=parents[k]
        return k
    def members(k):
        return {i for i in groups if find(i)==find(k)}
    def blocked(a,b):
        aa,bb = members(a),members(b)
        fa = {o['frame_key'] for k in aa for o in groups[k]['observations']}
        fb = {o['frame_key'] for k in bb for o in groups[k]['observations']}
        return bool(fa & fb) or any(tuple(sorted([x,y])) in negative for x in aa for y in bb)
    def inspect(a,b,pools,tag):
        dest = folder/tag; dest.mkdir(exist_ok=True)
        LOG.info('Contextual pair review %s / %s',a,b)
        prior = dict(forward_same=any(forward[x]==forward[y] for x in members(a) for y in members(b)),
                     reverse_same=any(reverse[x]==reverse[y] for x in members(a) for y in members(b)))
        decision = adjudicate_pair(a,b,pools,frames,history,output,backend,dest,fps,prior)
        record = dict(ids=[a,b],decision=decision.decision,
            video_review=decision.model_dump(), crop_review=None,
            review_policy='joint context and appearance; no crop veto', prior=prior,
            evidence=str(dest.relative_to(output)),phase=tag)
        decisions.append(record)
        atomic_json(folder/'decisions.json',decisions)
        return record
    def merge(a,b,record):
        a,b=find(a),find(b)
        if a==b or blocked(a,b): return False
        target,old = sorted([a,b]); parents[old]=target
        joins.append(dict(target=target,retired=old,method='Qwen3-VL bidirectional final review',
            reason=record['video_review']['visual_evidence'],pair_verification=record['evidence']))
        return True

    fp,rp=partition_pairs(forward),partition_pairs(reverse)
    candidates = sorted(set(edges),key=lambda p:(p not in fp&rp,p))
    checked = []
    # Evaluate evidence before unioning: negative decisions constrain transitive merges.
    for a,b in candidates:
        if visible_frames(groups[a]) & visible_frames(groups[b]):
            negative.add((a,b)); continue
        record=inspect(a,b,groups,'direction_pairs'); checked.append(record)
        if record['decision']=='different': negative.add((a,b))
    for record in checked:
        a,b=record['ids']
        record['merged']=record['decision']=='same' and merge(a,b,record)

    # Global final proposal scans EVERY remaining identity, not only nearby shots.
    pools={}
    for gid,e in groups.items():
        key=find(gid)
        pools.setdefault(key,dict(label=e['label'],kind=e['kind'],appearance=e['appearance'],observations=[],history=[]))['observations'].extend(e['observations'])
        pools[key]['history'].append(dict(shot_id=e['shot_id'], scene_summary=e.get('scene_summary',''),
                                        label=e['label'], appearance=e['appearance']))
    keys=list(pools); pages=[keys[i:i+6] for i in range(0,len(keys),6)]
    proposed=set(); proposal_records=[]
    for i in range(len(pages)):
        for j in range(i,len(pages)):
            ids=list(dict.fromkeys(pages[i]+pages[j]))
            if len(ids)<2:continue
            dest=folder/f'global_{i:02d}_{j:02d}';dest.mkdir(exist_ok=True)
            entries={k:pools[k] for k in ids}
            forbidden=[list(p) for p in combinations(ids,2) if blocked(*p)]
            if len(forbidden)==len(ids)*(len(ids)-1)//2:continue
            LOG.info('Final global gallery pages %d/%d and %d/%d: %d IDs',i+1,len(pages),j+1,len(pages),len(ids))
            images,meta=gallery(entries,output,dest,'references')
            prompt=POLICY+'''Final global audit: find plausible duplicate physical individuals among these IDs using their multiple actual crop views, including long-range human reappearances. Different lighting, pose or clothes need not mean a different person. Names/species/roles alone are insufficient. A carrier and its passenger are distinct. Propose only pairs for a separate full-scene verifier; never join forbidden pairs. Return ONLY JSON {"groups":[{"ids":["ID_A","ID_B"],"reason":"visible evidence"}]}; [] if none.\n'''+json.dumps(dict(version=VERSION,entities=meta,forbidden_pairs=forbidden,shot_history=history),ensure_ascii=False)
            def validate(result):
                for g in result.groups:
                    if not set(g.ids)<=set(ids) or len(g.ids)!=len(set(g.ids)):
                        raise ValueError('Use distinct listed IDs.')
                return result
            try:
                result=backend.request(prompt,MergeReview,images=images,validate=validate,max_tokens=1600,image_pixels=491520)
                proposal_records.append(dict(page=[i,j],**result.model_dump()))
                atomic_json(dest/'result.json',result.model_dump())
                for g in result.groups:proposed.update(tuple(sorted(p)) for p in combinations(g.ids,2))
            except ModelOutputError as exc:
                proposal_records.append(dict(page=[i,j],error=str(exc)))
    global_checked=[]
    LOG.info('Final global gallery proposed %d candidate pairs',len(proposed))
    for a,b in sorted(proposed):
        if blocked(a,b):continue
        record=inspect(a,b,pools,'global_pairs');global_checked.append(record)
        if record['decision']=='different':
            negative.update(tuple(sorted([x,y])) for x in members(a) for y in members(b))
    for record in global_checked:
        a,b=record['ids'];record['merged']=record['decision']=='same' and merge(a,b,record)
    atomic_json(folder/'decisions.json',decisions)
    atomic_json(folder/'global_proposals.json',proposal_records)
    agreement=dict(forward_entities=len(set(forward.values())),reverse_entities=len(set(reverse.values())),
        same_in_both=[list(p) for p in sorted(fp&rp)],forward_only=[list(p) for p in sorted(fp-rp)],
        reverse_only=[list(p) for p in sorted(rp-fp)],
        pair_jaccard=len(fp&rp)/len(fp|rp) if fp|rp else 1.,
        note='Agreement measures consistency, not accuracy against ground truth. Reverse matching starts from shot-local groups; forward cross-shot IDs are withheld.',
        final_entities=len({find(k) for k in groups}),uncertain_pair_reviews=sum(d['decision']=='uncertain' for d in decisions))
    atomic_json(root/'direction_comparison.json',agreement)
    return {k:find(k) for k in groups},joins,agreement


def run_sequential(raw_all,frames,timeline,output,backend,settings,proposals):
    root=output/'sequential';root.mkdir(exist_ok=True)
    fps=float(Fraction(timeline['fps']));memory={};groups={};forward={};edges=[];events=[];history=[]
    for index,shot in enumerate(timeline['shots'],1):
        sid=shot['id'];folder=root/'forward'/sid;folder.mkdir(parents=True,exist_ok=True)
        sf=[f for f in frames if f['shot_id']==sid]
        current=local_seeds(deepcopy([o for o in raw_all if o['shot_id']==sid]),settings,index)
        LOG.info('Forward shot memory %s: %d local / %d previous entities',sid,len(current),len(memory))
        result=review_shot(sid,sf,current,memory,proposals,output,backend,fps,folder,history)
        materialize_additions(result,current,sf,output)
        for i,entity in enumerate(result.entities,1):
            # Keep user-facing labels brief and Korean without guessing identity.
            if len(entity.label)>24 or len(entity.label.split())>5 or not any('\uac00'<=c<='\ud7a3' for c in entity.label) or any('\u4e00'<=c<='\u9fff' for c in entity.label):
                try:
                    short=backend.request('Rewrite this visual entity label as 2-5 Korean words, at most 24 characters. Do not add a proper name or new facts. ONLY JSON {"label":"한국어 짧은 이름"}. '+entity.label,
                        ShortLabel,max_tokens=100)
                    entity.label=short.label if any('\uac00'<=c<='\ud7a3' for c in short.label) and not any('\u4e00'<=c<='\u9fff' for c in short.label) else {'person':'인물','animal':'동물','object':'물체'}[entity.kind]
                except ModelOutputError:
                    entity.label={'person':'인물','animal':'동물','object':'물체'}[entity.kind]
            gid=f'S{index:03d}_{i:03d}'
            entry=dict(label=entity.label,appearance=entity.appearance,kind=entity.kind,shot_id=sid,
                scene_summary=result.scene_summary,match_reason=entity.reason,
                observations=[o for local_id in entity.ids for o in current[local_id]['observations']])
            for o in entry['observations']:
                o.update(label=entity.label,appearance=entity.appearance,entity_kind=entity.kind,shot_local_id=gid)
            groups[gid]=entry
            prior=list(memory[entity.match_id]['members']) if entity.decision=='same' else []
            target,blocked=update_memory(memory,gid,entry,entity.decision,entity.match_id,'F')
            forward[gid]=target
            if prior and not blocked:edges.append(tuple(sorted([gid,prior[-1]])))
            events.append(dict(shot_id=sid,local_id=gid,**entity.model_dump(),assigned_id=target,co_visible_blocked=blocked))
        history.append(dict(shot_id=sid, scene_summary=result.scene_summary,
            entities=[dict(local_id=e['local_id'],memory_id=e['assigned_id'],label=e['label'],reason=e['reason'])
                      for e in events if e['shot_id']==sid]))
        atomic_json(root/'shot_history.json',history)
        atomic_json(folder/'memory_after.json',snapshot(memory))
    atomic_json(root/'local_entities.json',groups)
    atomic_json(root/'forward.json',dict(assignments=forward,memory=snapshot(memory),events=events))
    reverse,reverse_edges=reverse_pass(groups,timeline['shots'],frames,output,backend,fps,root)
    final,joins,comparison=reconcile(groups,forward,reverse,edges+reverse_edges,frames,output,backend,fps,root,history)
    registry={};visual={'assignments':{}};observed=[];export_ids={}
    for gid in groups:
        export_ids.setdefault(final[gid],f'E{len(export_ids)+1:04d}')
    for gid,entry in groups.items():
        eid=export_ids[final[gid]]
        registry.setdefault(eid,dict(entity_id=eid,identity_status='reviewed',first_seen_sec=float('inf'),last_seen_sec=0))
        for o in entry['observations']:
            registry[eid]['first_seen_sec']=min(registry[eid]['first_seen_sec'],o['original_time_sec'])
            registry[eid]['last_seen_sec']=max(registry[eid]['last_seen_sec'],o['original_time_sec'])
            visual['assignments'][o['detection_id']]=dict(entity_id=eid,shot_local_id=gid,
                forward_id=forward[gid],reverse_id=reverse[gid],identity_merge_method='Qwen3-VL bidirectional final review')
            observed.append(o)
    atomic_json(root/'final_mapping.json',{k:dict(entity_id=export_ids[final[k]],forward_id=forward[k],reverse_id=reverse[k]) for k in groups})
    atomic_json(output/'identity_before_review.json',dict(registry={},assignments={o['detection_id']:dict(entity_id=k) for k,e in groups.items() for o in e['observations']}))
    atomic_json(root/'summary.json',dict(version=VERSION,shots=len(timeline['shots']),shot_local_entities=len(groups),
        final_entities=len(registry),observations=len(observed),comparison=comparison))
    return registry,visual,observed,joins
