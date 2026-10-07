"""Evidence-backed identity reconciliation using ordered, annotated scene video."""
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import json
import logging
from typing import Literal
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, Field, model_validator
from low_freq_event.identity import canonical
from low_freq_event.backend import ModelOutputError
from low_freq_event.storage import atomic_json
from low_freq_event.video import Clip, Frame

TEMPORAL_VERSION = 7
PROPOSAL_PROMPT_VERSION = 5
LOG = logging.getLogger('entities')


def split_shot_seeds(registry,visual,raw_all,output):
    """Discard unverified cross-cut identity links before temporal reasoning.

    An appearance ID can already contain different creatures in different shots.
    Merging that ID later cannot repair it, so seed identities are shot-local.
    """
    pools=defaultdict(list)
    for obs in raw_all:
        old=canonical(registry,visual['assignments'][obs['detection_id']]['entity_id'])
        pools[(old,obs['shot_id'])].append(obs)
    new_registry={};new_visual={'assignments':{}};mapping=[]
    counters={prefix:max((int(k.rsplit('_',1)[1]) for k in registry if k.startswith(prefix+'_')),default=0)
        for prefix in ('person','object')}
    used=set()
    for (old,shot),observed in pools.items():
        if old not in used:
            eid=old;used.add(old)
        else:
            prefix=old.split('_')[0];counters[prefix]+=1;eid=f'{prefix}_{counters[prefix]:04d}'
        entry=deepcopy(registry[old]);entry.pop('alias_of',None)
        entry.update(entity_id=eid,first_seen_sec=min(o['original_time_sec'] for o in observed),
            last_seen_sec=max(o['original_time_sec'] for o in observed),possible_matches=[],
            gallery=[o['crop'] for o in observed[:8]],seed_shot=shot,pre_split_entity_id=old)
        new_registry[eid]=entry
        for obs in observed:
            proof=deepcopy(visual['assignments'][obs['detection_id']])
            proof.update(entity_id=eid,pre_split_entity_id=old,possible_matches=[],
                identity_scope='within-shot seed; cross-shot links require temporal review')
            new_visual['assignments'][obs['detection_id']]=proof
        mapping.append(dict(previous_id=old,shot_id=shot,seed_id=eid,observations=len(observed)))
    atomic_json(output/'shot_identity_seeds.json',dict(
        reason='Existing appearance identities may mix different individuals across cuts; split before temporal matching.',
        original_ids=len({old for old,shot in pools}),shot_local_ids=len(new_registry),mapping=mapping))
    LOG.info('Shot-local identity seeds: %d -> %d',len({old for old,shot in pools}),len(new_registry))
    return new_registry,new_visual


class TemporalGroup(BaseModel):
    ids: list[str] = Field(min_length=2)
    reason: str = ''


class TemporalReview(BaseModel):
    scene_summary: str = ''
    groups: list[TemporalGroup]

    @model_validator(mode='before')
    @classmethod
    def omit_singletons(cls, value):
        # A one-ID group proposes no change. Ignore it without discarding other
        # valid proposals; all actual merges still pass the strict validator.
        if isinstance(value,dict) and isinstance(value.get('groups'),list):
            value=dict(value)
            value['groups']=[{'ids':g} if isinstance(g,list) else g for g in value['groups']]
            value['groups']=[g for g in value['groups'] if not (
                isinstance(g,dict) and isinstance(g.get('ids'),list) and len(g['ids'])==1)]
        return value

class PairDecision(BaseModel):
    decision: Literal['same', 'different', 'uncertain']
    kind: Literal['person', 'animal', 'object']
    label: str
    visual_evidence: str
    temporal_evidence: str

    @model_validator(mode='after')
    def readable_label(self):
        if not any('\uac00'<=c<='\ud7a3' for c in self.label):
            self.label={'person':'인물','animal':'동물·생물','object':'물체'}[self.kind]
        return self


class AppearanceDecision(BaseModel):
    decision: Literal['same','different','ambiguous']
    reason: str


def verify_primary_subject(a,b,folder,backend):
    image=Image.open(folder/f'{a}__{b}.jpg').convert('RGB')
    prompt='''Compare ONLY the visible primary subject in column A versus column B. Each column has up to two views of its target, NOT two separate candidates.
Judge individual identity from distinctive face/head/body/wing anatomy. Ignore story, temporal adjacency, same species and background similarity. Pose, viewpoint, lighting and apparent image size may change.
If a crop contains two overlapping creatures (for example an adult holding a smaller creature), do NOT match the adult in that crop to the primary smaller creature in the other crop, or vice versa. If the primary subject cannot be isolated, or two distinct heads/subjects occupy the target crop, answer ambiguous. An unambiguous primary head/body may have another creature's partial limb in the background.
Return JSON only: {"decision":"same|different|ambiguous","reason":"specific visible anatomy agreement/difference or multiple-subject ambiguity"}. Use same only when the PRIMARY individuals match; otherwise use different or ambiguous.'''
    try:
        result=backend.request(prompt,AppearanceDecision,images=[('A left; B right; inspect primary subjects',image)],
            image_pixels=786432,max_tokens=600)
    except ModelOutputError as exc:
        result=AppearanceDecision(decision='ambiguous',reason=str(exc))
    atomic_json(folder/f'{a}__{b}_appearance.json',result.model_dump())
    return result


def verify_pair(a,b,local,selected_frames,output,backend,dest,font,fps):
    """Independent close inspection; previous labels/proposal reasons are withheld."""
    selected=set(); crops=Image.new('RGB',(800,390),'#202020'); gd=ImageDraw.Draw(crops)
    for index,eid in enumerate([a,b]):
        pool=sorted(local[eid],key=lambda o:(o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]),reverse=True)
        refs=[pool[0]]
        others=[o for o in pool if o['shot_id']!=refs[0]['shot_id']]
        if others: refs.append(others[0])
        else:
            others=[o for o in pool if abs(o['full_time_sec']-refs[0]['full_time_sec'])>.15]
            if others: refs.append(others[0])
        gd.text((index*400+6,5),f'{chr(65+index)}: {eid}',font=font,fill='white')
        for j,obs in enumerate(refs):
            selected.add(obs['frame_key'])
            im=ImageOps.contain(Image.open(output/obs['crop']).convert('RGB'),(390,160))
            crops.paste(im,(index*400+(400-im.width)//2,30+j*180+(160-im.height)//2))
            gd.text((index*400+3,193+j*180),f"{obs['original_time_sec']:.3f}s",font=font,fill='white')
    # Include scene middles as context while retaining only a compact video.
    for shot in dict.fromkeys(f['shot_id'] for f in selected_frames):
        fs=[f for f in selected_frames if f['shot_id']==shot]
        selected.add(fs[len(fs)//2]['frame_key'])
    video=[]; inventory=[]
    for f in selected_frames:
        if f['frame_key'] not in selected: continue
        im=Image.open(output/f['image']).convert('RGB').resize((960,540)); draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,960,24),fill='black')
        draw.text((5,3),f"{f['shot_id']} | {f['original_time_sec']:.3f}s",font=font,fill='white')
        labels=[]
        for index,eid in enumerate([a,b]):
            for obs in local[eid]:
                if obs['frame_key']!=f['frame_key']:continue
                xy=[obs['bbox'][0]*.96,obs['bbox'][1]*.54,obs['bbox'][2]*.96,obs['bbox'][3]*.54]
                c='#00ff00' if index==0 else '#ffff00'
                draw.rectangle(xy,outline=c,width=3);draw.text((xy[0],max(25,xy[1]-20)),chr(65+index),font=font,fill=c)
                labels.append(chr(65+index))
        video.append(Frame(f['full_time_sec'],im))
        inventory.append(dict(frame_key=f['frame_key'],time=f['original_time_sec'],visible_labels=labels))
    clip=Clip(video[0].time_sec,video[-1].time_sec+1/fps,fps,video)
    tag=f'{a}__{b}';crops.save(dest/f'{tag}.jpg',quality=95)
    prompt='''Independently decide whether target A and target B are the SAME physical individual.
The crop sheet has A on the left and B on the right, up to two views each. The chronological video supplies scene context with A/B boxes. Compare the actual boxed subjects, not the background or another overlapping creature.
Use distinctive head/face, body, wings, horns, clothing AND continuity of action across cuts. A pursuer/carrier and its prey/passenger are DIFFERENT individuals even when their crops overlap. Flying and being carried upside-down can be two views of one small creature if the visible anatomy agrees.
Do not infer identity from narrative roles or similarity of species alone. If the image is too dark, ambiguous, or anatomically inconsistent, answer uncertain or different. No previous detector labels are ground truth.
Return ONLY JSON: {"decision":"same|different|uncertain","kind":"person|animal|object","label":"concise visual Korean label","visual_evidence":"specific agreement or difference in visible anatomy/clothing","temporal_evidence":"actual temporal connection or its absence"}.
'''+json.dumps(inventory,ensure_ascii=False)
    atomic_json(dest/f'{tag}_input.json',dict(ids=[a,b],prompt=prompt,frames=inventory))
    try:
        result=backend.request(prompt,PairDecision,clip=clip,images=[('A and B enlarged crops',crops)],
            max_tokens=700,image_pixels=524288)
    except ModelOutputError as exc:
        atomic_json(dest/f'{tag}_result.json',dict(decision='uncertain',error=str(exc)))
        return None
    atomic_json(dest/f'{tag}_result.json',result.model_dump())
    return result


def review_temporal_identities(registry, visual, raw_all, frames, timeline, output, backend):
    joins = []; reviews = []
    folder = output / 'temporal_review'; folder.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    fps = float(Fraction(timeline['fps']))
    by_frame = defaultdict(list)
    for obs in raw_all: by_frame[obs['frame_key']].append(obs)
    shots = [s['id'] for s in timeline['shots']]

    def assigned(obs):
        return canonical(registry, visual['assignments'][obs['detection_id']]['entity_id'])

    # Overlap two shots: a merge in one window becomes an anchor in the next.
    for offset in range(max(1, len(shots)-2)):
        window = shots[offset:offset+3]
        local_frames = [f for f in frames if f['shot_id'] in window]
        history = defaultdict(list)
        for obs in raw_all: history[assigned(obs)].append(obs)
        local = defaultdict(list)
        for obs in raw_all:
            if obs['shot_id'] in window: local[assigned(obs)].append(obs)
        ids = sorted(local)
        if len(ids) < 2: continue
        frame_sets = {eid: {o['frame_key'] for o in history[eid]} for eid in ids}
        forbidden = [[a,b] for i,a in enumerate(ids) for b in ids[i+1:] if frame_sets[a] & frame_sets[b]]
        if len(forbidden) == len(ids)*(len(ids)-1)//2: continue

        # Keep one clear observation of EVERY candidate, plus scene context.
        selected = set(); references={}
        for eid, observations in local.items():
            best = max(observations, key=lambda o: (o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]))
            references[eid]=best
            selected.add(best['frame_key'])
        for shot in window:
            fs = [f for f in local_frames if f['shot_id']==shot]
            for i in {0, len(fs)//2, len(fs)-1}: selected.add(fs[i]['frame_key'])
        selected_frames = [f for f in local_frames if f['frame_key'] in selected]
        tag = f'{window[0]}_to_{window[-1]}'
        dest = folder/tag; dest.mkdir(exist_ok=True)
        video_frames = []; frame_inventory = []
        sheet = Image.new('RGB', (960, 222*((len(selected_frames)+2)//3)), '#202020')
        for index, f in enumerate(selected_frames):
            im = Image.open(output/f['image']).convert('RGB').resize((960,540))
            draw = ImageDraw.Draw(im)
            draw.rectangle((0,0,960,25), fill='black')
            draw.text((4,3), f"{f['frame_key']} | original {f['original_time_sec']:.3f}s", font=font, fill='white')
            observations = by_frame[f['frame_key']]
            for obs in observations:
                eid = assigned(obs); box = obs['bbox']
                xy = [box[0]*.96,box[1]*.54,box[2]*.96,box[3]*.54]
                draw.rectangle(xy, outline='#ffff00', width=2)
                x=min(xy[0],815); y=max(26,xy[1]-20)
                draw.rectangle((x,y,x+145,y+20), fill='black')
                draw.text((x+2,y),eid,font=font,fill='#ffff00')
            im.save(dest/f'{index:03d}_{f["frame_key"]}.jpg',quality=92)
            thumb=ImageOps.contain(im,(320,180)); x=index%3*320; y=index//3*222
            sheet.paste(thumb,(x,y)); sd=ImageDraw.Draw(sheet)
            sd.text((x+2,y+183),f['frame_key'],font=font,fill='white')
            sd.text((x+2,y+203),f"{f['original_time_sec']:.3f}s",font=font,fill='white')
            video_frames.append(Frame(f['full_time_sec'],im))
            frame_inventory.append(dict(frame_key=f['frame_key'],shot=f['shot_id'],
                original_time_sec=f['original_time_sec'],ids=[assigned(o) for o in observations]))
        sheet.save(dest/'storyboard.jpg',quality=92)
        gallery=Image.new('RGB',(768,244*((len(ids)+2)//3)),'#202020')
        gd=ImageDraw.Draw(gallery)
        for index,eid in enumerate(ids):
            ref=references[eid]; x=index%3*256; y=index//3*244
            crop=ImageOps.contain(Image.open(output/ref['crop']).convert('RGB'),(248,190))
            gallery.paste(crop,(x+(256-crop.width)//2,y+26+(190-crop.height)//2))
            gd.text((x+4,y+3),eid,font=font,fill='white')
            gd.text((x+4,y+219),ref['frame_key'],font=font,fill='white')
        gallery.save(dest/'reference_crops.jpg',quality=95)
        clip = Clip(video_frames[0].time_sec,video_frames[-1].time_sec+1/fps,fps,video_frames)
        inventory = [dict(id=eid,reference_frame=references[eid]['frame_key']) for eid in ids]
        prompt = '''Find candidate identity links across this chronological sequence of 3 shots.
Each gallery cell is a crop of the ID marked in the video. Different IDs may show the SAME actual individual at different times. Compare distinctive anatomy/clothing and continuity of action across cuts. Include within-shot fragments as well as cross-shot links.
ID prefixes person/object are unreliable detector classifications, NOT evidence of species. A small flying creature can later be carried upside-down by a larger creature. The carrier and the carried creature are distinct. Use the actual boxed subject, not an overlapping body/background. Do not assume a human hand is an independently identifiable person.
Propose pairs of IDs that may be the same individual for a separate detailed verifier. Prefer distinctive visual evidence plus temporal continuity; do not join solely by species or story role. The forbidden pairs have simultaneous separate detections and cannot be the same.
Return concise JSON ONLY: {"scene_summary":"brief observed sequence", "groups":[{"ids":["ID_A","ID_B"],"reason":"specific visual and temporal connection"}]}. Use the exact listed IDs. IDs may participate in several pairs. No singleton groups. Use groups=[] if there are no plausible links. Do not reproduce the input metadata.
'''+json.dumps(dict(version=PROPOSAL_PROMPT_VERSION,entities=inventory,frames=frame_inventory,forbidden_pairs=forbidden),ensure_ascii=False)
        def validate(result):
            # Invalid proposals are dropped individually; only the independent
            # pair verifier can authorize a merge. Raw answers remain in traces.
            result.groups=[g for g in result.groups if len(set(g.ids))==len(g.ids) and set(g.ids)<=set(ids)]
            return result
        atomic_json(dest/'input.json',dict(prompt=prompt,frames=frame_inventory,entities=inventory,forbidden_pairs=forbidden))
        LOG.info('Temporal review %s: %d IDs, %d frames',tag,len(ids),len(video_frames))
        try:
            result=backend.request(prompt,TemporalReview,clip=clip,images=[('ID reference crops',gallery)],
                image_pixels=1048576,validate=validate,max_tokens=1400)
        except ModelOutputError as exc:
            # Fail closed for this window; never invent a merge to complete export.
            failure=dict(window=tag,status='rejected',error=str(exc),groups=[])
            atomic_json(dest/'result.json',failure); reviews.append(failure)
            atomic_json(folder/'review.json',dict(version=TEMPORAL_VERSION,joins=joins,reviews=reviews))
            LOG.warning('Temporal review rejected; IDs retained: %s',tag)
            continue
        atomic_json(dest/'result.json',result.model_dump())
        reviews.append(dict(window=tag,**result.model_dump()))
        pairs={tuple(sorted(pair)) for group in result.groups for pair in combinations(group.ids,2)}
        # Nearby observations first; confirmed links shrink later candidate sets.
        pairs=sorted(pairs,key=lambda pair:(min(abs(x['full_time_sec']-y['full_time_sec'])
            for x in local[pair[0]] for y in local[pair[1]]),pair))
        for anchor,candidate in pairs:
            a=canonical(registry,anchor); b=canonical(registry,candidate)
            if a==b: continue
            current_a={o['frame_key'] for o in raw_all if assigned(o)==a}
            current_b={o['frame_key'] for o in raw_all if assigned(o)==b}
            if current_a & current_b: continue
            LOG.info('Temporal pair verification %s / %s',anchor,candidate)
            decision=verify_pair(anchor,candidate,local,local_frames,output,backend,dest,font,fps)
            if decision is None or decision.decision!='same': continue
            pair=[a,b]
            preferred=[eid for eid in pair if eid.startswith('person_' if decision.kind=='person' else 'object_')]
            target=min(preferred or pair); old=b if target==a else a
            registry[old]['alias_of']=target
            registry[target]['first_seen_sec']=min(registry[target]['first_seen_sec'],registry[old]['first_seen_sec'])
            registry[target]['last_seen_sec']=max(registry[target]['last_seen_sec'],registry[old]['last_seen_sec'])
            joins.append(dict(target=target,retired=old,method='Qwen3-VL temporal scene review',review=tag,
                reason=decision.visual_evidence,temporal_evidence=decision.temporal_evidence,
                evidence_frames=[f['frame_key'] for f in json.loads((dest/f'{anchor}__{candidate}_input.json').read_text())['frames'] if f['visible_labels']],pair_verification=f'temporal_review/{tag}/{anchor}__{candidate}_result.json'))
            registry[target]['identity_status']='matched'
            registry[target]['temporal_semantics']=dict(kind=decision.kind,label=decision.label)
        atomic_json(folder/'review.json',dict(version=TEMPORAL_VERSION,joins=joins,reviews=reviews))
    for obs in raw_all:
        proof=visual['assignments'][obs['detection_id']]
        eid=assigned(obs)
        if proof['entity_id']!=eid:
            proof['identity_merge_method']='Qwen3-VL temporal scene review'
            proof['entity_id']=eid
        semantics=registry[eid].get('temporal_semantics')
        if semantics:
            obs['pre_temporal_label']=obs['label']; obs['pre_temporal_entity_kind']=obs['entity_kind']
            obs['label']=semantics['label']; obs['entity_kind']=semantics['kind']
    return joins


def review_retrieved_candidates(registry,visual,raw_all,frames,timeline,output,backend):
    """Recall supplement: DINO proposes nearby cross-shot pairs; Qwen decides."""
    folder=output/'temporal_retrieval';folder.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    fps=float(Fraction(timeline['fps']));shots=[s['id'] for s in timeline['shots']]
    shot_index={s:i for i,s in enumerate(shots)}
    def assigned(o):return canonical(registry,visual['assignments'][o['detection_id']]['entity_id'])
    history=defaultdict(list)
    for o in raw_all:history[assigned(o)].append(o)
    objects=[o for o in raw_all if o.get('object_feature')]
    candidates={}
    for i,x in enumerate(objects):
        a=assigned(x);sx=shot_index[x['shot_id']]
        for y in objects[i+1:]:
            b=assigned(y);sy=shot_index[y['shot_id']]
            if a==b or not 1<=abs(sx-sy)<=2:continue
            if {o['frame_key'] for o in history[a]} & {o['frame_key'] for o in history[b]}:continue
            score=float(np.dot(x['object_feature'],y['object_feature']))
            if score<.25:continue
            # Keep the strongest view comparison per candidate and shot pair.
            key=(a,b,sx,sy)
            if score>candidates.get(key,-1):candidates[key]=score
    nearest=defaultdict(list)
    for (a,b,sx,sy),score in candidates.items():
        nearest[(a,sy)].append((score,b,sx,sy,a))
        nearest[(b,sx)].append((score,a,sy,sx,b))
    selected={}
    for choices in nearest.values():
        seen=set()
        for score,b,sx,sy,a in sorted(choices,reverse=True):
            if b in seen:continue
            seen.add(b);pair=tuple(sorted([a,b]))
            if pair not in selected or score>selected[pair]['similarity']:
                selected[pair]=dict(ids=list(pair),similarity=score,shot_indices=sorted([sx,sy]))
            if len(seen)==2:break
    records=sorted(selected.values(),key=lambda r:(-r['similarity'],r['ids']))
    atomic_json(folder/'candidates.json',dict(feature='DINOv2 object_feature',top_k_per_other_shot=2,
        max_shot_distance=2,min_candidate_similarity=.25,candidates=records))
    joins=[];decisions=[]
    for index,record in enumerate(records):
        a,b=record['ids'];ca=canonical(registry,a);cb=canonical(registry,b)
        if ca==cb:continue
        fa={o['frame_key'] for o in raw_all if assigned(o)==ca}
        fb={o['frame_key'] for o in raw_all if assigned(o)==cb}
        if fa & fb:continue
        start=min(record['shot_indices'][0],max(0,len(shots)-3));window=shots[start:start+3]
        local={eid:[o for o in history[eid] if o['shot_id'] in window] for eid in [a,b]}
        local_frames=[f for f in frames if f['shot_id'] in window]
        LOG.info('Retrieved temporal pair %d/%d: %s / %s (DINO %.3f)',index+1,len(records),a,b,record['similarity'])
        decision=verify_pair(a,b,local,local_frames,output,backend,folder,font,fps)
        appearance=None
        if decision is not None and decision.decision=='same' and record['similarity']<.7:
            appearance=verify_primary_subject(a,b,folder,backend)
        decisions.append(dict(**record,decision=decision.model_dump() if decision else None,
            appearance_check=appearance.model_dump() if appearance else None))
        if decision is not None and decision.decision=='same' and (appearance is None or appearance.decision=='same'):
            preferred=[eid for eid in [ca,cb] if eid.startswith('person_' if decision.kind=='person' else 'object_')]
            target=min(preferred or [ca,cb]);old=cb if target==ca else ca
            registry[old]['alias_of']=target
            registry[target]['first_seen_sec']=min(registry[target]['first_seen_sec'],registry[old]['first_seen_sec'])
            registry[target]['last_seen_sec']=max(registry[target]['last_seen_sec'],registry[old]['last_seen_sec'])
            registry[target]['identity_status']='matched'
            registry[target]['temporal_semantics']=dict(kind=decision.kind,label=decision.label)
            evidence=json.loads((folder/f'{a}__{b}_input.json').read_text())['frames']
            joins.append(dict(target=target,retired=old,method='Qwen3-VL temporal retrieved-pair review',
                reason=decision.visual_evidence,temporal_evidence=decision.temporal_evidence,
                candidate_similarity=record['similarity'],
                appearance_check=appearance.model_dump() if appearance else None,
                evidence_frames=[f['frame_key'] for f in evidence if f['visible_labels']],
                pair_verification=f'temporal_retrieval/{a}__{b}_result.json'))
        atomic_json(folder/'review.json',dict(joins=joins,decisions=decisions))
    for obs in raw_all:
        proof=visual['assignments'][obs['detection_id']];eid=assigned(obs)
        if eid!=proof['entity_id']:
            proof['entity_id']=eid;proof['identity_merge_method']='Qwen3-VL temporal retrieved-pair review'
        semantics=registry[eid].get('temporal_semantics')
        if semantics:
            obs.setdefault('pre_temporal_label',obs['label']);obs.setdefault('pre_temporal_entity_kind',obs['entity_kind'])
            obs['label']=semantics['label'];obs['entity_kind']=semantics['kind']
    atomic_json(folder/'review.json',dict(joins=joins,decisions=decisions))
    return joins
