"""Experimental bounded batch identity review; no saved catalog or labels are read."""
from copy import deepcopy
from collections import defaultdict
import json
from pathlib import Path
from pydantic import BaseModel, Field, RootModel
from low_freq_event.storage import atomic_json
from catalog_repair import FrameAssignments, FrameDecision, Body, board, validate_frame

VERSION=17
PHASES=('context_reassign:', 'batch_global:', 'batch_frames:', 'batch_profiles:', 'local_temporal_profile:')

class Partition(BaseModel):
    groups:list[list[str]]

class BatchFrames(RootModel[dict[str,FrameAssignments]]):
    @property
    def frames(self):return self.root

class LimitedBackend:
    """Includes real generations and retries; resumed calls cannot reset the cap."""
    def __init__(self,backend,limit):
        self.backend=backend;self.limit=limit;self.reserve=0
    def __getattr__(self,key):return getattr(self.backend,key)
    @property
    def phase(self):return self.backend.phase
    @phase.setter
    def phase(self,value):self.backend.phase=value
    def used(self):return sum(e.get('phase','').startswith(PHASES) for e in self.ledger['events'])
    def request(self,*args,**kwargs):
        old=self.backend.reserve
        self.backend.reserve=max(old,self.reserve,self.remaining-max(0,self.limit-self.used()))
        try:return self.backend.request(*args,**kwargs)
        finally:self.backend.reserve=old


def global_catalog(bundle,local,output,backend,review_settings=None):
    from compact_identity import visual_board
    from story_semantics import local_lookup,cannot_links,global_input_fingerprint,visual_candidates
    import gc,torch
    from story_pipeline import default_global_model,THINKING_REVISION
    entries=local_lookup(local);keys=list(entries);order={k:i for i,k in enumerate(keys)}
    groups={f'I{i+1:03d}':dict(local_ids=[lid],kind=entries[lid]['kind']) for i,lid in enumerate(keys)}
    forbidden=cannot_links(bundle,local);retrieval=visual_candidates(bundle,local)
    dest=Path(output)/'batch_identity';dest.mkdir(parents=True,exist_ok=True)
    original=backend.backend.settings
    review_model=review_settings or dict(model_id=default_global_model(),model_revision=THINKING_REVISION)
    reviews=[];sequence=0
    def review(selected,strong=False):
        nonlocal sequence
        if len(selected)<2:return
        if len(selected)>24:raise ValueError('Catalog page exceeds 24 candidates; no silent truncation')
        sequence+=1;images=[];sources={};aliases={lid:k for k in selected for lid in groups[k]['local_ids']}
        for offset,key in enumerate(selected):
            page=[key]
            shown_groups=deepcopy(groups)
            tracks={t['track_id']:t for shot in bundle['shots'] for t in shot['tracks']}
            for key in page:
                lids=groups[key]['local_ids']
                if len(lids)>3:
                    def area(lid):return max(((o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]) for tid in entries[lid]['track_ids'] for o in tracks[tid]['observations']),default=0)
                    ordered=sorted(lids,key=order.get)
                    shown_groups[key]['local_ids']=list(dict.fromkeys([ordered[0],ordered[-1],max(lids,key=area)]))
            ims,shown,src=visual_board(bundle,entries,shown_groups,page,output)
            for label,im in ims:
                im=im.crop((0,0,300,360))
                im.save(dest/f'catalog_{sequence:02d}_{offset:02d}.jpg',quality=94)
                images.append((f'Candidate ID {key}. This image belongs ONLY to {key}; its three crops are views of that target.',im))
            sources.update(src)
        prohibited=sorted({tuple(sorted((aliases[a],aliases[b]))) for a,b in forbidden if a in aliases and b in aliases and aliases[a]!=aliases[b]})
        hints={k:sorted({aliases[x['local_id']] for lid in groups[k]['local_ids'] for x in retrieval.get(lid,[]) if x['local_id'] in aliases and aliases[x['local_id']]!=k}) for k in selected}
        inventory=dict(ids=selected,cannot_link=prohibited,retrieval_candidates=hints,
                       shot_members={k:sorted({entries[lid]['shot_id'] for lid in groups[k]['local_ids']}) for k in selected},
                       local_visual_notes={k:[entries[lid].get('appearance','') for lid in groups[k]['local_ids']] for k in selected})
        def validate(r):
            ids=[k for g in r.groups for k in g]
            if any(not g for g in r.groups) or len(ids)!=len(set(ids)) or not set(ids)<=set(selected):raise ValueError('No duplicate/unknown IDs or empty groups are permitted')
            # Omitted IDs stay unresolved singletons, never disappear or get guessed.
            missing=[k for k in selected if k not in ids]
            if missing:reviews.append(dict(request=sequence,missing_ids=missing,handling='retained as singleton; unreviewed'))
            r.groups.extend([[k] for k in missing])
            for g in r.groups:
                if any(a in g and b in g for a,b in prohibited):raise ValueError('Distinct observed subjects cannot merge')
            return r
        prompt='Partition the supplied candidate groups by the SAME PHYSICAL INDIVIDUAL across shots. Return ONLY ONE JSON object {"groups":[["I001","I002"],["I003"]]}. Every ID exactly once. No prose, code fences, profiles, or invented IDs.\nThe cards may show the same individual in different lighting, front/back/side view, scale, mask, or pose. Use distinctive anatomy, face/hair shape when visible, outfit construction, gear and continuity. A hidden face is missing evidence, NOT evidence of a different person. Compare distant/silhouetted views to detailed candidates; a pose or lighting change alone must not fragment a recurring individual. Do not merge separate individuals merely because they have similar color or species. Different clearly visible faces or incompatible anatomy must stay apart. Separate handheld objects/carried animals are not parts of their owner. A local track can switch subjects; match its dominant recurring body provisionally and the source-frame correction will examine exact observations. The groups are provisional, not prevalidated truth. Retrieval lists only suggest which groups deserve close comparison and do NOT prove identity. Local visual notes are earlier fallible image observations, not identity labels. Inspect the target-specific face/anatomy they describe. A nearly black or tiny crop must not be treated as a match merely because the target cannot be seen. Visible beard/age/face differences are strong contradictions. Respect cannot_link. Do not use movie knowledge or assume a cast size. Inspect images before answering.\nINPUT: '+json.dumps(inventory,ensure_ascii=False)
        backend.phase=f'batch_global:{sequence:03d}'
        if strong and any(getattr(backend.settings,k)!=v for k,v in review_model.items()):
            backend.backend.model=None;backend.backend.processor=None;gc.collect();torch.cuda.empty_cache()
            backend.backend.settings=original.model_copy(update=review_model)
        options=None
        if 'thinking' in str(backend.settings.model_id).lower():
            budget=4096 if len(selected)>12 else (2048 if groups[selected[0]]['kind']=='person' else 1536)
            options=dict(do_sample=True,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,seed=0,thinking_token_budget=budget)
        response=backend.request(prompt,Partition,images=images,max_tokens=1600,image_pixels=1572864,validate=validate,generation_options=options)
        atomic_json(dest/f'global_{sequence:03d}.json',dict(input=inventory,sources=sources,response=response.model_dump()))
        for proposal in response.groups:
            anchor=proposal[0]
            for other in proposal[1:]:groups[anchor]['local_ids']+=groups.pop(other)['local_ids']
    try:
        # Keep human and non-human visual questions separate; never silently drop pages.
        for kind in ('person','animal','object','unknown'):
            selected=[k for k,g in groups.items() if g['kind']==kind]
            size=21
            seen=set()
            for offset in range(0,len(selected),size):
                fresh=[k for k in selected[offset:offset+size] if k in groups]
                fresh_lids={lid for k in fresh for lid in groups[k]['local_ids']}
                prior=[k for k,g in groups.items() if k not in fresh and g['kind']==kind and any(lid in seen for lid in g['local_ids'])]
                def similarity(k):return max((row.get('multiview_similarity',0) for lid in groups[k]['local_ids'] for row in retrieval.get(lid,[]) if row['local_id'] in fresh_lids),default=0)
                anchors=sorted(prior,key=lambda k:(similarity(k),len(groups[k]['local_ids'])),reverse=True)[:3]
                review(anchors+fresh,strong=True)
                seen.update(fresh_lids)
        people=[k for k,g in groups.items() if g['kind']=='person']
        for offset in range(0,len(people),21):
            review([k for k in people[offset:offset+21] if k in groups],strong=True)
    finally:
        backend.backend.model=None;backend.backend.processor=None;gc.collect();torch.cuda.empty_cache();backend.backend.settings=original
    entities=[];mapping={}
    for i,g in enumerate(sorted(groups.values(),key=lambda g:min(order[k] for k in g['local_ids']))):
        lids=g['local_ids'];eid=f'G{i+1:04d}';anchor=entries[min(lids,key=order.get)]
        entities.append(dict(entity_id=eid,local_ids=lids,kind=anchor['kind'],label=anchor['label'],appearance=anchor.get('appearance',''),reason='Joint visual batch identity partition; physical review follows.',uncertain=all(entries[k].get('uncertain',False) for k in lids)))
        mapping.update({k:eid for k in lids})
    return dict(version=VERSION,strategy='batched',entities=entities,mapping=mapping,cannot_link=forbidden,
        rejected_local_ids=[],rejection_reasons={},shots=[],external_context={},errors=reviews,
        global_input_fingerprint=global_input_fingerprint(bundle,local),review_model=review_model,
        physical_review_model={k:getattr(backend.settings,k) for k in ('model_id','model_revision')})


def build_batched_profiles(bundle,local,output,backend):
    from compact_identity import prepare_temporal,visual_board,validate_catalog,CatalogReview
    from story_semantics import local_lookup
    from view_quality import face_detail
    prepared,entries_doc,audit=prepare_temporal(bundle,local,output)
    entries=local_lookup(entries_doc);tracks={t['track_id']:t for s in prepared['shots'] for t in s['tracks']};families=defaultdict(list)
    for lid,e in entries.items():
        if e['kind']=='person':
            for tid in e['track_ids']:
                parent=tracks[tid].get('temporal_candidate',{}).get('parent_track_id')
                if parent:families[parent].append(lid)
    def priority(item):
        parent,lids=item;quality=0
        for lid in set(lids):
            for tid in entries[lid]['track_ids']:
                for o in tracks[tid].get('representatives',[]):
                    face=face_detail(prepared,o,Path(output))
                    if face:quality=max(quality,face['quality'])
        return quality,len(set(lids)),parent
    selected=[]
    for parent,lids in sorted(families.items(),key=priority,reverse=True):
        lids=list(dict.fromkeys(lids))
        if len(lids)<=6 and len(selected)+len(lids)<=16:selected+=lids
        if len(selected)>=12:break
    groups={f'T{i+1:02d}':dict(local_ids=[lid],kind='person') for i,lid in enumerate(dict.fromkeys(selected))};profiles={}
    folder=Path(output)/'repair_profile_cache';folder.mkdir(parents=True,exist_ok=True)
    if groups:
        images,shown,sources=visual_board(prepared,entries,groups,list(groups),output)
        images[0][1].save(folder/'profile_evidence.jpg',quality=94)
        prompt='''Describe ONLY the main physical target in EACH labeled T card independently. Each card shows a temporal fragment of a fallible tracker. Different cards may be DIFFERENT PEOPLE even in similar clothing. Inspect visible scalp/hair, facial hair and face separately for every card. Ignore a peripheral/background person's face at the edge of the target crop. Hidden traits are unknown. If target changes across its own views or is ambiguous, mixed=true. Do not group cards or use movie knowledge. Return ONE JSON {"profiles":[{"id":"T01","kind":"person","label":"short Korean visual label","appearance":"short target description","hair":"unknown","facial_hair":"unknown","presentation":"unknown","evidence":"visible details","frame":0,"mixed":false}],"groups":[]}. Every T ID exactly once. facial_hair: yes/no/unknown; presentation: female/male/unknown; hair: visible color/bald/unknown. Cite one actual shown frame per card.
INPUT: '''+json.dumps(dict(ids=list(groups),shown_frames=shown))
        backend.phase='batch_profiles:001'
        response=backend.request(prompt,CatalogReview,images=images,max_tokens=3500,image_pixels=1572864,validate=lambda r:validate_catalog(r,list(groups),shown))
        atomic_json(folder/'profile_review.json',dict(sources=sources,response=response.model_dump()))
        for profile in response.profiles:
            lid=groups[profile.id]['local_ids'][0];known=sum(getattr(profile,k)!='unknown' for k in ('hair','facial_hair','presentation'))
            profiles[lid]=dict(profile.model_dump(),target_verified=profile.verified and not profile.mixed and profile.kind=='person' and known>=2)
            entries[lid]['target_profile']=profiles[lid]
    atomic_json(folder/'reviewed_tracking.json',prepared);atomic_json(folder/'local_entities.json',entries_doc)
    atomic_json(folder/'compact_identity/profiles.json',profiles);atomic_json(folder/'candidates.json',audit)
    return folder


def review_frames(bundle,selected,entities,assignments,prototypes,variants,output,backend):
    """Several independently keyed exposures per request; G references shared once."""
    from PIL import Image,ImageDraw,ImageOps,ImageFont
    result={};selected=sorted(selected);dest=Path(output)/'batch_identity'
    # Reserve contextual judgments; fit physical frames into remaining calls.
    import math
    available=min(backend.limit-backend.used(),backend.remaining)-backend.reserve
    if selected and available<1:raise ValueError('No generation budget left for physical review')
    batch_size=max(10,math.ceil(len(selected)/max(1,available)))
    if batch_size>16:raise ValueError('Physical batch would exceed 16 frames; increase explicit budget')
    for offset in range(0,len(selected),batch_size):
        chunk=selected[offset:offset+batch_size];inputs={};images=[]
        allowed=sorted({x for sid,frame,rows in chunk for k,g,t,o in rows for x in ({g}|variants[g]) if x in prototypes})
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
        ref=Image.new('RGB',(280*min(5,max(1,len(allowed))),320*((len(allowed)+4)//5)),'#edf1f7');draw=ImageDraw.Draw(ref)
        for i,g in enumerate(allowed):
            o=prototypes[g];path=Path(bundle['asset_root'])/(o.get('crop') or o['image'])
            with Image.open(path) as im:crop=im.convert('RGB') if o.get('crop') else im.convert('RGB').crop(o['bbox'])
            crop=ImageOps.contain(crop,(270,270));x=(i%5)*280;y=(i//5)*320;ref.paste(crop,(x,y+35));draw.text((x+5,y+3),g,font=font,fill='black')
        if allowed:images.append(('REFERENCE identities G: NOT extra bodies in any frame',ref))
        for n,(sid,frame,rows) in enumerate(chunk):
            key=f'F{offset+n+1:03d}';im=board(bundle,[o for k,g,t,o in rows],{},[],zoom_context=True)
            im.save(dest/f'{key}.jpg',quality=94);images.append((f'{key}: ONE exposure, D crops are magnifications of that SAME exposure',im))
            inputs[key]=dict(shot=sid,frame=frame,ids=[f'D{i+1}' for i in range(len(rows))],
                hints={f'D{i+1}':assignments.get(k,g) for i,(k,g,t,o) in enumerate(rows)})
        def validate(r):
            if set(r.frames)!=set(inputs):raise ValueError('Use exactly these frame keys: '+str(list(inputs)))
            for key,meta in inputs.items():r.frames[key].bodies(len(meta['ids']),allowed)
            return r
        prompt='''Independently review ALL labeled exposures Fxxx. G cards are reference photos from other times, NOT detections. Each F has its OWN D1,D2,... labels. Return {"F001":{"assignments":{"D1":"G0001","D2":"new1"},"evidence":{"D1":"short visible reason","D2":"short visible reason"}}} for every supplied frame and every D in it, exactly once. The ROOT keys are F IDs; do not wrap them in a frames property. G IDs may only appear as assignment VALUES, never frame or detection keys.
First count PHYSICAL BODIES in each exposure. Same torso/face/attached limbs and duplicate tight/full crops get the SAME value. A shadow belongs with its visible owner, not a new person. Different physically separate people or animals must have different values. A carried animal is independent. Judge the MAIN FOREGROUND TORSO at the CENTER of each D crop, not the other person's face behind its edge. Then match body to G reference using visible hair, face, build, clothing and scene continuity. Back/side views and attached limbs may use the supported owner's identity. Existing hints can contain tracking switches and are NOT proof. Do not force incompatible appearance into a hint. Unrecognizable independent bodies may use new1,new2, unique within that frame. Do NOT create an extra body because an attached part lacks a face. Reasons must be short, target-specific, and consistent with the assignments.
INPUT: '''+json.dumps(dict(frames=inputs,references={g:entities[g].get('label','') for g in allowed}),ensure_ascii=False)
        backend.phase=f'batch_frames:{offset//batch_size+1:03d}'
        response=backend.request(prompt,BatchFrames,images=images,max_tokens=4500,image_pixels=1572864,validate=validate,repair_json_closures=True,repair_root_closure=True)
        atomic_json(dest/f'frames_{offset//batch_size+1:03d}.json',dict(input=inputs,response=response.model_dump()))
        for n,(sid,frame,rows) in enumerate(chunk):
            key=f'F{offset+n+1:03d}';result[(sid,frame)]=(response.frames[key].bodies(len(rows),allowed),allowed)
    return result


def label_consensus(bundle,local,catalog,profile_root):
    """Name a group from independently bound profile votes, never alter identity."""
    from catalog_repair import observation_key
    from compact_identity import canonical
    root=Path(profile_root);prepared=json.loads((root/'reviewed_tracking.json').read_text());pl=json.loads((root/'local_entities.json').read_text());profiles=json.loads((root/'compact_identity/profiles.json').read_text())
    owners={tid:catalog['mapping'][e['local_id']] for s in local['shots'] for e in s['entities'] for tid in e['track_ids']}
    current={observation_key(s['shot_id'],o):owners[t['track_id']] for s in bundle['shots'] for t in s['tracks'] if t['track_id'] in owners for o in t['observations']}
    tracks={t['track_id']:t for s in prepared['shots'] for t in s['tracks']};votes=defaultdict(list)
    colors={'붉은색':'red','붉은':'red','빨간색':'red','빨강':'red','갈색':'brown','검은색':'black','검정':'black','금발':'blond','흰색':'white','회색':'gray','대머리':'bald'}
    for shot in pl['shots']:
        for e in shot['entities']:
            profile=profiles.get(e['local_id'],{})
            if not profile.get('target_verified'):continue
            cited=[o for tid in e['track_ids'] for o in tracks[tid]['observations'] if o['frame']==profile.get('frame')]
            gids={current[observation_key(shot['shot_id'],o)] for o in cited if observation_key(shot['shot_id'],o) in current}
            if len(gids)!=1:continue
            votes[next(iter(gids))].append((shot['shot_id'],dict(profile,hair=colors.get(profile['hair'],profile['hair']))))
    for e in catalog['entities']:
        rows=votes.get(e['entity_id'],[])
        if not rows or e['kind']!='person':continue
        combined=dict(rows[0][1])
        for field in ('hair','facial_hair','presentation'):
            options={p[field] for sid,p in rows if p[field]!='unknown'}
            def support(v):return len({sid for sid,p in rows if p[field]==v}),sum(p[field]==v for sid,p in rows)
            ranked=sorted(options,key=lambda v:(support(v),v),reverse=True)
            combined[field]=ranked[0] if ranked and (len(ranked)==1 or support(ranked[0])>support(ranked[1])) else 'unknown'
        named=canonical(dict(local_ids=['vote'],kind='person'),{'vote':combined})
        e['label']=named['label'];e['label_evidence']=dict(method='Target-profile consensus; does not affect membership',profile_count=len(rows),fields={k:combined[k] for k in ('hair','facial_hair','presentation')})
    return catalog


def run_batched(bundle,local,output,backend,max_calls=10,review_settings=None,context=None):
    from catalog_repair import repair
    from context_reassignment import reassign
    limited=LimitedBackend(backend,max_calls)
    catalog=global_catalog(bundle,local,output,limited,review_settings)
    profiles=build_batched_profiles(bundle,local,output,limited)
    limited.reserve=2
    def batching(*args):return review_frames(*args,backend=limited)
    bundle,local,catalog=repair(bundle,local,catalog,output,limited,profiles,batch_review=batching)
    limited.reserve=0
    catalog=label_consensus(bundle,local,catalog,profiles)
    catalog=reassign(bundle,local,catalog,output,limited,review_settings or catalog['review_model'],context)
    catalog=label_consensus(bundle,local,catalog,profiles)
    from story_semantics import global_input_fingerprint
    catalog['external_context']=context or {}
    catalog['global_input_fingerprint']=global_input_fingerprint(bundle,local)
    catalog.update(strategy='batched',identity_reconciliation=dict(strategy='context_reassignment',generations=limited.used(),max_generations=max_calls,cannot_link=[]))
    atomic_json(Path(output)/'batch_identity/audit.json',dict(version=VERSION,generations=limited.used(),limit=max_calls,policy='Visual provisional catalog, physical review, chronological contextual reassignment; no prior global catalog reused.'))
    return bundle,local,catalog
