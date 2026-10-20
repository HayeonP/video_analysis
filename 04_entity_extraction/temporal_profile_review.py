"""Targeted, within-shot appearance checks for provisional track intervals."""
from copy import deepcopy
import json
from pathlib import Path
from low_freq_event.storage import atomic_json
from low_freq_event.backend import ModelOutputError


def review_temporal_profiles(bundle,local,output,backend,max_calls=4):
    from compact_identity import CatalogReview,visual_board,validate_catalog,canonical
    from story_semantics import local_lookup,BudgetExhausted
    from view_quality import face_detail
    output=Path(output);local=deepcopy(local);entries=local_lookup(local)
    tracks={t['track_id']:t for s in bundle['shots'] for t in s['tracks']}
    families={}
    for lid,e in entries.items():
        if e['kind']!='person':continue
        for tid in e['track_ids']:
            parent=tracks[tid].get('temporal_candidate',{}).get('parent_track_id')
            if parent:families.setdefault(parent,[]).append(lid)
    def priority(item):
        parent,lids=item;quality=0.
        for lid in set(lids):
            for tid in entries[lid]['track_ids']:
                for obs in tracks[tid].get('representatives',[]):
                    detail=face_detail(bundle,obs,output)
                    if detail:quality=max(quality,detail['quality'])
        return (quality,len(set(lids)),parent)
    candidates=sorted(families.items(),key=priority,reverse=True)
    original_reserve=backend.reserve;records=[]
    dest=output/'temporal_profile_review';dest.mkdir(exist_ok=True)
    for index,(parent,lids) in enumerate(candidates,1):
        lids=list(dict.fromkeys(lids))
        if len(lids)>6:continue
        groups={f'T{i+1:02d}':dict(local_ids=[lid],kind=entries[lid]['kind']) for i,lid in enumerate(lids)}
        images,shown,sources=visual_board(bundle,entries,groups,list(groups),output)
        folder=dest/f'{index:03d}';folder.mkdir(exist_ok=True)
        atomic_json(folder/'input.json',dict(parent=parent,groups=groups,shown_frames=shown,sources=sources))
        for n,(_,im) in enumerate(images):im.save(folder/f'evidence_{n:02d}.jpg',quality=90)
        prompt='''Describe ONLY the physical target in EACH labeled card, independently. These are temporal fragments of a possibly incorrect tracker. Different cards can be DIFFERENT PEOPLE even when clothing is similar. Carefully distinguish visible scalp/baldness from hair, facial hair, and facial appearance. Do not copy the first person's attributes into other cards. If a card contains multiple people and the target cannot be identified, or its own views switch people, mixed=true. Otherwise mixed=false. Hidden or unclear traits must be unknown. Do not infer names or identity across shots. Do not group cards. Return JSON {"profiles":[{"id":"T01","kind":"person","label":"short visual label","appearance":"visible target","hair":"unknown","facial_hair":"unknown","presentation":"unknown","evidence":"specific visible facial or scalp details","frame":0,"mixed":false}],"groups":[]} with exactly one profile per input ID. facial_hair must be yes/no/unknown, presentation female/male/unknown. hair can be a visible color, bald or unknown; do not copy slash-separated lists.
INPUT: '''+json.dumps(dict(ids=list(groups),shown_frames=shown))
        backend.phase=f'local_temporal_profile:{index:03d}'
        consumed=sum(e.get('phase','').startswith('local_temporal_profile:') for e in backend.ledger['events'])
        backend.reserve=max(original_reserve,backend.remaining-max(0,max_calls-consumed))
        try:
            response=backend.request(prompt,CatalogReview,images=images,max_tokens=2000,image_pixels=1572864,
                validate=lambda result:validate_catalog(result,list(groups),shown),repair_json_closures=True)
        except (ModelOutputError,BudgetExhausted) as exc:
            atomic_json(folder/'failure.json',dict(error=str(exc)));continue
        atomic_json(folder/'decision.json',response.model_dump())
        for p in response.profiles:
            lid=groups[p.id]['local_ids'][0]
            known=sum(getattr(p,k)!='unknown' for k in ('hair','facial_hair','presentation'))
            verified=p.verified and not p.mixed and p.kind=='person' and known>=2
            for shot in local['shots']:
                for entry in shot['entities']:
                    if entry['local_id']!=lid:continue
                    entry['target_profile']=dict(p.model_dump(),target_verified=verified,source=str(folder.relative_to(output)),model_id=backend.settings.model_id)
                    if verified:entry.update(canonical(dict(local_ids=[lid],kind='person'),{lid:p.model_dump()}))
            records.append(dict(local_id=lid,verified=verified,profile=p.model_dump()))
    backend.reserve=original_reserve
    atomic_json(dest/'audit.json',dict(generations=sum(e.get('phase','').startswith('local_temporal_profile:') for e in backend.ledger['events']),max_generations=max_calls,records=records,note='Within-shot target appearance only; no cross-shot identity decisions.'))
    return local
