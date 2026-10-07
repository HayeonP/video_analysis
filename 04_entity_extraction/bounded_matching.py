"""Chronological batch matching plus one catalog audit; an absolute generation budget."""
from copy import deepcopy
from fractions import Fraction
import json
import logging
import gc
import torch
from pydantic import BaseModel, Field
from typing import Literal
from low_freq_event.backend import ModelOutputError
from low_freq_event.storage import atomic_json, read_json
from identity_review import ReviewBackend
from sequential_identity import gallery, scene_clip
from efficient_tracks import rank_candidates

LOG=logging.getLogger('entities')
VERSION=2


class BudgetExhausted(RuntimeError):pass


class BudgetBackend(ReviewBackend):
    def __init__(self,settings,trace_dir,limit):
        super().__init__(settings,trace_dir)
        self.path=trace_dir.parent/'qwen_budget.json'
        self.limit=limit;self.phase='unassigned';self.reserve=0
        self.ledger=read_json(self.path) if self.path.exists() else dict(limit=limit,generations=0,events=[])
        if self.ledger['limit']!=limit:raise ValueError('Qwen budget changed; use a new output directory.')
        self.save()

    @property
    def remaining(self):return max(0,self.limit-self.ledger['generations'])

    def save(self):atomic_json(self.path,self.ledger)

    def _generate(self,*args,**kwargs):
        if self.remaining <= self.reserve:raise BudgetExhausted('Qwen generation limit reached or final slot reserved')
        # Reserve before generation: failures, OOMs and structured-output retries count too.
        self.ledger['generations']+=1
        row=dict(number=self.ledger['generations'],phase=self.phase,status='started')
        self.ledger['events'].append(row);self.save()
        LOG.info('Qwen actual generation %d/%d [%s]',self.ledger['generations'],self.limit,self.phase)
        try:
            result=super()._generate(*args,**kwargs)
            row['status']='generated';return result
        except BaseException as exc:
            row.update(status='failed',error=str(exc));raise
        finally:self.save()


class EntityGroup(BaseModel):
    ids:list[str]=Field(min_length=1)
    match_id:str|None=None
    label:str=Field(min_length=1,max_length=60)
    appearance:str=Field(default='',max_length=400)
    kind:Literal['person','animal','object']
    reason:str


class KindCorrection(BaseModel):
    id:str
    kind:Literal['person','animal','object']
    reason:str


class BatchDecision(BaseModel):
    scene_summary:str=''
    corrections:list[KindCorrection]=Field(default_factory=list)
    groups:list[EntityGroup]=Field(default_factory=list)
    rejected_ids:list[str]=Field(default_factory=list)


POLICY='''Use observed visual identity and scene context together. Targets can change lighting, clothes, pose and apparent size. An obscured face is unknown evidence, not evidence of another person. Ignore provisional names/classes as identity truth. Consider long-range reappearances and action continuity, but never use an invented story or movie knowledge. A carrier and the carried animal are separate individuals. NEVER merge a person with an animal/object. Different simultaneously visible boxes must remain separate. Existing summaries and assignments are hypotheses, not ground truth. Do not create entities for clothing, body parts, scenery, text or shadows.\n'''


def is_person(entry):return entry['kind']=='person'


def frame_keys(entry):return {o['frame_key'] for o in entry['observations']}


def compatible(entries):
    if len({is_person(e) for e in entries})>1:return False
    occupied=set()
    for entry in entries:
        keys=frame_keys(entry)
        if occupied & keys:return False
        occupied.update(keys)
    return True


def compact_frames(frames,shots,count=6):
    selected=[]
    for sid in shots:
        pool=[f for f in frames if f['shot_id']==sid]
        if not pool:continue
        indices={round(i*(len(pool)-1)/max(1,min(count,len(pool))-1)) for i in range(min(count,len(pool)))}
        selected.extend(pool[i] for i in sorted(indices))
    return selected


def ask(backend,prompt,frames,entries,output,dest,fps,*,final=False):
    images,meta=gallery(entries,output,dest,'targets')
    meta=[{k:e[k] for k in ('id','label','kind','appearance','shots')} for e in meta]
    inventory=dict(version=VERSION,entities=meta,
        frames=[dict(frame_key=f['frame_key'],shot_id=f['shot_id'],time=f['full_time_sec']) for f in frames])
    request=prompt+'\nTARGETS: '+json.dumps(inventory,ensure_ascii=False)
    atomic_json(dest/'input.json',dict(prompt=request))
    # At most one retry at a site; reserve one actual generation for the final audit.
    backend.reserve=0 if final else 1
    for attempt in range(2):
        try:
            result=backend.request(request,BatchDecision,clip=scene_clip(frames,entries,output,fps),
                images=images,max_tokens=3072,image_pixels=196608)
            atomic_json(dest/'result.json',result.model_dump());return result
        except BudgetExhausted as exc:
            atomic_json(dest/'budget_fallback.json',dict(error=str(exc)))
            request+='\nReturn concise valid JSON following the schema exactly.'
            continue
        except (ModelOutputError,torch.OutOfMemoryError) as exc:
            atomic_json(dest/f'failure_{attempt+1}.json',dict(error=str(exc)))
            request+='\nReturn concise valid JSON following the schema exactly.'
            if isinstance(exc,torch.OutOfMemoryError):
                gc.collect();torch.cuda.empty_cache()
    return None


def apply_groups(result,current,memory,assignments,joins,phase):
    used=set();rejected=set()
    if result:
        for c in result.corrections:
            if c.id in current and c.reason.strip():
                current[c.id]['kind']=c.kind
                current[c.id]['kind_correction']=c.model_dump()
        # Validate entire proposed groups before changing memory; never silently lose IDs.
        for group in result.groups:
            ids=group.ids
            if len(ids)!=len(set(ids)) or not set(ids)<=set(current) or used & set(ids):continue
            target=group.match_id
            if target is not None and target not in memory:continue
            entries=[current[k] for k in ids]+([memory[target]] if target else [])
            if not compatible(entries) or (group.kind=='person')!=is_person(entries[0]):continue
            if target is None:
                target=f'G{len(memory)+1:04d}'
                memory[target]=dict(label=group.label,appearance=group.appearance,kind=group.kind,
                                    observations=[],members=[],history=[])
            for key in ids:
                if memory[target]['members']:
                    joins.append(dict(target=target,retired=key,method='Qwen bounded batch',reason=group.reason))
                memory[target]['observations'].extend(current[key]['observations'])
                memory[target]['members'].append(key);assignments[key]=target
                memory[target]['history'].append(dict(shot_id=current[key]['shot_id'],reason=group.reason,
                                                      label=group.label,appearance=group.appearance))
            used.update(ids)
        rejected=(set(result.rejected_ids)&set(current))-used
    for key,entry in current.items():
        if key in used or key in rejected:continue
        # Unreviewed or invalid model output stays provisional; never silently forced together.
        target=f'G{len(memory)+1:04d}'
        memory[target]=dict(deepcopy(entry),members=[key],history=[])
        assignments[key]=target
    return sorted(rejected)


def match_bounded(groups,frames,timeline,output,settings,limit=8,batch_shots=4):
    root=output/'bounded';root.mkdir(exist_ok=True)
    backend=BudgetBackend(settings,output/'review_traces',limit)
    memory={};assignments={};joins=[];history=[];rejected=[]
    shots=[s['id'] for s in timeline['shots']];fps=float(Fraction(timeline['fps']))
    # Fixed finite batches. Long inputs use larger batches, never increase the call ceiling.
    import math
    batch_shots=max(batch_shots,math.ceil(len(shots)/max(1,limit-1)))
    for index,start in enumerate(range(0,len(shots),batch_shots)):
        selected=shots[start:start+batch_shots]
        current={k:e for k,e in groups.items() if e['shot_id'] in selected}
        if not current:
            history.append(dict(shots=selected,summary='No detector proposals.'));continue
        dest=root/f'batch_{index+1:02d}';dest.mkdir(exist_ok=True)
        candidates={k:rank_candidates(e,memory,limit=3) for k,e in current.items()}
        refs={r['entity_id'] for values in candidates.values() for r in values}
        memory_subset={k:memory[k] for k in memory if k in refs}
        images,meta=gallery(memory_subset,output,dest,'memory')
        # ask() renders both current and retrieved memory entries; only current IDs may be assigned.
        entries={**current,**memory_subset}
        meta=[{k:e[k] for k in ('id','label','kind','appearance','shots')} for e in meta]
        prompt=POLICY+'''Match all CURRENT tracklets jointly, with the chronological scene frames and MEMORY crops. Consolidate fragments within and across these shots. For each physical individual give a concise Korean label and stable appearance. Match an existing memory ID if supported; otherwise null creates a new identity. Compare current people even when outfit/lighting differs. Look carefully at anatomy and size to keep distinct flying creatures separate while linking their multiple views. Rejected IDs must be clear false positives, never merely dark/small subjects. If a detector misclassified a dragon as person or a human as another class, explicitly correct that CURRENT tracklet via corrections[{id,kind,reason}] BEFORE identity grouping. This is a visual class correction, not a cross-species identity merge.
Return each current tracklet exactly once in groups[].ids or rejected_ids. Memory IDs may appear ONLY in match_id, NEVER in ids. A group may combine several CURRENT IDs. No new boxes or IDs.
ONLY JSON {"scene_summary":"brief observed actions/interactions", "corrections":[{"id":"current ID if misclassified","kind":"person|animal|object","reason":"visible subject evidence"}], "groups":[{"ids":["CURRENT ID"],"match_id":null,"label":"짧은 한국어 이름","appearance":"stable visual traits","kind":"person|animal|object","reason":"specific identity evidence"}],"rejected_ids":[]}.
'''+json.dumps(dict(current_ids=list(current),retrieved_candidates=candidates,memory=meta,shot_history=history),ensure_ascii=False)
        backend.phase=f'batch_{index+1}'
        result=ask(backend,prompt,compact_frames(frames,selected),entries,output,dest,fps)
        rejected+=apply_groups(result,current,memory,assignments,joins,backend.phase)
        history.append(dict(shots=selected,summary=result.scene_summary if result else 'Unreviewed: generation budget or invalid response'))
        atomic_json(dest/'memory.json',{k:{a:v for a,v in e.items() if a!='observations'} for k,e in memory.items()})
    # One global catalog request, no page-pair search and no feedback into preceding stages.
    if len(memory)>1:
        dest=root/'final';dest.mkdir(exist_ok=True)
        backend.phase='final_catalog'
        prompt=POLICY+'''Perform ONE final joint clustering of this whole entity catalog. Find fragments of the SAME individual across the video, including people with changed clothing and animals seen from different views. Use full scenes and multiple crop views. Names like bird/airplane/dragon may be detector errors. Do not assume an entity count. Group each catalog ID once, either as a singleton or with duplicates. Only merge with positive visual/contextual evidence. Keep unclear subjects as singletons. match_id must be null. You may reject clear false-positive IDs only. Explicit class corrections may be supplied as corrections[{id,kind,reason}] before grouping; distinguish a wrong detector class from an identity merge.
ONLY JSON {"scene_summary":"overall observed sequence", "corrections":[], "groups":[{"ids":["catalog IDs"],"match_id":null,"label":"짧은 한국어 이름","appearance":"stable appearance","kind":"person|animal|object","reason":"grounded evidence"}],"rejected_ids":[]}.
'''+json.dumps(dict(shot_history=history),ensure_ascii=False)
        result=ask(backend,prompt,compact_frames(frames,shots,2),memory,output,dest,fps,final=True)
        if result:
            for c in result.corrections:
                if c.id in memory and c.reason.strip():memory[c.id]['kind']=c.kind
            consumed=set()
            for group in result.groups:
                ids=group.ids
                if group.match_id is not None or len(ids)!=len(set(ids)) or not set(ids)<=set(memory) or consumed&set(ids):continue
                entries=[memory[k] for k in ids]
                if not compatible(entries) or (group.kind=='person')!=is_person(entries[0]):continue
                target=ids[0];memory[target].update(label=group.label,appearance=group.appearance,kind=group.kind)
                for old in ids[1:]:
                    memory[target]['observations'].extend(memory[old]['observations'])
                    memory[target]['members'].extend(memory[old]['members'])
                    for track in memory[old]['members']:assignments[track]=target
                    joins.append(dict(target=target,retired=old,method='Qwen one final catalog audit',reason=group.reason))
                consumed.update(ids)
            # Final audit cannot erase unreviewed targets. Only explicit unused false positives.
            for old in (set(result.rejected_ids)&set(memory))-consumed:
                for track in memory[old]['members']:assignments.pop(track,None);rejected.append(track)
    atomic_json(root/'summary.json',dict(version=VERSION,tracklets=len(groups),
        final_entities=len(set(assignments.values())),rejected_tracklets=sorted(set(rejected)),
        actual_qwen_generations=backend.ledger['generations'],qwen_limit=limit,
        fixed_batch_shots=batch_shots,history=history,no_reverse_pass=True,no_pairwise_model_loop=True))
    atomic_json(root/'assignments.json',assignments)
    if backend.model is not None:
        import gc,torch
        del backend.model;gc.collect();torch.cuda.empty_cache()
    return memory,assignments,joins


def export_mapping(groups,memory,assignments,*,identity_method='bounded batch memory and one catalog audit'):
    registry={};visual={'assignments':{}};observed=[];ids={}
    for track,entry in groups.items():
        if track not in assignments:continue
        key=assignments[track];ids.setdefault(key,f'E{len(ids)+1:04d}')
        eid=ids[key];entity=memory[key]
        status=entity.get('identity_status','unresolved' if key.startswith('U') else 'estimated')
        registry.setdefault(eid,dict(entity_id=eid,identity_status=status,reference_key=key))
        for source in entry['observations']:
            o=deepcopy(source);o.update(label=entity['label'],appearance=entity['appearance'],entity_kind=entity['kind'])
            observed.append(o);visual['assignments'][o['detection_id']]=dict(entity_id=eid,tracklet_id=track,
                identity_method=identity_method,identity_status=status,reference_key=key)
    return registry,visual,observed
