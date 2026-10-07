"""Finite, evidence-grounded shot -> cast -> narrative review.

Identity annotations used for evaluation are deliberately not imported here.
Every detector track must be assigned, explicitly rejected, or reported unreviewed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, Field, model_validator

from low_freq_event.backend import ModelOutputError
from low_freq_event.storage import atomic_json, read_json
from identity_review import ReviewBackend

LOG = logging.getLogger('story')
VERSION = 6
NARRATIVE_BATCH_SIZE = 10
Kind = Literal['person', 'animal', 'object', 'unknown']


class BudgetExhausted(RuntimeError):
    pass


class StoryBackend(ReviewBackend):
    """Durable actual-generation counter; cache hits never consume a call."""
    def __init__(self, settings, output, limit=24):
        super().__init__(settings, output / 'review_traces')
        self.path = output / 'qwen_budget.json'
        self.limit = limit
        self.phase = 'initializing'
        self.reserve = 0
        self.ledger = read_json(self.path) if self.path.exists() else {
            'limit': limit, 'generations': 0, 'events': []}
        if self.ledger['limit'] != limit:
            raise ValueError('A resumed run must keep its original generation budget.')
        atomic_json(self.path, self.ledger)

    @property
    def remaining(self):
        return max(0, self.limit - self.ledger['generations'])

    def load(self):
        if self.model is not None:return
        import torch
        from transformers import AutoConfig
        from low_freq_event.backend import QwenBackend
        config=AutoConfig.from_pretrained(self.settings.model_id,revision=self.settings.model_revision)
        text_config=getattr(config,'text_config',config)
        hidden=getattr(text_config,'hidden_size',0)
        layers=getattr(text_config,'num_hidden_layers',0)
        intermediate=getattr(text_config,'intermediate_size',0)
        vocab=getattr(text_config,'vocab_size',0)
        # Conservative dense-transformer estimate, plus room for vision and KV state.
        estimated_parameters=layers*(4*hidden*hidden+3*hidden*intermediate)+vocab*hidden
        total=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        if estimated_parameters*2>total*.75:
            self.precision_mode='NF4 4bit (BF16 compute)'
            if torch.cuda.device_count()>1:
                from transformers import AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration
                quantization=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_skip_modules=['visual','lm_head'])
                self.processor=AutoProcessor.from_pretrained(self.settings.model_id,revision=self.settings.model_revision)
                limits={i:int(torch.cuda.get_device_properties(i).total_memory*.65) for i in range(torch.cuda.device_count())}
                self.model=Qwen3VLForConditionalGeneration.from_pretrained(self.settings.model_id,
                    revision=self.settings.model_revision,quantization_config=quantization,dtype=torch.bfloat16,
                    device_map='balanced',max_memory=limits,attn_implementation='sdpa').eval()
                self.precision_mode+='; distributed across GPUs'
            else:QwenBackend.load(self)
        else:
            self.precision_mode='BF16'
            ReviewBackend.load(self)

    def request(self,prompt,schema,**kwargs):
        rendering=dict(image_pixels=kwargs.get('image_pixels') or self.settings.reference_pixels,
                       max_tokens=kwargs.get('max_tokens') or self.settings.max_new_tokens)
        return super().request(prompt+'\nREQUEST_RENDERING: '+json.dumps(rendering,sort_keys=True),schema,**kwargs)

    def _generate(self, *args, **kwargs):
        if self.remaining <= self.reserve:
            raise BudgetExhausted(f'Qwen budget exhausted/reserved: {self.remaining} remaining, {self.reserve} reserved')
        self.ledger['generations'] += 1
        row = dict(number=self.ledger['generations'], phase=self.phase, status='started',
                   model_id=self.settings.model_id,model_revision=self.settings.model_revision)
        self.ledger['events'].append(row)
        atomic_json(self.path, self.ledger)
        started = time.perf_counter()
        LOG.info('Qwen %d/%d [%s]', self.ledger['generations'], self.limit, self.phase)
        try:
            result = super()._generate(*args, **kwargs)
            row['status'] = 'generated'
            return result
        except BaseException as exc:
            row.update(status='failed', error=str(exc))
            raise
        finally:
            row['elapsed_sec'] = round(time.perf_counter() - started, 3)
            row['precision']=getattr(self,'precision_mode','not_loaded')
            if self.model is not None:row['device_map']={k:str(v) for k,v in getattr(self.model,'hf_device_map',{}).items()}
            atomic_json(self.path, self.ledger)


class Segment(BaseModel):
    track_id: str
    first_frame: int
    last_frame: int


class LocalEntity(BaseModel):
    local_id: str
    track_ids: list[str] = Field(min_length=1)
    kind: Kind
    label: str
    appearance: str
    reason: str = ''
    segments: list[Segment] = Field(default_factory=list)
    uncertain: bool = False


class Event(BaseModel):
    subject: str
    action: str
    object: str | None = None
    start_sec: float = 0
    end_sec: float = 0
    evidence_frames: list[int] = Field(default_factory=list)
    certainty: Literal['observed', 'inferred', 'uncertain'] = 'observed'


class TrackRelation(BaseModel):
    left: str
    right: str
    relation: Literal['same_individual','part_of','held_by','carried_by','different_individual','uncertain']
    reason: str = ''


class ShotDecision(BaseModel):
    track_assignments: dict[str,str | list[str]] = Field(default_factory=dict)
    entities: list[LocalEntity]
    rejected_tracks: list[str] = Field(default_factory=list)
    rejection_reasons: dict[str, str] = Field(default_factory=dict)
    summary: str
    events: list[Event] = Field(default_factory=list)
    missing_objects: list[str] = Field(default_factory=list)
    relations: list[TrackRelation] = Field(default_factory=list)
    track_descriptions: dict[str,str] = Field(default_factory=dict)
    track_roles: dict[str,Literal['whole_subject','body_part','handled_object','worn_item','background','uncertain']] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def assignments_to_partition(cls,value):
        if not isinstance(value,dict) or not value.get('track_assignments'):return value
        value=deepcopy(value)
        owners={e['local_id']:[] for e in value.get('entities',[])};rejected=[]
        for key,target in value['track_assignments'].items():
            if target=='reject':rejected.append(key);continue
            for local_id in target if isinstance(target,list) else [target]:
                if local_id not in owners:raise ValueError(f'Unknown local ID in assignment: {local_id}')
                owners[local_id].append(key)
        for entity in value['entities']:entity['track_ids']=owners[entity['local_id']]
        value['rejected_tracks']=rejected
        return value


class GlobalEntity(BaseModel):
    local_ids: list[str] = Field(min_length=1)
    kind: Kind
    label: str
    appearance: str
    reason: str
    uncertain: bool = False


class GlobalDecision(BaseModel):
    entities: list[GlobalEntity]
    rejected_local_ids: list[str] = Field(default_factory=list)
    rejection_reasons: dict[str,str] = Field(default_factory=dict)


class RevisedShot(BaseModel):
    shot_id: str
    summary: str
    events: list[Event] = Field(default_factory=list)
    correction_reason: str = ''


class NarrativeDecision(BaseModel):
    shots: list[RevisedShot]


def _font(size=18):
    for path in (Path(__file__).parent/'assets/NanumGothic-Regular.ttf',
                 Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def asset_path(bundle, output, relative):
    p = Path(relative)
    return p if p.is_absolute() else Path(bundle.get('asset_root', output)) / p


def samples_evenly(items, count):
    if len(items) <= count:
        return items
    return [items[i] for i in sorted({round(i*(len(items)-1)/(count-1)) for i in range(count)})]


def replace_ids(text, mapping):
    if not mapping:return text
    pattern=r'(?<![A-Za-z0-9_:])('+ '|'.join(re.escape(k) for k in sorted(mapping,key=len,reverse=True))+r')(?![A-Za-z0-9_:])'
    return re.sub(pattern,lambda match:mapping[match.group(1)],text)


def ensure_representatives(bundle,shot,track,output):
    """A split interval always gets real crops, even between original keyframes."""
    if track.get('representatives'):return
    import av
    selected=samples_evenly(track['observations'],min(6,len(track['observations'])))
    desired={o['frame']:o for o in selected}
    root=Path(bundle.get('asset_root',output)); dest=root/'split_assets'/shot['shot_id'];dest.mkdir(parents=True,exist_ok=True)
    with av.open(shot['source']) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index not in desired:continue
            obs=desired[index];im=frame.to_image().convert('RGB')
            image_path=dest/f'f{index:08d}.jpg';im.save(image_path,quality=94)
            crop_path=dest/f'{track["track_id"].replace(":","_")}_{index}.jpg'
            im.crop(tuple(round(v) for v in obs['bbox'])).save(crop_path,quality=94)
            obs.update(image=str(image_path.relative_to(root)),crop=str(crop_path.relative_to(root)))
    track['representatives']=selected


def crop_sheet(bundle, tracks, output, label, *, views=6, masked=False, context=False):
    refs = []
    for track in tracks:
        pool = track.get('representatives') or [o for o in track['observations'] if o.get('crop')]
        refs.extend(pool)
    refs = sorted(refs, key=lambda o: (o.get('original_time_sec', 0), o['frame']))
    refs = samples_evenly(refs, views)
    render_refs=[(ref,False) for ref in refs]
    if context and len(refs)>=2:
        render_refs=[(ref,False) for ref in samples_evenly(refs,4)]+[(refs[0],True),(refs[-1],True)]
    refs=[ref for ref,_ in render_refs]
    columns = min(3, max(1, len(refs)))
    rows = (len(refs)+columns-1)//columns
    sheet = Image.new('RGB', (columns*224, max(1, rows)*194+32), '#f3f5f8')
    draw = ImageDraw.Draw(sheet)
    draw.text((9, 5), label, font=_font(19), fill='#17223b')
    for i, (ref,is_context) in enumerate(render_refs):
        if not ref.get('crop'):
            continue
        if is_context and ref.get('image'):
            with Image.open(asset_path(bundle,output,ref['image'])) as source:
                source=source.convert('RGB');x1,y1,x2,y2=ref['bbox'];w=x2-x1;h=y2-y1
                region=(max(0,int(x1-1.5*w)),max(0,int(y1-1.5*h)),min(source.width,int(x2+1.5*w)),min(source.height,int(y2+1.5*h)))
                im=source.crop(region);pen=ImageDraw.Draw(im)
                pen.rectangle((x1-region[0],y1-region[1],x2-region[0],y2-region[1]),outline='#ffca44',width=2)
                thumb=ImageOps.contain(im,(216,158))
        else:
            with Image.open(asset_path(bundle, output, (ref.get('masked_crop') if masked else None) or ref['crop'])) as im:
                thumb = ImageOps.contain(im.convert('RGB'), (216, 158))
        x=(i%columns)*224; y=32+(i//columns)*194
        sheet.paste(thumb, (x+(224-thumb.width)//2, y))
        text=('CONTEXT ' if is_context else '')+f"f{ref['frame']}  {ref.get('time_sec', 0):.2f}s"
        draw.text((x+6,y+162), text, font=_font(15), fill='#17223b')
    return sheet


def gallery_pages(items, per_page=6):
    """Keep labeled candidates in a common coordinate system, at readable scale."""
    pages=[]
    for offset in range(0,len(items),per_page):
        batch=items[offset:offset+per_page]
        cols=min(2,len(batch));rows=(len(batch)+cols-1)//cols
        sheet=Image.new('RGB',(cols*680,rows*432),'#d8dce5')
        for index,(label,im) in enumerate(batch):
            thumb=ImageOps.contain(im,(672,424))
            sheet.paste(thumb,((index%cols)*680+4,(index//cols)*432+4))
        pages.append(('Labeled candidates '+', '.join(label for label,_ in batch),sheet))
    return pages


def geometry_context(shot,aliases):
    graph=shot.get('mask_graph',{})
    def mapped(pairs):
        return [[aliases[a],aliases[b]] for a,b in pairs if a in aliases and b in aliases]
    region_only=graph.get('cannot_link_scope')=='foreground_regions'
    return dict(cannot_merge=mapped(graph.get('physical_cannot_link',[]) if region_only else graph.get('cannot_link',[])),
        separated_mask_pairs=mapped(graph.get('cannot_link',[])) if region_only else [],
        inspect_relationship=mapped(graph.get('conditional_separation',[])),
        identity_change_candidates=[dict(track=aliases[t['track_id']],boundaries=t['identity_boundary_candidates'])
            for t in shot['tracks'] if t.get('identity_boundary_candidates')],
        explanation='cannot_merge: independently confirmed physical distinction. separated_mask_pairs/inspect_relationship: distinct masks may be separate individuals OR body parts OR background errors; inspect visual evidence. Same species does NOT mean same individual.')


def scene_sheet(bundle, shot, output, aliases):
    frames = samples_evenly(shot['scene_frames'], 9)
    sheet = Image.new('RGB', (1152, ((len(frames)+2)//3)*242), '#f3f5f8')
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        with Image.open(asset_path(bundle, output, frame['image'])) as im:
            im=im.convert('RGB')
            pen=ImageDraw.Draw(im)
            for track in shot['tracks']:
                if track['track_id'] not in aliases:continue
                obs=next((o for o in track['observations'] if o['frame']==frame['frame']),None)
                if obs:
                    box=obs['bbox']; pen.rectangle(box, outline='#ffcb47', width=3)
                    pen.text((max(0,box[0]),max(0,box[1]-20)), aliases[track['track_id']],
                             font=_font(20), fill='#ffcb47', stroke_width=1, stroke_fill='black')
            thumb=ImageOps.contain(im,(376,210))
        x=i%3*384; y=i//3*242
        sheet.paste(thumb,(x+(384-thumb.width)//2,y))
        draw.text((x+5,y+213),f"{shot['shot_id']} f{frame['frame']} {frame.get('time_sec',0):.2f}s",
                  font=_font(16),fill='#17223b')
    return sheet


LOCAL_PROMPT = '''Analyze ONE shot. The first sheet shows chronological scene frames. Each Txxx panel shows chronological crops of one tracker hypothesis, which may include detector errors or a change of subject.
First name what is visibly inside EACH Txxx panel in track_descriptions and classify its track_roles: whole_subject/body_part/handled_object/worn_item/background/uncertain. Then assign every track to a local physical Entity ID (L01, L02, ...) or "reject". Group views/fragments of the SAME INDIVIDUAL, not merely the same species. A face/arm belongs to its owner; a held tool or carried animal is a separate Entity. Reject background, terrain, shadows and worn clothing. Use visible anatomy and scene context; a tiny human may have no recognizable face. Animals/monsters are valid entities and use kind=animal.
Inspect simultaneous foreground geometry. cannot_merge pairs have confirmed physical distinction. separated_mask_pairs describe different MASK REGIONS and may still be parts of the same body, or false terrain detections. Use the synchronous pair images to decide. Distinct complete animals, including carrier/passenger, need separate IDs. Return relations for such pairs when useful.
If a track switches subjects, use multiple local IDs and exact nonoverlapping segments {track_id,first_frame,last_frame} covering its observations. Temporally split candidates may freely reunify when they show the same subject.
Write a brief concrete Korean scene description. Do not invent motion, intent or story. Fades/camera motion are not subject actions. events may be empty; global narrative review will inspect scene evidence later. Copy supplied frame numbers if giving events. Use concise stable appearance labels, no film/character names.
Return complete JSON:
{"track_descriptions":{"T001":"what is visibly depicted","T002":"..."},"track_roles":{"T001":"whole_subject","T002":"background"},"track_assignments":{"T001":"L01","T002":"reject"},"entities":[{"local_id":"L01","kind":"person|animal|object|unknown","label":"...","appearance":"...","reason":"visible identity evidence","uncertain":false}],"rejection_reasons":{"T002":"visible reason"},"summary":"brief scene observation using local IDs","events":[],"missing_objects":[],"relations":[]}
'''


def validate_local(result, shot, aliases):
    result=reconcile_handled_objects(result)
    from local_consistency import separate_conflicting_subjects
    result=separate_conflicting_subjects(result,shot,aliases)
    expected=set(aliases.values()); reverse={v:k for k,v in aliases.items()}
    ids=[e.local_id for e in result.entities]
    if len(ids)!=len(set(ids)):
        raise ValueError('local_id must be unique')
    owners={key:[] for key in expected}
    for entity in result.entities:
        if len(entity.track_ids)!=len(set(entity.track_ids)):
            raise ValueError('Repeated track within entity')
        for key in entity.track_ids:
            if key not in expected: raise ValueError(f'Unknown track {key}')
            owners[key].append(entity)
        if any(segment.track_id not in entity.track_ids for segment in entity.segments):
            raise ValueError('Segment must belong to an entity track_id')
    rejected=result.rejected_tracks
    if len(rejected)!=len(set(rejected)) or not set(rejected)<=expected:
        raise ValueError('Invalid rejected_tracks')
    for key in expected:
        entities=owners[key]
        if key in rejected:
            if entities: raise ValueError(f'{key} both assigned and rejected')
            continue
        if not entities: raise ValueError(f'Missing track {key}; return all tracks')
        if len(entities)>1 or any(s.track_id==key for e in entities for s in e.segments):
            track=next(t for t in shot['tracks'] if t['track_id']==reverse[key])
            actual={o['frame'] for o in track['observations']}; seen=set()
            for entity in entities:
                segments=[s for s in entity.segments if s.track_id==key]
                if not segments: raise ValueError(f'Split {key} requires exact nonoverlapping segments')
                if any(s.first_frame<0 or s.last_frame<s.first_frame or s.last_frame>=shot['frame_count'] for s in segments):
                    raise ValueError('Invalid split segment range')
                selected={f for f in actual if any(s.first_frame<=f<=s.last_frame for s in segments)}
                if not selected:raise ValueError('Split segment has no observed frames')
                if seen & selected: raise ValueError('Split segments overlap')
                seen |= selected
            if seen!=actual: raise ValueError('Split segments omitted observations')
    context=geometry_context(shot,aliases)
    physical_pairs=context['cannot_merge']+[[a,b] for a,b in context['separated_mask_pairs']
        if result.track_roles.get(a) in ('whole_subject','handled_object') and result.track_roles.get(b) in ('whole_subject','handled_object')]
    for left,right in physical_pairs:
        common={e.local_id for e in owners[left]} & {e.local_id for e in owners[right]}
        def assigned_frames(alias,entity):
            track=next(t for t in shot['tracks'] if t['track_id']==reverse[alias])
            spans=[s for s in entity.segments if s.track_id==alias]
            return {o['frame'] for o in track['observations'] if not spans or any(s.first_frame<=o['frame']<=s.last_frame for s in spans)}
        for local_id in common:
            entity=next(e for e in result.entities if e.local_id==local_id)
            if assigned_frames(left,entity)&assigned_frames(right,entity):
                raise ValueError(f'{left} and {right} are simultaneously separated physical foregrounds; cannot share {local_id}')
    for relation in result.relations:
        if relation.left not in expected or relation.right not in expected or relation.left==relation.right:
            raise ValueError('Relationship must refer to two supplied different track IDs')
        common={e.local_id for e in owners[relation.left]} & {e.local_id for e in owners[relation.right]}
        if common and relation.relation in ('held_by','carried_by','different_individual'):
            raise ValueError('Held/carried/distinct subjects require separate physical Entity IDs')
    duration=shot['frame_count']/float(shot['fps'])
    for event in result.events:
        if event.subject not in ids or event.object and event.object not in ids:
            raise ValueError('Event references unknown local entity')
        if not 0<=event.start_sec<=event.end_sec<=duration+1e-6:
            raise ValueError('Event times must be within this shot')
        if any(not 0<=frame<shot['frame_count'] for frame in event.evidence_frames):
            raise ValueError('Event evidence frame outside shot')
    return result


def reconcile_handled_objects(result):
    """An explicitly identified held object cannot be its holder's body.

    This enforces the model's own role labels without guessing an unseen box or
    using film-specific identities. The global review supplies the final name.
    """
    result=result.model_copy(deep=True);used={e.local_id for e in result.entities}
    additions=[]
    for entity in result.entities:
        handled=[tid for tid in entity.track_ids if result.track_roles.get(tid)=='handled_object']
        if not handled or entity.kind not in ('person','animal'):continue
        remaining=[tid for tid in entity.track_ids if tid not in handled]
        if not remaining:
            entity.kind='object';entity.label='소지 물체'
            entity.appearance='; '.join(result.track_descriptions.get(t,'') for t in handled)
            entity.reason+='; 분류 일관성 보정: 모든 관측이 VLM에서 소지 물체로 분류됨'
            continue
        entity.track_ids=remaining
        for tid in handled:
            index=1
            while f'L{index:02d}' in used:index+=1
            key=f'L{index:02d}';used.add(key)
            spans=[s for s in entity.segments if s.track_id==tid]
            entity.segments=[s for s in entity.segments if s.track_id!=tid]
            additions.append(LocalEntity(local_id=key,track_ids=[tid],kind='object',label='소지 물체',
                appearance=result.track_descriptions.get(tid,'검토에서 소지 물체로 분류된 대상'),
                reason='VLM의 handled_object 판정에 따라 소지자와 분리한 물리 개체',segments=spans))
            if result.track_assignments:result.track_assignments[tid]=key
            result.relations.append(TrackRelation(left=tid,right=remaining[0],relation='held_by',
                reason='VLM가 명시한 소지 물체 역할과 원래 소지자 배정을 보존'))
    result.entities.extend(additions)
    return result


def review_shots(bundle, output, backend):
    results=[]; failures=[]
    for index,shot in enumerate(bundle['shots']):
        sid=shot['shot_id']; dest=output/'local_review'/sid; dest.mkdir(parents=True,exist_ok=True)
        aliases={t['track_id']:f'T{i+1:03d}' for i,t in enumerate(shot['tracks'])}
        images=[('Full scene, chronological frames',scene_sheet(bundle,shot,output,aliases))]
        panels=[]
        for track in shot['tracks']:
            alias=aliases[track['track_id']]
            panels.append((alias,crop_sheet(bundle,[track],output,alias,masked=False)))
        images.extend(gallery_pages(panels))
        from story_relationship_images import relationship_images
        images.extend(relationship_images(bundle,shot,output,aliases))
        metadata=[]
        for track in shot['tracks']:
            obs=track['observations']
            metadata.append(dict(id=aliases[track['track_id']],frames=[obs[0]['frame'],obs[-1]['frame']],
                                 count=len(obs),temporal_candidate=track.get('temporal_candidate')))
        for n,(label,image) in enumerate(images): image.save(dest/f'input_{n:03d}.jpg',quality=94)
        prompt=f'PROTOCOL VERSION {VERSION}\n'+LOCAL_PROMPT+'\nSHOT: '+json.dumps(dict(id=sid,duration_sec=shot['frame_count']/float(shot['fps']),
            valid_frame_range=[0,shot['frame_count']-1],scene_frame_numbers=[f['frame'] for f in shot['scene_frames']],
            required_track_assignments={v:'Lxx or reject' for v in aliases.values()},tracks=metadata,
            foreground_geometry=geometry_context(shot,aliases)),ensure_ascii=False)
        backend.phase=f'local:{sid}'
        backend.reserve=max(0,len(bundle['shots'])-index-1)+2+max(1,(len(bundle['shots'])+NARRATIVE_BATCH_SIZE-1)//NARRATIVE_BATCH_SIZE)
        LOG.info('Local review %s: %d tracks, %d image sheets',sid,len(shot['tracks']),len(images))
        try:
            result=backend.request(prompt,ShotDecision,images=images,max_tokens=6000,image_pixels=1572864,
                                   validate=lambda r:validate_local(r,shot,aliases))
        except (ModelOutputError,BudgetExhausted) as exc:
            failures.append(dict(shot_id=sid,error=str(exc)))
            atomic_json(dest/'failure.json',failures[-1])
            continue
        raw=result.model_dump(); atomic_json(dest/'decision.json',raw)
        reverse={v:k for k,v in aliases.items()}; idmap={e.local_id:f'{sid}:{e.local_id}' for e in result.entities}
        row=dict(shot_id=sid,entities=[],rejected_tracks=[reverse[x] for x in result.rejected_tracks],
                 rejection_reasons={reverse.get(k,k):v for k,v in result.rejection_reasons.items()},
                 summary=raw['summary'],events=[],missing_objects=result.missing_objects,
                 track_descriptions={reverse.get(k,k):v for k,v in result.track_descriptions.items()},
                 track_roles={reverse.get(k,k):v for k,v in result.track_roles.items()},
                 relations=[dict(r.model_dump(),left=reverse[r.left],right=reverse[r.right]) for r in result.relations])
        row['summary']=replace_ids(row['summary'],idmap)
        track_lookup={t['track_id']:t for t in shot['tracks']}
        split_originals=set()
        for entity in result.entities:
            e=entity.model_dump(); e['local_id']=idmap[e['local_id']]; e['track_ids']=[]
            for alias in entity.track_ids:
                key=reverse[alias]; segments=[s for s in entity.segments if s.track_id==alias]
                if segments:
                    original=track_lookup[key]; child=deepcopy(original)
                    child['track_id']=key+':'+entity.local_id
                    child['observations']=[o for o in original['observations'] if any(s.first_frame<=o['frame']<=s.last_frame for s in segments)]
                    child['representatives']=[o for o in original.get('representatives',[]) if any(s.first_frame<=o['frame']<=s.last_frame for s in segments)]
                    child['split_from']=key
                    ensure_representatives(bundle,shot,child,output)
                    shot['tracks'].append(child);e['track_ids'].append(child['track_id']);split_originals.add(key)
                else: e['track_ids'].append(key)
            final_track_map={t['track_id']:t for t in shot['tracks']}
            observations=[o for tid in e['track_ids'] for o in final_track_map[tid]['observations']]
            e['observation_support']=dict(total_frames=len({o['frame'] for o in observations}),
                original_detection_frames=len({o['frame'] for o in observations if o.get('detector_supported') or o.get('evidence')=='observed_detection'}),
                note='Mask propagation alone does not independently confirm objectness or identity.')
            row['entities'].append(e)
        shot['tracks']=[t for t in shot['tracks'] if t['track_id'] not in split_originals]
        if split_originals:shot['split_original_tracks']=sorted(split_originals)
        for event in raw['events']:
            event['subject']=idmap[event['subject']]
            if event.get('object'):event['object']=idmap[event['object']]
            event['time_basis']='shot'; row['events'].append(event)
        results.append(row)
        atomic_json(output/'local_entities.json',dict(version=VERSION,shots=results,failures=failures))
    return dict(version=VERSION,shots=results,failures=failures)


def local_lookup(local):
    return {e['local_id']:dict(e,shot_id=s['shot_id']) for s in local['shots'] for e in s['entities']}


def global_images(bundle,local,output,reverse=False):
    trackmap={t['track_id']:t for s in bundle['shots'] for t in s['tracks']}
    items=list(local_lookup(local).values())
    if reverse:items.reverse()
    images=[]
    for entity in items:
        tracks=[trackmap[t] for t in entity['track_ids']]
        images.append((entity['local_id'],crop_sheet(bundle,tracks,output,entity['local_id'],views=6,masked=False,context=True)))
    return gallery_pages(images)


def visual_candidates(bundle,local):
    tracks={t['track_id']:t for s in bundle['shots'] for t in s['tracks']}
    entries=local_lookup(local);features={}
    for key,entry in entries.items():
        rows=[o['feature'] for tid in entry['track_ids'] for o in tracks[tid].get('representatives',[]) if o.get('feature')]
        if rows:
            matrix=np.asarray(rows,dtype=np.float32)
            features[key]=matrix/np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-8)
    result={}
    for key,feat in features.items():
        matches=[]
        for other,matrix in features.items():
            if other==key or entries[other]['shot_id']==entries[key]['shot_id']:continue
            similarities=feat@matrix.T
            score=float(np.sort(similarities.ravel())[-min(3,similarities.size):].mean())
            matches.append(dict(local_id=other,multiview_similarity=round(score,4)))
        result[key]=sorted(matches,key=lambda x:-x['multiview_similarity'])[:4]
    return result


def cannot_links(bundle, local):
    """Distinct non-overlapping simultaneously observed individuals cannot merge."""
    tracks={t['track_id']:t for s in bundle['shots'] for t in s['tracks']}; result=[]
    from entity_models import iou
    source_shots={s['shot_id']:s for s in bundle['shots']}
    for shot in local['shots']:
        entries=shot['entities']
        geometry=source_shots[shot['shot_id']].get('mask_graph')
        if geometry is not None:
            owners={}
            for entity in entries:
                for tid in entity['track_ids']:
                    track=tracks[tid]
                    for source in {tid,*track.get('source_track_ids',[]),*track.get('mask_graph_members',[])}:
                        owners.setdefault(source,set()).add(entity['local_id'])
            pairs=list(geometry.get('cannot_link',[]))
            if geometry.get('cannot_link_scope')=='foreground_regions':
                pairs=[pair for pair in pairs if all(shot.get('track_roles',{}).get(key) in ('whole_subject','handled_object') for key in pair)]
                pairs.extend(geometry.get('physical_cannot_link',[]))
            pairs.extend([r['left'],r['right']] for r in shot.get('relations',[])
                         if r['relation'] in ('held_by','carried_by','different_individual'))
            for left,right in pairs:
                for a in owners.get(left,[]):
                    for b in owners.get(right,[]):
                        if a!=b:result.append(sorted([a,b]))
            continue
        for i,a in enumerate(entries):
            ao={o['frame']:o for t in a['track_ids'] for o in tracks[t]['observations']}
            for b in entries[i+1:]:
                bo={o['frame']:o for t in b['track_ids'] for o in tracks[t]['observations']}
                shared=set(ao)&set(bo)
                separated=[f for f in shared if iou(ao[f]['bbox'],bo[f]['bbox'])<.2]
                if len(separated)>=min(3,len(shared)) and separated:
                    result.append([a['local_id'],b['local_id']])
    return [list(pair) for pair in sorted({tuple(sorted(pair)) for pair in result})]


GLOBAL_PROMPT='''Build a global physical-entity catalog across ALL shots from the labeled multi-view crop sheets and shot evidence below. Each input local entity is a hypothesis, not a different global identity. Inspect the visual appearances across angles, poses, lighting and scale. Connect reappearances when distinctive appearance and/or grounded visual-temporal continuity support the same individual. A hidden face, changed pose or different wording is not evidence of a new individual. Human/nonhuman classes and old labels can be corrected based on images. Different animals of the same species remain distinct when evidence supports distinct individuals. Same-color clothing alone is insufficient. Use local shot action/relationships as supporting evidence, never fictional movie knowledge. The output is a PARTITION: every local_id exactly once. Do not omit brief or uncertain entities. Merge duplicate local fragments when demonstrated; respect cannot_link pairs of simultaneously visible distinct individuals. Explain visual evidence for each group, including uncertainties. Korean compact labels and descriptions. Return COMPLETE concise JSON:
{"entities":[{"local_ids":["shot_0001:L01","shot_0002:L01"],"kind":"person","label":"...","appearance":"...","reason":"...","uncertain":false}]}
'''
GLOBAL_PROMPT+='''If a local entity is visibly a false detection (terrain, background, shadow, worn garment), put its local_id in rejected_local_ids with a specific visual rejection_reasons entry. Every input local_id must occur exactly once in an entity OR rejected_local_ids. An unfamiliar species or uncertain identity is not a rejection reason. Correct local classification mistakes after examining original crop context. Same-species resemblance alone cannot prove same individual. Do not preserve a false local entity just because the local model called it a person.\n'''


GLOBAL_PROMPT += "\nEach panel ends with CONTEXT views: yellow boxes mark the candidate inside the expanded original image. Nearby subjects are not automatically part of this candidate. Reassess rock/background hypotheses using context. Many propagated mask frames from one detection are not independent confirmations of a physical subject.\n"


def validate_global(result, expected, forbidden):
    ids=[x for e in result.entities for x in e.local_ids]+result.rejected_local_ids
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('Partition must cover every local_id exactly once; missing='+str(set(expected)-set(ids)))
    for entity in result.entities:
        members=set(entity.local_ids)
        if any(set(pair)<=members for pair in forbidden):
            raise ValueError('Cannot merge simultaneously visible distinct individuals')
    if any(not result.rejection_reasons.get(key,'').strip() for key in result.rejected_local_ids):
        raise ValueError('Every rejected local entity requires an explicit visual reason')
    return result


def resolve_global(bundle, local, output, backend):
    entries=local_lookup(local); forbidden=cannot_links(bundle,local)
    evidence=dict(entities=list(entries.values()),shot_summaries=[dict(shot_id=s['shot_id'],summary=s['summary'],events=s['events']) for s in local['shots']],
                  cannot_link=forbidden,visual_retrieval=visual_candidates(bundle,local),
                  retrieval_note='DINO multi-view similarity proposes candidates, not a hard identity decision; all entities remain shown.')
    result=None; errors=[]
    for reverse in (False,True):
        phase='global_reverse' if reverse else 'global_forward'; backend.phase=phase
        backend.reserve=max(1,(len(local['shots'])+NARRATIVE_BATCH_SIZE-1)//NARRATIVE_BATCH_SIZE)+(0 if reverse else 1)
        prompt=f'PROTOCOL VERSION {VERSION}\n'+GLOBAL_PROMPT+'\nEVIDENCE: '+json.dumps(evidence,ensure_ascii=False)
        if result is not None:
            prompt+='\nREVERSE AUDIT: inspect later shots first, then reassess earlier views. Previous partition is provisional; correct fragmentation and false merges. Return the complete revised partition.\n'+result.model_dump_json()
        try:
            result=backend.request(prompt,GlobalDecision,images=global_images(bundle,local,output,reverse),
                max_tokens=6000,image_pixels=1572864,validate=lambda r:validate_global(r,entries,forbidden))
            atomic_json(output/f'{phase}.json',result.model_dump())
        except (ModelOutputError,BudgetExhausted) as exc:
            errors.append(dict(phase=phase,error=str(exc)))
    if result is None:
        raise RuntimeError('Global review failed; no invented identity result: '+str(errors))
    entities=[]; mapping={}
    order={key:i for i,key in enumerate(entries)}
    for i,entity in enumerate(sorted(result.entities,key=lambda e:min(order[k] for k in e.local_ids))):
        row=entity.model_dump(); row['entity_id']=f'G{i+1:04d}'; entities.append(row)
        for key in row['local_ids']:mapping[key]=row['entity_id']
    value=dict(version=VERSION,entities=entities,mapping=mapping,cannot_link=forbidden,errors=errors,
               rejected_local_ids=result.rejected_local_ids,rejection_reasons=result.rejection_reasons)
    atomic_json(output/'global_entities.json',value)
    return value


def apply_global_rejections(local,global_result,output):
    """Keep the original local decisions; explicitly account for later false detections."""
    rejected=set(global_result.get('rejected_local_ids',[]))
    if not rejected:return local
    updated=deepcopy(local)
    for shot in updated['shots']:
        removed=[e for e in shot['entities'] if e['local_id'] in rejected]
        shot['global_rejected_entities']=removed
        for entity in removed:
            for tid in entity['track_ids']:
                shot['rejected_tracks'].append(tid)
                shot.setdefault('rejection_reasons',{})[tid]=global_result['rejection_reasons'][entity['local_id']]
        shot['entities']=[e for e in shot['entities'] if e['local_id'] not in rejected]
        shot['events']=[e for e in shot['events'] if e['subject'] not in rejected and e.get('object') not in rejected]
        shot['initial_summary_before_global_rejection']=shot['summary']
        shot['summary']=replace_ids(shot['summary'],{k:'[전역 검토에서 제외된 오검출]' for k in rejected})
    atomic_json(output/'final_local_entities.json',updated)
    return updated


def revise_narrative(bundle,local,global_result,output,backend):
    mapping=global_result['mapping']; result=[]; failures=[]
    shotmap={s['shot_id']:s for s in bundle['shots']}
    inputs=[]
    for shot in local['shots']:
        row=deepcopy(shot); row['initial_summary']=shot['summary']
        row['summary']=replace_ids(row['summary'],mapping)
        for event in row['events']:
            event['subject']=mapping[event['subject']]
            if event.get('object'):event['object']=mapping[event['object']]
        row['entities']=[mapping[e['local_id']] for e in shot['entities']]
        row['duration_sec']=shotmap[shot['shot_id']]['frame_count']/float(shotmap[shot['shot_id']]['fps'])
        row['scene_frame_numbers']=[f['frame'] for f in shotmap[shot['shot_id']]['scene_frames']]
        inputs.append(row)
    pages=max(1,(len(inputs)+NARRATIVE_BATCH_SIZE-1)//NARRATIVE_BATCH_SIZE)
    for page in range(pages):
        selected=inputs[page*len(inputs)//pages:(page+1)*len(inputs)//pages]
        if not selected:continue
        expected=[s['shot_id'] for s in selected]; backend.phase=f'narrative_{page+1}'; backend.reserve=pages-page-1
        prompt=f'PROTOCOL VERSION {VERSION}\n'+'''Revise the selected shot summaries using the FINAL global entity catalog and all chronological shot evidence. Local summaries are provisional hypotheses; correct entity names, actor/object roles and action continuity only when supported by supplied observations. Use global IDs explicitly. Do not invent dialogue, intentions, actions in other shots, or causal backstory. A global entity may occur in this shot only if present in its provided entities list. Write specific Korean chronological scene descriptions useful for screenplay planning. Preserve observed vs inferred vs uncertain. Event timestamps and evidence_frames remain local to EACH shot. Return every selected shot exactly once, concise complete JSON:
{"shots":[{"shot_id":"...","summary":"G0001 ...","events":[{"subject":"G0001","action":"...","object":null,"start_sec":0,"end_sec":1,"evidence_frames":[0],"certainty":"observed"}],"correction_reason":"..."}]}
'''+'''The supplied chronological scene images are primary evidence. Correct an initial summary when the images contradict it. Do not describe a stationary person as flying because a nearby animal has wings. Global fading/dimming is an edit, not physical exit. A title card has visible text and no invented action. Use the supplied scene frame numbers for evidence_frames.\n'''+json.dumps(dict(global_entities=global_result['entities'],all_shots=inputs,selected_shots=expected),ensure_ascii=False)
        def validate(response):
            if [s.shot_id for s in response.shots]!=expected:raise ValueError('Return selected_shots once in given order')
            lookup={s['shot_id']:s for s in selected}
            for row in response.shots:
                src=lookup[row.shot_id]
                if not set(re.findall(r'(?<![A-Za-z0-9_])G[0-9]+(?![A-Za-z0-9_])',row.summary))<=set(src['entities']):
                    raise ValueError('Summary names an entity not observed in this shot')
                for event in row.events:
                    if event.subject not in src['entities'] or event.object and event.object not in src['entities']:
                        raise ValueError('Event refers to an entity not observed in this shot')
                    if not 0<=event.start_sec<=event.end_sec<=src['duration_sec']+1e-6:raise ValueError('Event time out of shot')
                    if any(not 0<=f<shotmap[row.shot_id]['frame_count'] for f in event.evidence_frames):
                        raise ValueError('Event evidence outside shot')
            return response
        try:
            local_by_shot={s['shot_id']:s for s in local['shots']}
            scene_images=[]
            for selected_shot in selected:
                sid=selected_shot['shot_id']
                aliases={tid:mapping[e['local_id']] for e in local_by_shot[sid]['entities'] for tid in e['track_ids']}
                scene_images.append((sid+' chronological visual evidence',scene_sheet(bundle,shotmap[sid],output,aliases)))
            response=backend.request(prompt,NarrativeDecision,images=scene_images,validate=validate,
                                     max_tokens=6500,image_pixels=786432)
            reviewed={s.shot_id:s.model_dump() for s in response.shots}
        except (ModelOutputError,BudgetExhausted) as exc:
            failures.append(dict(phase=backend.phase,error=str(exc))); reviewed={}
        for source in selected:
            sid=source['shot_id']; row=reviewed.get(sid,dict(shot_id=sid,summary=source['summary'],events=source['events'],correction_reason='Global ID substitution only; semantic revision failed'))
            row.update(initial_summary=source['initial_summary'],entities=list(dict.fromkeys(source['entities'])),
                       start_sec=shotmap[sid]['full_start_sec'],end_sec=shotmap[sid]['full_start_sec']+source['duration_sec'],
                       reviewed=sid in reviewed)
            for event in row['events']:
                event['local_start_sec']=event['start_sec'];event['local_end_sec']=event['end_sec']
                event['start_sec']+=row['start_sec'];event['end_sec']+=row['start_sec'];event['time_basis']='full'
            result.append(row)
    global_result.update(shots=result,narrative_failures=failures)
    atomic_json(output/'global_entities.json',global_result)
    return global_result
