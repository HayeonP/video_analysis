"""Appearance identities survive optional bounded semantic review failures.

Anchor reference observations are evidence only: they are never enrolled as
appearances in the output. The only exported membership is track -> anchor.
Reference membership starts from observed appearance links. Qwen labels known
identities, assigns ambiguous tracks, and reviews catalog relations once.
There are no pairwise model calls or reverse traversal.

Appearance thresholds below are starting heuristics, not calibrated identity
probabilities. Face/body/generic DINO support prevents a narrative-only answer
from attaching an unrelated target. An obscured face is not a negative verdict.
Every actual model generation, including retries and OOM, uses BudgetBackend's
persistent absolute ceiling. A final slot is reserved before generation; cached
answers remain usable with zero budget. This module does not launch inference
unless match_anchored() is explicitly invoked by the pipeline.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
import gc
import json
import logging
import math
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field, ValidationError, model_validator
import torch

from low_freq_event.backend import ModelOutputError
from low_freq_event.storage import atomic_json
from bounded_matching import BudgetBackend, BudgetExhausted, compact_frames
from entity_output import font, representative_score
from efficient_tracks import appearance_evidence, rank_candidates
from sequential_identity import scene_clip

LOG = logging.getLogger('entities')
VERSION = 3


@dataclass(frozen=True)
class MatchingConfig:
    """General input budgets and uncalibrated appearance heuristics.

    These values describe resource limits and embedding similarities, not any
    video's number of shots, characters, species, or expected identity count.
    The shared persistent Qwen generation ceiling remains independently capped.
    """
    max_batches: int = 5
    max_batch_tracks: int = 32
    overview_tracks: int = 24
    final_tracks: int = 24
    candidates: int = 3
    all_anchors_limit: int = 8
    overview_scene_frames: int = 2
    batch_scene_frames: int = 4
    final_scene_frames: int = 2
    max_scene_frames: int = 40
    clear_face_contradiction: float = .20
    batch_face: float = .50
    final_face: float = .45
    batch_body: float = .90
    final_body: float = .86
    batch_visual: float = .90
    final_visual: float = .86
    batch_joint_visual: float = .66
    final_joint_visual: float = .60
    batch_joint_body: float = .70
    final_joint_body: float = .65
    batch_object: float = .78
    final_object: float = .68

    def __post_init__(self):
        integer_fields = {
            'max_batches', 'max_batch_tracks', 'overview_tracks', 'final_tracks', 'candidates',
            'all_anchors_limit', 'overview_scene_frames', 'batch_scene_frames',
            'final_scene_frames', 'max_scene_frames',
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in integer_fields:
                if type(value) is not int or value < 1:
                    raise ValueError(f'{field.name} must be a positive integer')
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value) or not -1 <= value <= 1):
                raise ValueError(f'{field.name} must be a finite cosine threshold in [-1, 1]')
        if self.clear_face_contradiction > min(self.batch_face, self.final_face):
            raise ValueError('clear_face_contradiction must not exceed positive face thresholds')


MAX_OVERVIEW_TRACKS = MatchingConfig().overview_tracks
MAX_FINAL_TRACKS = MatchingConfig().final_tracks


class Correction(BaseModel):
    kind: Literal['person', 'animal', 'object']
    reason: str = Field(min_length=1)


class ReferenceCorrection(Correction):
    track_id: str


class AnchorSpec(BaseModel):
    anchor_id: str = Field(pattern=r'^A[0-9]{3}$')
    label: str = Field(min_length=1, max_length=24)
    kind: Literal['person', 'animal', 'object']
    appearance: str = Field(min_length=1, max_length=350)
    reference_track_ids: list[str] = Field(min_length=1)
    evidence: str = Field(min_length=1, max_length=350)


class OverviewDecision(BaseModel):
    scene_summary: str = Field(default='', max_length=1000)
    corrections: list[ReferenceCorrection] = Field(default_factory=list)
    anchors: list[AnchorSpec] = Field(default_factory=list)


class TrackDecision(BaseModel):
    track_id: str
    decision: Literal['match', 'unknown', 'reject']
    anchor_id: str | None = None
    correction: Correction | None = None
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)


class AssignmentDecision(BaseModel):
    scene_summary: str = ''
    assignments: list[TrackDecision] = Field(default_factory=list)
    invalid_items: list[dict] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def salvage_items(cls,value):
        if not isinstance(value,dict):return value
        data=dict(value);rows=data.get('assignments',[]);valid=[]
        invalid=[e for e in data.get('invalid_items',[]) if isinstance(e,dict)] if isinstance(data.get('invalid_items',[]),list) else []
        if not isinstance(rows,list):
            invalid.append(dict(reason='assignments must be a list',value=rows));rows=[]
        for index,row in enumerate(rows):
            try:valid.append(row if isinstance(row,TrackDecision) else TrackDecision.model_validate(row))
            except (ValidationError,ValueError,TypeError) as exc:
                invalid.append(dict(index=index,reason=str(exc),value=row))
        if not isinstance(data.get('scene_summary',''),str):data['scene_summary']=''
        data.update(assignments=valid,invalid_items=invalid)
        return data


POLICY = '''Identify physical individuals, not semantic categories or all members of a species.
Use visible appearance plus specific observed actions/interactions. Clothing, lighting,
pose, apparent size and hidden faces can change. Missing detail is not contradictory
evidence. Do not invent a story, movie knowledge, names or a preset number of entities.
Co-visible distinct individuals remain distinct even when they touch or interact;
interaction alone never proves identity. Hands, clothing,
shadows and scenery are not independent entities. Small/dark subjects are UNKNOWN,
not automatically false detections. Detector classifications are fallible.
Person versus non-person identities must stay separate: a wrong detector class may
only be corrected explicitly with a visual reason before assigning that track.
All anchor names and summaries are fallible hypotheses; the actual target images
take priority. Never group all people/animals/equipment into one identity.
'''


def _person(kind):
    return kind == 'person'


def _kind(entry):
    # Group kind is the validated classification; raw observation kinds remain
    # unchanged to preserve detector evidence after an explicit correction.
    return entry.get('kind', 'object')


def _effective_entry(entry, kind=None):
    result = dict(entry)
    result['kind'] = kind or _kind(entry)
    result.pop('entity_kind', None)
    return result


def _references(entry, output, limit=2):
    observed = entry['observations']
    ranked = sorted(observed, key=lambda o: (-representative_score(o, output), o['detection_id']))
    selected = []
    for obs in ranked:
        if not selected or obs['shot_id'] not in {s['shot_id'] for s in selected}:
            selected.append(obs)
        if len(selected) >= limit:
            return selected
    for obs in ranked:
        if obs not in selected:
            selected.append(obs)
        if len(selected) >= limit:
            break
    return selected


def render_gallery(entries, output, dest, prefix, *, anchors=False):
    """Small ID-only current galleries; no detector name/COCO category leakage."""
    images = []
    metadata = []
    keys = list(entries)
    for offset in range(0, len(keys), 8):
        page = keys[offset:offset+8]
        sheet = Image.new('RGB', (800, 218*math.ceil(len(page)/4)), '#20252c')
        draw = ImageDraw.Draw(sheet)
        for index, key in enumerate(page):
            entry = entries[key]
            refs = _references(entry, output, limit=2 if anchors else 1)
            x, y = index % 4*200, index//4*218
            draw.text((x+5, y+3), key, fill='white', font=font(16))
            for j, obs in enumerate(refs):
                with Image.open(output/obs['crop']) as source:
                    size = (94, 163) if len(refs)>1 else (190, 163)
                    crop = ImageOps.contain(source.convert('RGB'), size)
                left = x+4+j*98 if len(refs)>1 else x+(200-crop.width)//2
                sheet.paste(crop, (left, y+28+(163-crop.height)//2))
            label = entry.get('label', '') if anchors else '관측 대상'
            draw.text((x+5, y+194), label[:13], fill='white', font=font(14))
            row = dict(id=key, shots=sorted({o['shot_id'] for o in entry['observations']}),
                       reference_frame_keys=[o['frame_key'] for o in refs])
            if anchors:
                row.update(label=entry['label'], kind=_kind(entry), appearance=entry['appearance'])
            metadata.append(row)
        path = dest/f'{prefix}_{offset//8:03d}.jpg'
        sheet.save(path, quality=88)
        images.append((f'{prefix}: {", ".join(page)}', sheet))
    return images, metadata


def select_overview(groups, frames, output, limit=MAX_OVERVIEW_TRACKS):
    """Cover the whole timeline first, then add shot/kind diversity within budget."""
    if limit <= 0:
        return {}
    shot_order = {}
    for frame in frames:
        shot_order.setdefault(frame['shot_id'], len(shot_order))
    by_shot = defaultdict(list)
    for key, entry in groups.items():
        if entry['observations']:
            best = max(representative_score(o, output) for o in entry['observations'])
            by_shot[entry['shot_id']].append((best, key))
    for values in by_shot.values():
        values.sort(key=lambda pair: (-pair[0], pair[1]))
    order = sorted(by_shot, key=lambda shot: (shot_order.get(shot, len(shot_order)), shot))
    if len(order) > limit:
        # A long video still supplies representatives from its beginning and
        # ending. Kind-rich early shots cannot consume the overview budget.
        indices = ([len(order)//2] if limit == 1 else
                   [round(i*(len(order)-1)/(limit-1)) for i in range(limit)])
        covered_shots = [order[index] for index in indices]
    else:
        covered_shots = order
    selected = [by_shot[shot][0][1] for shot in covered_shots]
    selected_set = set(selected)
    kinds_seen = {shot: {_kind(groups[by_shot[shot][0][1]])} for shot in covered_shots}
    while len(selected) < limit:
        changed = False
        for shot in covered_shots:
            remaining = [key for _, key in by_shot[shot] if key not in selected_set]
            if not remaining:
                continue
            novel = [key for key in remaining if _kind(groups[key]) not in kinds_seen[shot]]
            key = (novel or remaining)[0]
            selected.append(key)
            selected_set.add(key)
            kinds_seen[shot].add(_kind(groups[key]))
            changed = True
            if len(selected) >= limit:
                break
        if not changed:
            break
    return {key: groups[key] for key in selected}


def bounded_scene_frames(frames, shots, per_shot, max_frames):
    """Chronological coverage of any video length, with an explicit image cap."""
    candidates = compact_frames(frames, shots, per_shot)
    if len(candidates) > max_frames:
        indices = ([len(candidates)//2] if max_frames == 1 else
                   [round(i*(len(candidates)-1)/(max_frames-1)) for i in range(max_frames)])
        selected = [candidates[index] for index in indices]
    else:
        selected = candidates
    requested = set(shots)
    available = [frame for frame in frames if frame['shot_id'] in requested]
    available_shots = {frame['shot_id'] for frame in available}
    covered_shots = {frame['shot_id'] for frame in selected}
    coverage = dict(available_frames=len(available), sampled_candidates=len(candidates),
        selected_frames=len(selected), omitted_frames=len(available)-len(selected),
        omitted_candidates=len(candidates)-len(selected), max_scene_frames=max_frames,
        per_shot_target=per_shot, available_shots=len(available_shots),
        covered_shots=len(covered_shots), omitted_shots=len(available_shots-covered_shots),
        selected_frame_keys=[frame['frame_key'] for frame in selected])
    return selected, coverage


def request_stage(backend, prompt, schema, selected_frames, current, anchors,
                  output, dest, fps, *, final=False, current_are_identities=False):
    dest.mkdir(parents=True, exist_ok=True)
    images, targets = render_gallery(current, output, dest, 'current',
                                     anchors=current_are_identities)
    refs, anchor_meta = render_gallery(anchors, output, dest, 'anchors', anchors=True)
    inventory = dict(version=VERSION, current=targets, anchors=anchor_meta,
        frames=[dict(frame_key=f['frame_key'], shot_id=f['shot_id'], time=f['full_time_sec'])
                for f in selected_frames])
    request = prompt+'\nINPUT: '+json.dumps(inventory, ensure_ascii=False)
    atomic_json(dest/'input.json', dict(prompt=request, inventory=inventory))
    # Anchor references stay in the gallery, never appear as target boxes in
    # scene frames. This also avoids falsely flagging a current reference as a
    # second simultaneous appearance of itself.
    clip = scene_clip(selected_frames, current, output, fps) if selected_frames else None
    backend.reserve = 0 if final else 1
    for attempt in range(2):
        try:
            result = backend.request(request, schema, clip=clip, images=images+refs,
                                     max_tokens=backend.settings.max_new_tokens, image_pixels=131072)
            atomic_json(dest/'result.json', result.model_dump())
            return result
        except BudgetExhausted as exc:
            atomic_json(dest/f'budget_{attempt+1}.json', dict(error=str(exc)))
        except (ModelOutputError, torch.OutOfMemoryError) as exc:
            atomic_json(dest/f'failure_{attempt+1}.json', dict(error=str(exc)))
            if isinstance(exc, torch.OutOfMemoryError):
                gc.collect()
                torch.cuda.empty_cache()
        # Both prompt cache keys are tried even when the generation ledger is
        # exhausted, so an earlier successful corrected answer can be replayed.
        request += '\nReturn concise valid JSON following the schema exactly.'
    return None


def _same_frame_duplicate(left, right, kind, settings):
    try:
        from duplicate_tracks import duplicate_evidence
        # Kind corrections apply to the comparison, never to stored detector
        # records. This is identical to the post-assignment export resolver.
        return duplicate_evidence(dict(left, entity_kind=kind),
                                  dict(right, entity_kind=kind), settings)
    except ImportError:
        return None


def make_anchors(result, selected, settings=None, *, config=None):
    config = MatchingConfig() if config is None else config
    anchors = {}
    corrections = {}
    rejected = []
    owned_references = set()
    if result is None:
        return anchors, corrections, rejected
    for correction in result.corrections:
        if correction.track_id in selected and correction.reason.strip():
            corrections[correction.track_id] = correction.model_dump()
    for spec in result.anchors:
        refs = spec.reference_track_ids
        if (spec.anchor_id in anchors or len(set(refs)) != len(refs)
                or not set(refs) <= set(selected)):
            rejected.append(dict(anchor_id=spec.anchor_id, reason='invalid reference IDs'))
            continue
        if owned_references & set(refs):
            rejected.append(dict(anchor_id=spec.anchor_id, reason='reference already owns a different anchor'))
            continue
        kinds = [corrections.get(key, {}).get('kind', _kind(selected[key])) for key in refs]
        if any(_person(kind) != _person(spec.kind) for kind in kinds):
            rejected.append(dict(anchor_id=spec.anchor_id, reason='reference kind needs explicit correction'))
            continue
        reference_conflict = None
        for index, left in enumerate(refs):
            for right in refs[index+1:]:
                a, b = selected[left]['observations'], selected[right]['observations']
                if any(x['frame_key'] == y['frame_key']
                       and _same_frame_duplicate(x, y, spec.kind, settings) is None for x in a for y in b):
                    reference_conflict = 'distinct simultaneous reference subjects'
                if _person(spec.kind):
                    proof = appearance_evidence(a, b, person=True).get('reliable_face')
                    if proof and proof['best'] < config.clear_face_contradiction:
                        reference_conflict = 'clear face contradiction between references'
        if reference_conflict:
            rejected.append(dict(anchor_id=spec.anchor_id, reason=reference_conflict))
            continue
        anchors[spec.anchor_id] = dict(label=spec.label, kind=spec.kind, appearance=spec.appearance,
            reference_track_ids=list(refs), evidence=spec.evidence,
            # These observations exist ONLY inside fixed reference evidence.
            observations=[deepcopy(o) for key in refs for o in selected[key]['observations']])
        owned_references.update(refs)
    return anchors, corrections, rejected


def _retrieval_anchor(anchor):
    """Reference frame presence is not enrolled membership or a cannot-link."""
    result = dict(anchor)
    result['observations'] = [dict(o, frame_key='anchor-reference:'+o['frame_key'])
                              for o in anchor['observations']]
    return result


def retrieve(entry, anchors, kind=None, *, config=None):
    config = config or MatchingConfig()
    proxy = {key: _retrieval_anchor(anchor) for key, anchor in anchors.items()}
    return rank_candidates(_effective_entry(entry, kind), proxy, limit=config.candidates)


def identity_support(entry, anchor, kind, *, final=False, config=None):
    config = config or MatchingConfig()
    if _person(kind) != _person(_kind(anchor)):
        return False, dict(reason='kind conflict')
    evidence = appearance_evidence(entry['observations'], anchor['observations'], person=_person(kind))
    reliable = evidence.get('reliable_face')
    if reliable and reliable['best'] < config.clear_face_contradiction:
        return False, dict(reason='clear face contradiction', similarities=evidence)
    def score(key):
        return evidence[key]['score'] if evidence.get(key) else float('-inf')
    def threshold(key):
        return getattr(config, ('final_' if final else 'batch_')+key)
    if _person(kind):
        supported = (score('face') >= threshold('face')
            or score('body') >= threshold('body')
            or score('visual') >= threshold('visual')
            or (score('visual') >= threshold('joint_visual')
                and score('body') >= threshold('joint_body')))
    else:
        supported = score('object') >= threshold('object')
    return bool(supported), dict(reason='positive appearance support' if supported else 'low appearance support',
                                 similarities=evidence)


def _collision(track_id, target, groups, mapping, settings, kind):
    by_frame = defaultdict(list)
    for other, assigned in mapping.items():
        if other != track_id and assigned == target:
            for obs in groups[other]['observations']:
                by_frame[obs['frame_key']].append(obs)
    for obs in groups[track_id]['observations']:
        for other in by_frame[obs['frame_key']]:
            duplicate = _same_frame_duplicate(obs, other, kind, settings)
            if duplicate is None:
                return True
    return False


def apply_assignments(result, current, anchors, groups, mapping, states,
                      corrected_kinds, settings, *, final=False, config=None,
                      preserve_existing=False, allowed_targets=None):
    """Validate each assignment independently; immutable anchors never accumulate crops."""
    decisions = {}
    duplicates = set()
    allowed=set(anchors) if allowed_targets is None else set(allowed_targets)&set(anchors)
    if result:
        for decision in result.assignments:
            if decision.track_id not in current:
                continue
            if decision.track_id in decisions:
                duplicates.add(decision.track_id)
            decisions[decision.track_id] = decision
    for key, entry in current.items():
        previous_state=deepcopy(states.get(key,{}))
        decision = decisions.get(key)
        previous_kind = corrected_kinds.get(key, {}).get('kind', _kind(entry))
        valid = (decision is not None and key not in duplicates
                 and bool(decision.reason.strip())
                 and any(s.strip() for s in decision.evidence))
        if valid:
            valid=((decision.decision=='match' and decision.anchor_id in allowed)
                   or (decision.decision=='unknown' and decision.anchor_id is None)
                   or (decision.decision=='reject' and decision.anchor_id is None
                       and len({s.strip() for s in decision.evidence if s.strip()})>=2))
        correction = (decision.correction if valid and decision.correction
                      and decision.correction.reason.strip() else None)
        kind = correction.kind if correction else previous_kind
        state = dict(status='unknown', reason='No valid supported assignment', kind=kind,
                     review_attempted=decision is not None,review_valid=valid,
                     review_decision=decision.decision if valid else None)
        previous = mapping.pop(key, None)
        if valid:
            state.update(reason=decision.reason, proposed_anchor=decision.anchor_id,
                         evidence=decision.evidence)
            if (decision.decision == 'reject' and decision.anchor_id is None
                    and len({s.strip() for s in decision.evidence if s.strip()}) >= 2):
                state['status'] = 'rejected'
            elif decision.decision == 'match' and decision.anchor_id in anchors and any(s.strip() for s in decision.evidence):
                target = decision.anchor_id
                supported, proof = identity_support(entry, anchors[target], kind, final=final, config=config)
                state['appearance_check'] = proof
                if supported and not _collision(key, target, groups, mapping, settings, anchors[target]['kind']):
                    mapping[key] = target
                    state['status'] = 'matched'
                else:
                    state['status'] = 'low_support' if not supported else 'co_visible_conflict'
        if state['status'] not in {'matched', 'rejected'}:
            # A rejected/ambiguous proposal must not change the kind used to
            # preserve the prior association. Commit corrections atomically
            # with a supported decision, never merely because JSON parsed.
            if correction:state['unapplied_correction']=correction.model_dump()
            kind=previous_kind;state['kind']=kind
            if preserve_existing and previous in anchors and _person(kind)==_person(anchors[previous]['kind']):
                mapping[key]=previous
                state['review_status']=state['status']
                state['status']=previous_state.get('status','visual_estimated')
                state['baseline_preserved']=True
                state['review_reason']=state['reason']
                state['reason']=previous_state.get('reason','Previous visual association preserved')
            else:mapping[key] = 'U_'+key
        elif correction:
            corrected_kinds[key]=correction.model_dump()
        state['previous_assignment'] = previous
        states[key] = state


def prune_anchor_references(anchors, groups, mapping, states):
    """Keep reference observations only while their actual track belongs here."""
    active=defaultdict(list)
    for track,target in mapping.items():
        if track in groups and states.get(track,{}).get('status')!='rejected':
            active[target].append(track)
    audit=dict(removed_anchors=[],pruned_references=[])
    for key,entry in list(anchors.items()):
        members=sorted(active.get(key,[]))
        if not members:
            anchors.pop(key);audit['removed_anchors'].append(key);continue
        owner={obs['detection_id']:track for track in members
               for obs in groups[track]['observations']}
        retained=[obs for obs in entry['observations'] if obs['detection_id'] in owner]
        removed=len(entry['observations'])-len(retained)
        if not retained:
            # A valid reassignment may leave an identity with new members only.
            # Use one accepted member rather than a rejected/retired reference.
            reference=max(members,key=lambda t:(len(groups[t]['observations']),t))
            retained=deepcopy(groups[reference]['observations'])
        entry['observations']=retained
        entry['track_ids']=members
        entry['reference_track_ids']=sorted({owner[obs['detection_id']] for obs in retained})
        entry['support_members']=[track for track in entry.get('support_members',[])
                                  if track in members]
        entry['shot_id']=min(retained,key=lambda obs:obs['full_time_sec'])['shot_id']
        if removed:audit['pruned_references'].append(dict(id=key,removed_observations=removed))
    return audit


def _anchor_metadata(anchors):
    return {key: {k: v for k, v in anchor.items() if k != 'observations'}
            for key, anchor in anchors.items()}


def finalize_memory(groups, anchors, mapping, states, corrected_kinds):
    """Only assigned observations become catalog presence; references add none."""
    memory = {}
    joins = []
    for key, target in mapping.items():
        entry = groups[key]
        if target in anchors:
            source = anchors[target]
        else:
            kind = corrected_kinds.get(key, {}).get('kind', _kind(entry))
            source = dict(label={'person': '미확인 인물', 'animal': '미확인 동물', 'object': '미확인 대상'}[kind],
                          kind=kind, appearance='확실한 동일 개체 연결을 확인하지 못함')
        if target not in memory:
            memory[target] = dict(label=source['label'], kind=source['kind'], appearance=source['appearance'],
                observations=[], members=[], history=[],
                identity_status='unresolved',
                reference_key=target if target in anchors else None)
        memory[target]['observations'].extend(deepcopy(entry['observations']))
        memory[target]['members'].append(key)
        memory[target]['history'].append(dict(track_id=key, **states.get(key, {})))
        state=states.get(key,{}).get('status')
        if state in {'matched','catalog_reviewed'}:memory[target]['identity_status']='estimated'
        elif state=='visual_estimated' and memory[target]['identity_status']!='estimated':
            memory[target]['identity_status']='visual_estimated'
        if target in anchors and len(memory[target]['members']) > 1:
            joins.append(dict(target=target, retired=key, method='fixed anchor assignment',
                              reason=states.get(key, {}).get('reason', '')))
    return memory, mapping, joins


ASSIGN_PROMPT = '''Assign EACH CURRENT track ID independently to one fixed canonical anchor,
or unknown, or reject only a clearly demonstrated false detection/body part.
Anchor reference images show examples of identity, not extra simultaneous objects.
Never return category-wide groups. Never modify/merge anchors or invent IDs.
Use positive visible traits and/or specific observed action continuity and explain
them in evidence. A shared label, species, outfit, or "main character" is insufficient.
If detector kind is wrong, set correction={kind,reason} explicitly; otherwise null.
Reject requires two distinct concrete visual observations in evidence.
Return every current ID once. Unknown/reject requires anchor_id=null.
ONLY JSON {"scene_summary":"observed sequence", "assignments":[{"track_id":"T_ID",
"decision":"match|unknown|reject", "anchor_id":null, "correction":null,
"reason":"specific comparison", "evidence":["visible evidence"]}]}.
'''


CATALOG_PROMPT = POLICY+"""Review ONLY the supplied known identity IDs. Their reference membership was
computed from observations; do not invent IDs or change reference membership.
Keep each descriptive field and evidence item to one short sentence.
Give each visible identity a concise Korean label, kind and stable appearance.
Propose a SAME relation only between two IDs that depict one physical individual,
using visual traits plus observed continuity. A shared category/name is insufficient.
It is valid to leave uncertain relations unresolved. Distinct co-visible subjects
remain separate. For a clear false detection/body part use decision=reject only
with two distinct concrete visual observations in evidence.
ONLY JSON {"scene_summary":"observed sequence",
"annotations":[{"id":"known ID","label":"짧은 이름","kind":"person|animal|object",
"appearance":"stable visible traits","decision":"keep|reject","reason":"visible grounds",
"evidence":["concrete observation"]}],
"relations":[{"left_id":"known ID","right_id":"known ID",
"relation":"same|different|uncertain","reason":"specific grounds",
"evidence":["identity evidence"]}]}.
"""


def match_anchored(groups, frames, timeline, output, settings, limit=8, batch_shots=4,
                   *, config=None, visual_config=None):
    from catalog_review import CatalogDecision, apply_catalog
    from visual_association import build_visual_identities
    config = MatchingConfig() if config is None else config
    if not isinstance(config, MatchingConfig):raise TypeError('config must be MatchingConfig')
    if type(batch_shots) is not int or batch_shots < 1:raise ValueError('batch_shots must be positive')
    if type(limit) is not int or not 1 <= limit <= 8:raise ValueError('Qwen limit must be in 1..8')
    output=Path(output);root=output/'anchored';root.mkdir(parents=True,exist_ok=True)
    backend=BudgetBackend(settings,output/'review_traces',limit)
    fps=float(Fraction(timeline['fps']));shots=[s['id'] for s in timeline['shots']]
    anchors,mapping,baseline=build_visual_identities(groups,settings,visual_config)
    atomic_json(root/'visual_association.json',baseline)
    atomic_json(root/'effective_config.json',asdict(config))
    states={};corrected_kinds={};history=[];failures=[];reviews=[];successes=0
    for identity,entry in anchors.items():
        entry['shot_id']=entry['observations'][0]['shot_id']
        entry['label']={'person':'인물','animal':'동물','object':'대상'}[entry['kind']]
        entry['appearance']='외형 근거로 구성한 관측 묶음; 의미 검토 전'
        for track in entry['track_ids']:
            states[track]=dict(status='visual_estimated' if entry['support_status']=='linked' else 'unknown',
                kind=entry['kind'],reason='Appearance association before Qwen',baseline_identity=identity)
    atomic_json(root/'visual_seeds.json',_anchor_metadata(anchors))
    coverage=dict(batches=[])

    def catalog_phase(phase,frames_for_review,frame_coverage,max_identities,*,final=False):
        nonlocal successes
        active={key:entry for key,entry in anchors.items() if key in set(mapping.values())}
        selected=select_overview(active,frames,output,limit=max_identities)
        coverage[phase]=dict(scene=frame_coverage,selected_identity_ids=list(selected),
            omitted_identity_ids=[key for key in active if key not in selected])
        if not selected:return None
        backend.phase=phase
        prompt=CATALOG_PROMPT+('\nThis is the ONE final catalog relationship review. Reconsider fragments across shots.' if final else '')
        prompt+='\n'+json.dumps(dict(known_ids=list(selected),scene_coverage=frame_coverage,
            shot_history=history,previous_aliases=reviews[-1].get('aliases',{}) if final and reviews else {}),ensure_ascii=False)
        result=request_stage(backend,prompt,CatalogDecision,frames_for_review,selected,{},
            output,root/phase,fps,final=final,current_are_identities=True)
        if result is None:
            failures.append(dict(phase=phase,reason='No valid response; appearance membership preserved'))
            return None
        for row in result.annotations:
            if row.id not in selected:result.invalid_items.append(dict(reason='Annotation ID was not shown',id=row.id))
        for row in result.relations:
            if row.left_id not in selected or row.right_id not in selected:
                result.invalid_items.append(dict(reason='Relation references an ID not shown',left=row.left_id,right=row.right_id))
        result.annotations=[row for row in result.annotations if row.id in selected]
        result.relations=[row for row in result.relations if row.left_id in selected and row.right_id in selected]
        audit=apply_catalog(result,anchors,groups,mapping,states,corrected_kinds,settings,config,phase)
        audit['reference_cleanup']=prune_anchor_references(anchors,groups,mapping,states)
        reviews.append(audit);atomic_json(root/phase/'catalog_audit.json',audit)
        if audit['annotations'] or audit['rejected'] or audit['relations']:successes+=1
        else:failures.append(dict(phase=phase,reason='No grounded catalog decision; baseline preserved'))
        if result.invalid_items:
            failures.append(dict(phase=phase,reason='Malformed catalog items skipped',items=result.invalid_items))
        return result

    overview_frames,overview_coverage=bounded_scene_frames(frames,shots,config.overview_scene_frames,config.max_scene_frames)
    overview=catalog_phase('overview',overview_frames,overview_coverage,config.overview_tracks)
    # A missing catalog response cannot disable assignment: visual seeds always
    # exist for observed tracks and remain available within the same budget.
    width=max(batch_shots,math.ceil(len(shots)/config.max_batches)) if shots else batch_shots
    for index,start in enumerate(range(0,len(shots),width),1):
        selected_shots=shots[start:start+width]
        available={key:entry for key,entry in groups.items()
                   if entry['shot_id'] in selected_shots and states.get(key,{}).get('status')!='rejected'}
        if not available:
            history.append(dict(shots=selected_shots,summary='No retained observations.'));continue
        current=select_overview(available,frames,output,limit=config.max_batch_tracks)
        selected_frames,frame_coverage=bounded_scene_frames(frames,selected_shots,config.batch_scene_frames,config.max_scene_frames)
        coverage['batches'].append(dict(batch=index,shots=selected_shots,scene=frame_coverage,
            selected_track_ids=list(current),omitted_track_ids=[k for k in available if k not in current]))
        candidates={key:retrieve(entry,anchors,corrected_kinds.get(key,{}).get('kind'),config=config)
                    for key,entry in current.items()}
        visible=set(anchors) if len(anchors)<=config.all_anchors_limit else {
            row['entity_id'] for rows in candidates.values() for row in rows}
        # Always include the existing appearance assignment, even if retrieval
        # ranks another view higher. The model may preserve or challenge it.
        visible.update(mapping[key] for key in current if key in mapping)
        references={key:anchors[key] for key in anchors if key in visible}
        prompt=POLICY+ASSIGN_PROMPT+json.dumps(dict(current_ids=list(current),candidates=candidates,
            current_assignments={key:mapping.get(key) for key in current},scene_coverage=frame_coverage,
            detector_kinds={key:corrected_kinds.get(key,{}).get('kind',_kind(entry)) for key,entry in current.items()},
            prior_summaries=history,overview=overview.scene_summary if overview else ''),ensure_ascii=False)
        backend.phase=f'batch_{index}'
        dest=root/f'batch_{index:02d}'
        decision=request_stage(backend,prompt,AssignmentDecision,selected_frames,current,references,
            output,dest,fps)
        if decision is None:failures.append(dict(phase=backend.phase,reason='No valid response; previous assignments preserved'))
        else:
            if decision.invalid_items:failures.append(dict(phase=backend.phase,reason='Malformed assignments skipped',items=decision.invalid_items))
        apply_assignments(decision,current,anchors,groups,mapping,states,corrected_kinds,settings,
            config=config,preserve_existing=True,allowed_targets=references)
        if decision is not None:
            if any(states[key].get('review_valid') for key in current):successes+=1
            else:failures.append(dict(phase=backend.phase,reason='No valid grounded assignment rows; baseline preserved'))
        atomic_json(dest/'reference_cleanup.json',prune_anchor_references(anchors,groups,mapping,states))
        history.append(dict(shots=selected_shots,summary=decision.scene_summary if decision else 'Appearance assignments retained; review unavailable.'))
        atomic_json(dest/'assignments.json',{key:states[key] for key in available})

    final_frames,final_coverage=bounded_scene_frames(frames,shots,config.final_scene_frames,config.max_scene_frames)
    catalog_phase('final',final_frames,final_coverage,config.final_tracks,final=True)
    memory,mapping,joins=finalize_memory(groups,anchors,mapping,states,corrected_kinds)
    unresolved=sum(e['identity_status']=='unresolved' for e in memory.values())
    unreviewed=[key for key,state in states.items()
                if not state.get('review_valid') and state.get('status') not in {'matched','rejected'}]
    coverage['unreviewed_track_ids']=unreviewed
    warnings=[]
    if failures:warnings.append(f'Qwen 검토 {len(failures)}개 단계에서 응답 실패 또는 일부 항목 오류가 있었습니다. 기본 외형 연결은 보존했습니다.')
    if groups and not successes:warnings.append('Qwen 의미 검토가 성공하지 않았습니다. 외형 기반 결과입니다.')
    if unresolved:warnings.append(f'동일성이 미확정인 관측 묶음이 {unresolved}개 남았습니다. 서로 다른 개체임을 뜻하지 않습니다.')
    if unreviewed:warnings.append(f'{len(unreviewed)}개 트랙은 유효한 Qwen 동일성 검토를 받지 못했습니다. 기존 외형 추정 상태로 보존했습니다.')
    quality=dict(status='complete_with_warnings' if warnings else 'complete',warnings=warnings,
        successful_qwen_requests=successes,failed_stages=failures,baseline_preserved_on_failure=True,
        unreviewed_tracklets=len(unreviewed),
        unresolved_identity_groups=unresolved,resolved_identity_groups=len(memory)-unresolved)
    atomic_json(root/'quality.json',quality)
    atomic_json(root/'coverage.json',coverage)
    atomic_json(root/'anchors.json',_anchor_metadata(anchors))
    atomic_json(root/'assignments.json',mapping)
    atomic_json(root/'assignment_evidence.json',states)
    atomic_json(root/'kind_corrections.json',corrected_kinds)
    atomic_json(root/'summary.json',dict(version=VERSION,config=asdict(config),tracklets=len(groups),
        canonical_anchors=len(anchors),final_entities=len(memory),
        matched_tracklets=sum(e['status'] in {'matched','catalog_reviewed'} for e in states.values()),
        visual_linked_tracklets=sum(e['status']=='visual_estimated' for e in states.values()),
        unknown_tracklets=sum(e['status']=='unknown' for e in states.values()),
        actual_qwen_generations=backend.ledger['generations'],qwen_limit=limit,
        fixed_batch_shots=width,max_batches=config.max_batches,history=history,quality=quality,
        no_pairwise_model_loop=True,reference_membership_from_observations=True))
    if backend.model is not None:
        del backend.model;gc.collect();torch.cuda.empty_cache()
    return memory,mapping,joins
