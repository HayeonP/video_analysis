"""Bounded semantic review of identities already supported by visual evidence.

The model may annotate supplied IDs and compare explicit ID pairs. It cannot
create identities, replace reference images, or return category-wide groups.
Malformed items are isolated during parsing so an unrelated valid decision is
not discarded. This module performs no model calls and preserves raw groups.
"""
from collections import defaultdict
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


VERSION = 1


class _EvidenceRow(BaseModel):
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)

    @field_validator('evidence', mode='before')
    @classmethod
    def accept_single_evidence(cls, value):
        # A single textual evidence item is an unambiguous wire-format repair.
        return [value] if isinstance(value, str) else value

    @field_validator('reason')
    @classmethod
    def nonblank_reason(cls, value):
        if not value.strip():
            raise ValueError('reason must contain text')
        return value.strip()


class CatalogAnnotation(_EvidenceRow):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: Literal['person', 'animal', 'object']
    appearance: str = Field(min_length=1)
    decision: Literal['keep', 'reject'] = 'keep'

    @field_validator('id', 'label', 'appearance')
    @classmethod
    def nonblank_fields(cls, value):
        if not value.strip():
            raise ValueError('ID, label and appearance must contain text')
        return value.strip()


class CatalogRelation(_EvidenceRow):
    left_id: str = Field(min_length=1)
    right_id: str = Field(min_length=1)
    relation: Literal['same', 'different', 'uncertain']

    @field_validator('left_id', 'right_id')
    @classmethod
    def nonblank_ids(cls, value):
        if not value.strip():
            raise ValueError('ID must contain text')
        return value.strip()


class CatalogDecision(BaseModel):
    scene_summary: str = ''
    annotations: list[CatalogAnnotation] = Field(default_factory=list)
    relations: list[CatalogRelation] = Field(default_factory=list)
    invalid_items: list[dict] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def salvage_independent_items(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        previous = data.get('invalid_items', [])
        invalid = [deepcopy(item) for item in previous if isinstance(item, dict)] if isinstance(previous, list) else []
        if not isinstance(data.get('scene_summary', ''), str):
            invalid.append(dict(section='scene_summary', reason='Expected text', value=data['scene_summary']))
            data['scene_summary'] = ''
        for section, schema in (('annotations', CatalogAnnotation), ('relations', CatalogRelation)):
            rows = data.get(section, [])
            if not isinstance(rows, list):
                invalid.append(dict(section=section, reason='Expected a list', value=rows))
                rows = []
            accepted = []
            for index, row in enumerate(rows):
                try:
                    item = row if isinstance(row, schema) else schema.model_validate(row)
                    accepted.append(item)
                except (ValidationError, TypeError, ValueError) as exc:
                    invalid.append(dict(section=section, index=index, reason=str(exc), value=deepcopy(row)))
            data[section] = accepted
        data['invalid_items'] = invalid
        return data


def _evidence(row):
    return list(dict.fromkeys(text.strip() for text in row.evidence if text.strip()))


def _grounded(row, *, rejection=False):
    # Rejection removes an observed subject from the catalog. Require two
    # distinct stated observations and an explicit reason for that operation;
    # an identity mismatch or uncertain appearance alone is not a rejection.
    return bool(row.reason.strip()) and len(_evidence(row)) >= (2 if rejection else 1)


def _strings(value):
    return [item for item in value if isinstance(item, str)] if isinstance(value, (list, tuple, set)) else []


def _track_ids(anchor_id, anchors, groups, mapping):
    actual = [track for track, assigned in mapping.items() if assigned == anchor_id and track in groups]
    entry = anchors[anchor_id]
    references = _strings(entry.get('reference_track_ids', []))
    support = _strings(entry.get('support_members', []))
    return sorted(set(actual + references + support) & set(groups))


def _co_visible_conflicts(left_id, right_id, groups, mapping, kind, settings):
    """Compare real membership only; reference galleries never add sightings."""
    from duplicate_tracks import duplicate_evidence

    left_frames = defaultdict(list)
    for track, assigned in mapping.items():
        if assigned != left_id or track not in groups:
            continue
        for obs in groups[track].get('observations', []):
            if obs.get('frame_key'):
                left_frames[obs['frame_key']].append((track, obs))
    conflicts, duplicates = [], []
    for track, assigned in mapping.items():
        if assigned != right_id or track not in groups:
            continue
        for obs in groups[track].get('observations', []):
            for other_track, other in left_frames.get(obs.get('frame_key'), []):
                if (obs.get('detection_id') is not None
                        and obs.get('detection_id') == other.get('detection_id')):
                    # The same detection cited twice is not two physical people.
                    continue
                evidence = duplicate_evidence(dict(other, entity_kind=kind),
                                              dict(obs, entity_kind=kind), settings)
                row = dict(frame_key=obs['frame_key'], left_track=other_track,
                           right_track=track, left_detection=other.get('detection_id'),
                           right_detection=obs.get('detection_id'))
                if evidence is None:
                    conflicts.append(row)
                else:
                    duplicates.append(dict(row, evidence=evidence))
    return conflicts, duplicates


def _combined_observations(*entries):
    observed, seen = [], set()
    for entry in entries:
        for obs in entry.get('observations', []):
            key = ('detection', obs['detection_id']) if obs.get('detection_id') else (
                'observation', obs.get('frame_key'), tuple(obs.get('bbox', [])), obs.get('crop'))
            if key not in seen:
                seen.add(key)
                observed.append(deepcopy(obs))
    return observed


def apply_catalog(decision, anchors, groups, mapping, states, corrected_kinds,
                  settings, matching_config, phase):
    """Apply validated annotations/pair relations in place and return an audit.

    ``groups`` and its raw observations remain untouched. Merged anchors own
    copied evidence. ``mapping`` alone determines exported track membership.
    Unknown or malformed model output retains the preexisting visual baseline.
    """
    if decision is not None and not isinstance(decision, CatalogDecision):
        raise TypeError('decision must be a CatalogDecision or None')
    audit = dict(version=VERSION, phase=phase, status='unreviewed' if decision is None else 'reviewed',
                 annotations=[], rejected=[], relations=[], merges=[], blocked=[],
                 invalid_items=deepcopy(decision.invalid_items) if decision is not None else [])
    if decision is None:
        return audit
    # Local import avoids a cycle when the matcher imports this review module.
    from anchored_matching import identity_support

    known = set(anchors)
    alias = {key: key for key in known}
    components = {key: {key} for key in known}
    annotated = set()
    annotation_rows = defaultdict(list)
    for row in decision.annotations:
        annotation_rows[row.id].append(row)
    for key, rows in annotation_rows.items():
        unique = {row.model_dump_json(): row for row in rows}
        if key not in known or len(unique) != 1:
            audit['blocked'].append(dict(operation='annotation', id=key,
                reason='unknown ID' if key not in known else 'conflicting duplicate annotations'))
            continue
        row = next(iter(unique.values()))
        if not _grounded(row, rejection=row.decision == 'reject'):
            audit['blocked'].append(dict(operation='annotation', id=key,
                reason='rejection needs explicit reason and two distinct evidence items'
                       if row.decision == 'reject' else 'missing positive evidence'))
            continue
        actual_tracks = sorted(track for track, assigned in mapping.items() if assigned == key)
        if row.decision == 'reject':
            for track in actual_tracks:
                previous = mapping.pop(track)
                states.setdefault(track, {}).update(status='rejected', reason=row.reason.strip(),
                    evidence=_evidence(row), review_phase=phase, previous_assignment=previous)
            anchors.pop(key)
            alias[key] = None
            audit['rejected'].append(dict(id=key, tracks=actual_tracks,
                reason=row.reason.strip(), evidence=_evidence(row)))
            continue
        entry = deepcopy(anchors[key])
        previous_kind = entry.get('kind', 'object')
        entry.update(label=row.label.strip(), kind=row.kind, appearance=row.appearance.strip(),
                     semantic_review=dict(phase=phase, reason=row.reason.strip(), evidence=_evidence(row)))
        anchors[key] = entry
        if previous_kind != row.kind:
            for track in _track_ids(key, anchors, groups, mapping):
                corrected_kinds[track] = dict(kind=row.kind, reason=row.reason.strip(),
                    evidence=_evidence(row), phase=phase, anchor_id=key)
        for track in actual_tracks:
            states.setdefault(track, {}).update(kind=row.kind, annotation_anchor=key, annotation_phase=phase)
        annotated.add(key)
        audit['annotations'].append(dict(id=key, label=row.label.strip(), kind=row.kind,
            previous_kind=previous_kind, reason=row.reason.strip(), evidence=_evidence(row)))

    relation_rows = defaultdict(list)
    for row in decision.relations:
        relation_rows[tuple(sorted((row.left_id, row.right_id)))].append(row)
    accepted = []
    for pair, rows in sorted(relation_rows.items()):
        if len(set(pair)) != 2 or not set(pair) <= known:
            audit['blocked'].append(dict(operation='relation', ids=list(pair), reason='invalid or unknown pair IDs'))
            continue
        if len({row.relation for row in rows}) != 1:
            audit['blocked'].append(dict(operation='relation', ids=list(pair), reason='contradictory pair relations'))
            continue
        grounded = [row for row in rows if _grounded(row)]
        if not grounded:
            audit['blocked'].append(dict(operation='relation', ids=list(pair), reason='missing positive evidence'))
            continue
        row = grounded[0]
        accepted.append(row)
        audit['relations'].append(row.model_dump())
    # Read all explicit different claims before any merge, including transitive
    # bridges such as A=B, B=C, A!=C in the same response.
    different = [frozenset((row.left_id, row.right_id)) for row in accepted if row.relation == 'different']
    for row in accepted:
        if row.relation != 'same':
            continue
        left, right = alias.get(row.left_id), alias.get(row.right_id)
        details = dict(operation='merge', ids=[row.left_id, row.right_id], reason=row.reason.strip(),
                       evidence=_evidence(row))
        if left is None or right is None:
            audit['blocked'].append(dict(details, gate='rejected anchor'))
            continue
        if left == right:
            audit['relations'].append(dict(ids=[row.left_id, row.right_id], status='already_same', target=left))
            continue
        union = components[left] | components[right]
        if any(pair <= union for pair in different):
            audit['blocked'].append(dict(details, gate='explicit different relation'))
            continue
        target, retired = sorted((left, right))
        first, second = anchors[target], anchors[retired]
        kind = first.get('kind', 'object')
        supported, appearance = identity_support(first, second, kind, final=True, config=matching_config)
        if not supported:
            audit['blocked'].append(dict(details, gate='appearance support', appearance_check=appearance))
            continue
        conflicts, duplicates = _co_visible_conflicts(target, retired, groups, mapping, kind, settings)
        if conflicts:
            audit['blocked'].append(dict(details, gate='distinct co-visible subjects', conflicts=conflicts,
                                         appearance_check=appearance))
            continue
        merged = deepcopy(first)
        if target not in annotated and retired in annotated:
            for field in ('label', 'kind', 'appearance', 'semantic_review'):
                if field in second:
                    merged[field] = deepcopy(second[field])
            annotated.add(target)
        merged['observations'] = _combined_observations(first, second)
        for field in ('reference_track_ids', 'support_members', 'track_ids'):
            merged[field] = sorted(set(_strings(first.get(field, [])) + _strings(second.get(field, []))))
        merged['retired_anchor_ids'] = sorted(set(_strings(first.get('retired_anchor_ids', []))
            + _strings(second.get('retired_anchor_ids', [])) + [retired]))
        merge_record = dict(target=target, retired=retired, phase=phase, reason=row.reason.strip(),
                            evidence=_evidence(row), appearance_check=appearance,
                            duplicate_observations=duplicates)
        merged['catalog_merge_history'] = deepcopy(first.get('catalog_merge_history', [])) + deepcopy(
            second.get('catalog_merge_history', [])) + [deepcopy(merge_record)]
        anchors[target] = merged
        anchors.pop(retired)
        components[target] = union
        components.pop(retired)
        for original in union:
            alias[original] = target
        redirected = []
        for track, assigned in list(mapping.items()):
            if assigned == retired:
                mapping[track] = target
                redirected.append(track)
            if assigned in (target, retired):
                merged_kind = merged.get('kind', kind)
                previous_kind = corrected_kinds.get(track, {}).get('kind', groups.get(track, {}).get('kind', 'object'))
                if previous_kind != merged_kind:
                    corrected_kinds[track] = dict(kind=merged_kind, reason=row.reason.strip(),
                        evidence=_evidence(row), phase=phase, anchor_id=target)
                state = states.setdefault(track, {})
                state.update(status='matched', kind=merged_kind,
                    proposed_anchor=target, catalog_review_phase=phase)
                state.setdefault('catalog_relations', []).append(dict(target=target, retired=retired,
                    reason=row.reason.strip(), evidence=_evidence(row)))
        audit['merges'].append(dict(merge_record, redirected_tracks=sorted(redirected)))
    audit['aliases'] = {key: value for key, value in sorted(alias.items()) if key != value}
    audit['active_anchor_count'] = len(anchors)
    audit['assigned_track_count'] = len(mapping)
    return audit
