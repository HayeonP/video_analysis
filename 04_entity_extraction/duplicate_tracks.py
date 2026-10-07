"""Remove duplicate observed boxes only after an identity has been assigned.

These conservative geometry/appearance thresholds are starting heuristics, not
calibrated probabilities. This module does not infer identity, merge anchors,
interpolate boxes, load models, or modify its inputs. Ordinary co-visible
subjects that lack duplicate evidence remain conflicts for the caller to handle.
"""
from collections import defaultdict
from copy import deepcopy
import math


VERSION = 1
THRESHOLDS = dict(person_iou=.20, person_area_ratio=.45, body_similarity=.80,
                  visual_similarity=.85, face_similarity=.65, face_quality=80.,
                  object_iou=.60, object_area_ratio=.60, object_similarity=.94)


def _box(obs):
    try:
        values = [float(value) for value in obs['bbox']]
    except (KeyError, TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    if values[0] >= values[2] or values[1] >= values[3]:
        return None
    return values


def _area(box):
    return (box[2]-box[0]) * (box[3]-box[1])


def _kind(obs):
    return obs.get('entity_kind', obs.get('kind', 'object'))


def _cosine(left, right, key):
    def value(obs):
        result = obs.get(key)
        # Both fields contain the same normalized DINO encoder output. A
        # semantic kind correction changes the expected field, not its space.
        alternate = {'visual_feature': 'object_feature', 'object_feature': 'visual_feature'}.get(key)
        if result is None and alternate:
            result = obs.get(alternate)
        return result
    try:
        a, b = [float(x) for x in value(left)], [float(x) for x in value(right)]
    except (KeyError, TypeError, ValueError):
        return None
    if not a or len(a) != len(b) or not all(math.isfinite(x) for x in a+b):
        return None
    na, nb = math.hypot(*a), math.hypot(*b)
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return max(-1., min(1., math.fsum(x/na * y/nb for x, y in zip(a, b))))


def _number(value, default=0.):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def duplicate_evidence(left, right, settings=None):
    """Return measured duplicate evidence for ONE observed frame, or ``None``.

    The caller must separately enforce equal assigned identity/anchor. Keeping
    this predicate independent of assignment lets an anchor matcher recognize
    duplicate observations before it applies the ordinary co-visibility veto.
    A face-only match requires high-quality face evidence in both observations.
    """
    if not left.get('frame_key') or left.get('frame_key') != right.get('frame_key'):
        return None
    if ('shot_id' in left and 'shot_id' in right and left['shot_id'] != right['shot_id']):
        return None
    person = _kind(left) == 'person'
    if person != (_kind(right) == 'person'):
        return None
    a, b = _box(left), _box(right)
    if a is None or b is None:
        return None
    area_a, area_b = _area(a), _area(b)
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    overlap = intersection / max(area_a+area_b-intersection, 1e-12)
    ratio = min(area_a, area_b) / max(area_a, area_b)
    evidence = dict(iou=overlap, area_ratio=ratio, family='person' if person else 'nonperson')
    if person:
        if overlap < THRESHOLDS['person_iou'] or ratio < THRESHOLDS['person_area_ratio']:
            return None
        body = _cosine(left, right, 'body_feature')
        visual = _cosine(left, right, 'visual_feature')
        face = _cosine(left, right, 'face_feature')
        quality_floor = max(THRESHOLDS['face_quality'], _number(getattr(settings, 'face_conflict_quality', 80.), 80.))
        qualities = [_number(obs.get('face_quality')) for obs in (left, right)]
        reliable_face = face is not None and min(qualities) >= quality_floor
        # A clear face disagreement cannot be erased by similar clothing.
        if reliable_face and face < .25:
            return None
        appearance_match = body is not None and visual is not None and body >= THRESHOLDS['body_similarity'] and visual >= THRESHOLDS['visual_similarity']
        face_match = reliable_face and face >= THRESHOLDS['face_similarity']
        if not appearance_match and not face_match:
            return None
        evidence.update(method='reliable face' if face_match else 'body and generic visual features',
                        body_similarity=body, visual_similarity=visual, face_similarity=face,
                        face_qualities=qualities, minimum_face_quality=quality_floor)
    else:
        if overlap < THRESHOLDS['object_iou'] or ratio < THRESHOLDS['object_area_ratio']:
            return None
        value = _cosine(left, right, 'object_feature')
        if value is None or value < THRESHOLDS['object_similarity']:
            return None
        evidence.update(method='generic object features', object_similarity=value)
    return evidence


def resolve_assigned_duplicates(groups, mapping, settings=None):
    """Return copied groups/mapping plus an audit; collapse only same-ID boxes.

    For each (assigned identity, frame), strongest-confidence boxes are kept
    first; equal confidence favors larger area. A removed box must directly
    match a retained box, so a chain of overlaps cannot merge separate subjects.
    Tracks with no surviving observations are removed from both returned maps.
    Unsupported simultaneous observations remain unchanged and appear in the
    audit's remaining_conflicts list.
    """
    copied_groups, copied_mapping = deepcopy(groups), deepcopy(mapping)
    buckets = defaultdict(list)
    for track_id, identity in copied_mapping.items():
        if identity is None or track_id not in copied_groups:
            continue
        for index, obs in enumerate(copied_groups[track_id]['observations']):
            if obs.get('frame_key'):
                buckets[identity, obs['frame_key']].append((track_id, index, obs))
    discarded = defaultdict(set)
    suppressions, remaining = [], []

    def rank(row):
        obs = row[2]
        box = _box(obs)
        confidence = _number(obs.get('detection_confidence'), -1.)
        return (-confidence, -(_area(box) if box else 0.), str(obs.get('detection_id', '')), row[0], row[1])

    for (identity, frame_key), rows in sorted(buckets.items()):
        kept = []
        for row in sorted(rows, key=rank):
            duplicate = None
            for prior in kept:
                evidence = duplicate_evidence(row[2], prior[2], settings)
                if evidence is not None:
                    duplicate = (prior, evidence)
                    break
            if duplicate is None:
                kept.append(row)
                continue
            prior, evidence = duplicate
            discarded[row[0]].add(row[1])
            suppressions.append(dict(entity_id=identity, frame_key=frame_key,
                                     suppressed_track_id=row[0], kept_track_id=prior[0],
                                     suppressed_detection_id=row[2].get('detection_id'),
                                     kept_detection_id=prior[2].get('detection_id'),
                                     suppressed_confidence=row[2].get('detection_confidence'),
                                     kept_confidence=prior[2].get('detection_confidence'), evidence=evidence))
        if len(kept) > 1:
            remaining.append(dict(entity_id=identity, frame_key=frame_key,
                                  track_ids=[row[0] for row in kept],
                                  detection_ids=[row[2].get('detection_id') for row in kept]))
    emptied = []
    for track_id, indices in discarded.items():
        group = copied_groups[track_id]
        group['observations'] = [obs for index, obs in enumerate(group['observations']) if index not in indices]
        if not group['observations']:
            emptied.append(track_id)
            del copied_groups[track_id]
            copied_mapping.pop(track_id, None)
    audit = dict(version=VERSION, thresholds=dict(THRESHOLDS), suppressed_count=len(suppressions),
                 suppressions=suppressions, removed_empty_tracks=sorted(emptied),
                 remaining_conflicts=remaining,
                 note='Heuristic duplicate suppression within an already assigned identity and observed frame; other co-visible conflicts are preserved for the caller.')
    return copied_groups, copied_mapping, audit
