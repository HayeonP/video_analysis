"""Fuse repeated observations before semantic identity assignment.

This module uses only cached box and embedding evidence. It does not infer new
boxes or identities, load models, or depend on any particular video. The kept
box, crop, and embeddings always belong to one actual detector observation.
Suppressed detection IDs and alternative detector classes remain in provenance.

Frame clusters and track clusters use complete linkage: every pair in a cluster
must satisfy the relevant rule. An overlap chain cannot absorb another subject.
"""
from collections import defaultdict
from copy import deepcopy
import math

from duplicate_tracks import duplicate_evidence


VERSION = 1
DEFAULT_THRESHOLDS = {
    'near_identical_iou': .94,
    'near_identical_area_ratio': .90,
    'near_identical_visual': .97,
    'track_min_support_frames': 2,
    'track_min_support_fraction': .50,
    'person_track_min_iou': .20,
    'person_track_min_area_ratio': .45,
    'object_track_min_iou': .60,
    'object_track_min_area_ratio': .60,
    'face_quality': 80.,
    'face_conflict_similarity': .25,
}


def _thresholds(settings):
    """Optional settings.observation_fusion mapping overrides generic defaults."""
    values = dict(DEFAULT_THRESHOLDS)
    overrides = (settings.get('observation_fusion', {}) if isinstance(settings, dict)
                 else getattr(settings, 'observation_fusion', {})) if settings is not None else {}
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError('observation_fusion settings must be a mapping')
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f'Unknown observation fusion settings: {sorted(unknown)}')
    values.update(overrides)
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid observation fusion setting: {key}')
        if key == 'track_min_support_frames':
            if value < 2 or int(value) != value:
                raise ValueError('track_min_support_frames must be an integer >= 2')
        elif key == 'face_quality':
            if value < 0:
                raise ValueError('face_quality must be nonnegative')
        elif not 0 <= value <= 1:
            raise ValueError(f'{key} must be between 0 and 1')
    return values


def _number(value, default=0.):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _box(obs):
    try:
        value = [float(v) for v in obs['bbox']]
    except (KeyError, TypeError, ValueError):
        return None
    if (len(value) != 4 or not all(math.isfinite(v) for v in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        return None
    return value


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1]) if box else 0.


def _geometry(left, right):
    a, b = _box(left), _box(right)
    if a is None or b is None:
        return None
    aa, ab = _area(a), _area(b)
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return dict(iou=intersection / (aa + ab - intersection),
                area_ratio=min(aa, ab) / max(aa, ab))


def _kind(obs):
    return obs.get('entity_kind', obs.get('kind', 'object'))


def _person(obs):
    return _kind(obs) == 'person'


def _cosine(left, right, key):
    def vector(obs):
        value = obs.get(key)
        if value is None and key in {'visual_feature', 'object_feature'}:
            value = obs.get('object_feature' if key == 'visual_feature' else 'visual_feature')
        try:
            value = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        if not value or not all(math.isfinite(v) for v in value):
            return None
        norm = math.hypot(*value)
        return [v / norm for v in value] if norm > 1e-12 else None
    a, b = vector(left), vector(right)
    if a is None or b is None or len(a) != len(b):
        return None
    return max(-1., min(1., math.fsum(x * y for x, y in zip(a, b))))


def _face_conflict(left, right, thresholds, settings):
    configured = (settings.get('face_conflict_quality', 80.) if isinstance(settings, dict)
                  else getattr(settings, 'face_conflict_quality', 80.))
    quality = max(thresholds['face_quality'], _number(configured, 80.))
    if min(_number(left.get('face_quality')), _number(right.get('face_quality'))) < quality:
        return False
    face = _cosine(left, right, 'face_feature')
    return face is not None and face < thresholds['face_conflict_similarity']


def _same_frame(left, right):
    return (bool(left.get('frame_key')) and left.get('frame_key') == right.get('frame_key')
            and not (left.get('shot_id') is not None and right.get('shot_id') is not None
                     and left['shot_id'] != right['shot_id']))


def _frame_evidence(left, right, thresholds, settings):
    if not _same_frame(left, right) or _face_conflict(left, right, thresholds, settings):
        return None
    geometry = _geometry(left, right)
    if geometry is None:
        return None
    visual = _cosine(left, right, 'visual_feature')
    # Near-identical crops can receive conflicting class labels. This exception
    # is intentionally too narrow to erase a contained prop or passenger.
    if (geometry['iou'] >= thresholds['near_identical_iou']
            and geometry['area_ratio'] >= thresholds['near_identical_area_ratio']
            and visual is not None and visual >= thresholds['near_identical_visual']):
        return dict(geometry, method='near-identical observed region',
                    visual_similarity=visual, cross_kind=_person(left) != _person(right))
    if _person(left) != _person(right):
        return None
    evidence = duplicate_evidence(left, right, settings)
    return dict(evidence, cross_kind=False) if evidence is not None else None


def _rank_observation(obs):
    return (-_number(obs.get('detection_confidence'), -1.), -_area(_box(obs)),
            str(obs.get('detection_id', '')))


def _source_ids(obs):
    result = set(obs.get('fusion_source_detection_ids', []))
    result.update(obs.get('deduplicated_detection_ids', []))
    if obs.get('detection_id') is not None:
        result.add(obs['detection_id'])
    return result


def _categories(obs):
    result = set(obs.get('fusion_alternative_categories', []))
    result.update(obs.get('alternative_categories', []))
    provenance = obs.get('detector_provenance') or {}
    result.update(provenance.get('alternative_categories', []))
    category = obs.get('detector_class', obs.get('category'))
    if category:
        result.add(category)
    return result


def _combine_observations(rows, method, source_tracks=None):
    winner = deepcopy(min(rows, key=_rank_observation))
    source_ids = set().union(*(_source_ids(o) for o in rows))
    categories = set().union(*(_categories(o) for o in rows))
    primary = winner.get('detector_class', winner.get('category'))
    categories.discard(primary)
    winner['fusion_source_detection_ids'] = sorted(source_ids)
    winner['deduplicated_detection_ids'] = sorted(source_ids - {winner.get('detection_id')})
    winner['fusion_alternative_categories'] = sorted(categories)
    winner['fusion_method'] = method
    if source_tracks is not None:
        tracks = set(source_tracks)
        for obs in rows:
            tracks.update(obs.get('fusion_source_track_ids', []))
        winner['fusion_source_track_ids'] = sorted(tracks)
    return winner


def fuse_frame_observations(raw_all, settings=None):
    """Return copied observations and audit, independent of assigned identity.

    Every suppressed observation must directly match every other member of its
    cluster. The highest-confidence original box/crop/features survive together.
    Input ordering is retained according to the surviving observation's index.
    """
    thresholds = _thresholds(settings)
    indexed = list(enumerate(raw_all))
    by_frame = defaultdict(list)
    untouched = []
    for index, obs in indexed:
        if not obs.get('frame_key'):
            untouched.append((index, deepcopy(obs)))
        else:
            by_frame[obs.get('shot_id'), obs['frame_key']].append((index, obs))
    output, fusions = list(untouched), []
    for _, rows in sorted(by_frame.items(), key=lambda item: str(item[0])):
        clusters = []
        for row in sorted(rows, key=lambda item: (_rank_observation(item[1]), item[0])):
            target = None
            for cluster in clusters:
                if all(_frame_evidence(row[1], prior[1], thresholds, settings) is not None for prior in cluster):
                    target = cluster
                    break
            if target is None:
                clusters.append([row])
            else:
                target.append(row)
        for cluster in clusters:
            winner_index, winner = cluster[0]
            if len(cluster) == 1:
                output.append((winner_index, deepcopy(winner)))
                continue
            fused = _combine_observations([obs for _, obs in cluster], 'same-frame duplicate evidence')
            output.append((winner_index, fused))
            evidence = [dict(left_detection_id=a.get('detection_id'), right_detection_id=b.get('detection_id'),
                             evidence=_frame_evidence(a, b, thresholds, settings))
                        for index, (_, a) in enumerate(cluster) for _, b in cluster[index + 1:]]
            fusions.append(dict(frame_key=winner['frame_key'], shot_id=winner.get('shot_id'),
                                kept_detection_id=winner.get('detection_id'),
                                suppressed_detection_ids=[o.get('detection_id') for _, o in cluster[1:]],
                                alternative_categories=fused['fusion_alternative_categories'], pair_evidence=evidence))
    copied_raw = [obs for _, obs in sorted(output, key=lambda row: row[0])]
    audit = dict(version=VERSION, thresholds=thresholds, input_observations=len(raw_all),
                 output_observations=len(copied_raw), suppressed_count=len(raw_all) - len(copied_raw),
                 fusion_clusters=fusions, identity_independent=True, complete_linkage=True,
                 note='Retains one original observed box and its corresponding crop/features; no synthetic boxes.')
    return copied_raw, audit


def _shot(group):
    shots = {obs.get('shot_id') for obs in group.get('observations', []) if obs.get('shot_id') is not None}
    if group.get('shot_id') is not None:
        shots.add(group['shot_id'])
    return next(iter(shots)) if len(shots) == 1 else None


def _by_frame(group):
    result = defaultdict(list)
    for obs in group.get('observations', []):
        if obs.get('frame_key'):
            result[obs['frame_key']].append(obs)
    return result


def _track_pair(left, right, thresholds, settings):
    shot = _shot(left)
    if shot is None or shot != _shot(right):
        return None
    a, b = _by_frame(left), _by_frame(right)
    common = sorted(set(a) & set(b))
    if len(common) < thresholds['track_min_support_frames']:
        return None
    evidence = []
    support = 0
    cross_family = _person(left) != _person(right)
    for frame in common:
        # A malformed track cannot be made valid by hiding an extra observation.
        if len(a[frame]) != 1 or len(b[frame]) != 1:
            return None
        oa, ob = a[frame][0], b[frame][0]
        if _face_conflict(oa, ob, thresholds, settings):
            return None
        geometry = _geometry(oa, ob)
        if geometry is None:
            return None
        direct = _frame_evidence(oa, ob, thresholds, settings)
        if cross_family or _person(oa) != _person(ob):
            if direct is None or direct['method'] != 'near-identical observed region':
                return None
        else:
            family = 'person' if _person(oa) else 'object'
            if (geometry['iou'] < thresholds[f'{family}_track_min_iou']
                    or geometry['area_ratio'] < thresholds[f'{family}_track_min_area_ratio']):
                return None
        support += int(direct is not None)
        evidence.append(dict(frame_key=frame, geometry=geometry, direct_duplicate_evidence=direct))
    fraction = support / len(common)
    if support < thresholds['track_min_support_frames'] or fraction < thresholds['track_min_support_fraction']:
        return None
    return dict(shot_id=shot, shared_frames=len(common), support_frames=support,
                support_fraction=fraction, cross_kind=cross_family, frame_evidence=evidence)


def _rank_track(item):
    key, group = item
    rows = group.get('observations', [])
    confidence = sum(_number(o.get('detection_confidence')) for o in rows) / max(1, len(rows))
    return (-len(rows), -confidence, str(key))


def reconcile_duplicate_tracklets(groups, settings=None):
    """Fuse same-shot tracks with repeated direct duplicate evidence.

    Returns (copied_groups, alias_map, audit). alias_map covers every input track
    ID, including unchanged tracks. Distinct simultaneous boxes and clear face
    disagreements veto a pair. Every pair in a merged cluster must qualify.
    """
    thresholds = _thresholds(settings)
    ordered = sorted(groups.items(), key=_rank_track)
    pairs = {}

    def pair(left, right):
        key = tuple(sorted((left, right)))
        if key not in pairs:
            pairs[key] = _track_pair(groups[left], groups[right], thresholds, settings)
        return pairs[key]

    clusters = []
    for key, _ in ordered:
        target = None
        for cluster in clusters:
            if all(pair(key, prior) is not None for prior in cluster):
                target = cluster
                break
        if target is None:
            clusters.append([key])
        else:
            target.append(key)
    merged, aliases, fusions = {}, {}, []
    for cluster in clusters:
        canonical = cluster[0]
        aliases.update({key: canonical for key in cluster})
        if len(cluster) == 1:
            merged[canonical] = deepcopy(groups[canonical])
            continue
        group = deepcopy(groups[canonical])
        frames = defaultdict(list)
        missing = []
        for key in cluster:
            for obs in groups[key].get('observations', []):
                if obs.get('frame_key'):
                    frames[obs['frame_key']].append((key, obs))
                else:
                    missing.append(deepcopy(obs))
        observations, suppression = list(missing), []
        for frame, rows in frames.items():
            winner = _combine_observations([obs for _, obs in rows], 'repeated same-shot duplicate track evidence',
                                           [key for key, _ in rows])
            observations.append(winner)
            if len(rows) > 1:
                suppression.append(dict(frame_key=frame, kept_detection_id=winner.get('detection_id'),
                                        suppressed_detection_ids=[obs.get('detection_id') for _, obs in rows
                                                                  if obs.get('detection_id') != winner.get('detection_id')]))
        group['observations'] = sorted(observations, key=lambda obs: (
            _number(obs.get('full_time_sec', obs.get('time_sec'))), str(obs.get('frame_key', '')), str(obs.get('detection_id', ''))))
        source_tracks = set(cluster)
        for key in cluster:
            source_tracks.update(groups[key].get('fusion_source_track_ids', []))
        group['fusion_source_track_ids'] = sorted(source_tracks)
        group['fusion_method'] = 'repeated same-shot duplicate track evidence'
        merged[canonical] = group
        fusions.append(dict(kept_track_id=canonical, merged_track_ids=cluster[1:],
                            pair_evidence=[dict(left_track_id=left, right_track_id=right, evidence=pair(left, right))
                                           for index, left in enumerate(cluster) for right in cluster[index + 1:]],
                            frame_suppressions=suppression))
    # Preserve the original encounter order of canonical tracks for exporters.
    copied_groups = {key: merged[key] for key in groups if key in merged}
    before = sum(len(group.get('observations', [])) for group in groups.values())
    after = sum(len(group.get('observations', [])) for group in copied_groups.values())
    audit = dict(version=VERSION, thresholds=thresholds, input_tracks=len(groups), output_tracks=len(copied_groups),
                 merged_track_count=len(groups) - len(copied_groups), suppressed_observation_count=before - after,
                 fusion_clusters=fusions, identity_independent=True, complete_linkage=True,
                 note='Requires repeated duplicate support and compatible geometry in every shared frame; preserves observed boxes.')
    return copied_groups, aliases, audit
