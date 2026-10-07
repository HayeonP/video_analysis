"""Bounded appearance tracking and candidate retrieval, without VLM calls.

Within a shot, one Hungarian assignment joins consecutive adaptive samples. A
track's eligibility is configurable (default three sample steps and two seconds); elapsed
time comes from the input timeline, not an assumed sampling FPS. Appearance and
predicted box position both contribute. The local appearance floor (.58 body,
.52 DINO) is deliberately lower than a cross-shot identity threshold: nearby
observations of a changing pose need not form a new identity every frame. Low
appearance alone cannot pass; it also needs convincing spatial continuity.
An unambiguous high-quality face can bridge a large camera move inside a shot.

The constants are conservative starting heuristics, not calibrated accuracy
claims. Identity across different shots is *not* decided here. rank_candidates
only retrieves a fixed number of appearance neighbors for the bounded reviewer.
It never treats a detector class, a shared label, or narrative role as identity.
Person/non-person compatibility and same-frame exclusion are hard constraints.
Detector class changes are allowed within the non-person family, so a class
error does not create a forced split.

No model loading, file writes, request retries, or recursive search occurs here.
Input observations remain unchanged; each returned track owns copied records.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


MAX_SAMPLE_GAP = 3
MAX_TIME_GAP = 2.0
REFERENCE_LIMIT = 12
ASSIGNMENT_FLOOR = .43
AMBIGUITY_MARGIN = .035


@dataclass(frozen=True)
class TrackingConfig:
    max_sample_gap: int = MAX_SAMPLE_GAP
    max_time_gap_sec: float = MAX_TIME_GAP
    assignment_floor: float = ASSIGNMENT_FLOOR
    ambiguity_margin: float = AMBIGUITY_MARGIN

    def __post_init__(self):
        if type(self.max_sample_gap) is not int or self.max_sample_gap < 1:
            raise ValueError('tracking.max_sample_gap must be a positive integer')
        if not math.isfinite(self.max_time_gap_sec) or self.max_time_gap_sec <= 0:
            raise ValueError('tracking.max_time_gap_sec must be positive and finite')
        for name in ('assignment_floor', 'ambiguity_margin'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'tracking.{name} must be in 0..1')


def _kind(entry):
    return entry.get('entity_kind', entry.get('kind', 'object'))


def compatible(a, b):
    """Keep people separate from all non-people despite unstable class labels."""
    return (_kind(a) == 'person') == (_kind(b) == 'person')


def _time(obs):
    for key in ('full_time_sec', 'original_time_sec', 'time_sec', 'source_time_sec'):
        if key in obs:
            return float(obs[key])
    return 0.0


def _box(obs):
    box = np.asarray(obs['bbox'], dtype=np.float64)
    return box / 1000.0


def _area(box):
    return max(0., float(box[2]-box[0])) * max(0., float(box[3]-box[1]))


def _iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    inter = float(np.maximum(hi-lo, 0.).prod())
    return inter / max(_area(a)+_area(b)-inter, 1e-12)


def _center(box):
    return (box[:2]+box[2:]) * .5


def _vector(obs, key):
    value = obs.get(key)
    if value is None and key in {'object_feature', 'visual_feature'}:
        # Both fields come from the same DINO encoder. A later explicit class
        # correction changes the group kind, never the immutable crop evidence.
        alternate = 'visual_feature' if key == 'object_feature' else 'object_feature'
        value = obs.get(alternate)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or not np.isfinite(arr).all() or norm < 1e-10:
        return None
    return arr / norm


def _references(observations, limit=REFERENCE_LIMIT):
    """Bound comparisons while retaining clear faces and temporal diversity."""
    ordered = sorted(observations, key=lambda o: (_time(o), o.get('detection_id', '')))
    if len(ordered) <= limit:
        return ordered
    chosen = []
    for index in sorted(range(len(ordered)),
                        key=lambda i: (-float(ordered[i].get('face_quality', 0) or 0), i))[:2]:
        chosen.append(index)
    for index in np.linspace(0, len(ordered)-1, limit, dtype=int):
        if int(index) not in chosen:
            chosen.append(int(index))
        if len(chosen) >= limit:
            break
    return [ordered[i] for i in sorted(chosen)]


def _feature_evidence(left, right, key, minimum_quality=0):
    a = [_vector(o, key) for o in left
         if float(o.get('face_quality', 0) or 0) >= minimum_quality]
    b = [_vector(o, key) for o in right
         if float(o.get('face_quality', 0) or 0) >= minimum_quality]
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if not a or not b or any(v.shape != a[0].shape for v in a+b):
        return None
    similarities = np.clip(np.asarray(a) @ np.asarray(b).T, -1., 1.)
    best = float(similarities.max())
    # Reward support in multiple views without letting a poor back view erase a
    # clear frontal match. Unique per-view maxima avoid counting one pair twice.
    left_support = np.sort(similarities.max(axis=1))[-min(3, len(a)):]
    right_support = np.sort(similarities.max(axis=0))[-min(3, len(b)):]
    robust = .5 * (float(left_support.mean()) + float(right_support.mean()))
    return dict(best=best, robust=robust, score=.65*best+.35*robust,
                left_views=len(a), right_views=len(b))


def appearance_evidence(left, right, *, person):
    """Cosine evidence only; scores are similarities, never probabilities."""
    left, right = _references(left), _references(right)
    if person:
        return dict(face=_feature_evidence(left, right, 'face_feature', 20),
                    reliable_face=_feature_evidence(left, right, 'face_feature', 80),
                    body=_feature_evidence(left, right, 'body_feature'),
                    visual=_feature_evidence(left, right, 'visual_feature'))
    return dict(object=_feature_evidence(left, right, 'object_feature'))


def _deduplicate(observations):
    """Suppress near-identical boxes only; preserve nested carrier/passenger."""
    def confidence(obs):
        value = obs.get('detection_confidence')
        return float(value) if isinstance(value, (float, int)) and math.isfinite(value) else -1.
    ranked = sorted(observations, key=lambda o: (-confidence(o), o.get('detection_id', '')))
    kept = []
    for obs in ranked:
        box = _box(obs)
        duplicate = next((prior for prior in kept
                          if compatible(obs, prior)
                          and min(_area(box), _area(_box(prior))) /
                          max(_area(box), _area(_box(prior)), 1e-12) >= .90
                          and _iou(box, _box(prior)) >= .94), None)
        if duplicate is None:
            kept.append(deepcopy(obs))
        else:
            duplicate.setdefault('deduplicated_detection_ids', []).append(obs['detection_id'])
    return sorted(kept, key=lambda o: o.get('detection_id', ''))


def _local_score(query, track, frame_index, config=None):
    config = config or TrackingConfig()
    observations = track['observations']
    latest = observations[-1]
    dt = _time(query)-_time(latest)
    gap = frame_index-track['_frame_index']
    if not compatible(query, track) or dt < 0 or dt > config.max_time_gap_sec or not 1 <= gap <= config.max_sample_gap:
        return -1., None
    if query['frame_key'] in track['_frames']:
        return -1., None
    evidence = appearance_evidence([query], observations[-6:], person=_kind(query) == 'person')
    reliable = evidence.get('reliable_face')
    if reliable is not None and reliable['best'] < .20:
        return -1., None
    face = evidence.get('face')
    face_same = face is not None and face['best'] >= .62
    feature = evidence.get('body' if _kind(query) == 'person' else 'object')
    appearance = feature['score'] if feature else None
    visual = evidence.get('visual')
    if visual:
        # Generic DINO helps with animated faces/silhouettes that person ReID
        # models did not learn. Downweight it: similar clothes are not identity.
        appearance = max(appearance if appearance is not None else -1., .92*visual['score'])
    current_box, previous_box = _box(query), _box(latest)
    area_ratio = min(_area(current_box), _area(previous_box)) / max(_area(current_box), _area(previous_box), 1e-12)
    # Strong size disagreement needs evidence beyond a similar silhouette.
    if area_ratio < .12 and not face_same:
        return -1., None
    predicted = previous_box.copy()
    if len(observations) >= 2:
        earlier = observations[-2]
        previous_dt = _time(latest)-_time(earlier)
        if previous_dt > 1e-6:
            shift = (_center(previous_box)-_center(_box(earlier))) * min(dt/previous_dt, 2.)
            shift = np.clip(shift, -.20, .20)
            predicted += np.tile(shift, 2)
    overlap = max(_iou(current_box, previous_box), _iou(current_box, predicted))
    distance = min(float(np.linalg.norm(_center(current_box)-_center(previous_box))),
                   float(np.linalg.norm(_center(current_box)-_center(predicted))))
    extent = max(float(np.linalg.norm(current_box[2:]-current_box[:2])),
                 float(np.linalg.norm(previous_box[2:]-previous_box[:2])))
    # The time term handles non-uniform adaptive sampling. A screen-space cap
    # prevents even a long held sample from granting unlimited travel distance.
    radius = min(.35, .025+.18*min(dt, 1.)+.35*extent)
    proximity = math.exp(-2.0*(distance/max(radius, 1e-6))**2)
    spatial = .55*proximity+.45*overlap
    floor = .58 if _kind(query) == 'person' else .52
    if face_same:
        app_score = min(1., .80+.20*(face['best']-.62)/.38)
    elif appearance is None:
        # Geometry-only continuity is restricted to a near-identical adjacent
        # box, not a distant box whose missing embeddings hide uncertainty.
        if gap != 1 or dt > 1.1 or overlap < .65:
            return -1., None
        app_score = .42
    else:
        if appearance < floor or (distance > radius and overlap < .15 and appearance < .91):
            return -1., None
        if appearance < .76 and spatial < .35:
            return -1., None
        app_score = max(0., min(1., (appearance-.35)/.65))
    score = .65*app_score+.35*spatial-.025*(gap-1)
    proof = dict(method='within-shot appearance and box continuity', score=float(score),
                 appearance=appearance, face=face['best'] if face else None,
                 iou=float(overlap), center_distance=distance, elapsed_sec=dt, sample_gap=gap)
    return float(score), proof


def build_tracklets(raw_all, frames, settings, config=None):
    """Return ordered T IDs; never connect observations across a shot boundary.

    ``config`` controls temporal reach and assignment acceptance independently
    of the older global re-identification settings.
    Each frame's assignments are one-to-one, and ambiguous matches start a new
    provisional track rather than corrupting an existing track's references.
    """
    config = config or TrackingConfig()
    by_frame = defaultdict(list)
    for obs in raw_all:
        by_frame[obs['frame_key']].append(obs)
    tracks = {}
    active = []
    last_shot = None
    ordered_frames = sorted(frames, key=lambda f: (_time(f), f['frame_key']))
    for index, frame in enumerate(ordered_frames):
        shot = frame['shot_id']
        if shot != last_shot:
            active = []
            last_shot = shot
        active = [key for key in active
                  if index-tracks[key]['_frame_index'] <= config.max_sample_gap
                  and _time(frame)-_time(tracks[key]['observations'][-1]) <= config.max_time_gap_sec]
        batch = _deduplicate(by_frame.get(frame['frame_key'], []))
        scores = np.full((len(batch), len(active)+len(batch)), -1., dtype=np.float64)
        scores[:, len(active):] = config.assignment_floor
        proofs = {}
        for row, obs in enumerate(batch):
            for col, key in enumerate(active):
                scores[row, col], proofs[row, col] = _local_score(obs, tracks[key], index, config)
        choices = dict(zip(*linear_sum_assignment(-scores))) if batch else {}
        for row, obs in enumerate(batch):
            col = choices[row]
            accepted = col < len(active) and scores[row, col] > config.assignment_floor
            if accepted:
                row_rivals = np.delete(scores[row, :len(active)], col)
                col_rivals = np.delete(scores[:, col], row)
                rival = max(config.assignment_floor,
                            float(row_rivals.max()) if row_rivals.size else -1.,
                            float(col_rivals.max()) if col_rivals.size else -1.)
                accepted = scores[row, col]-rival >= config.ambiguity_margin
            if accepted:
                key = active[col]
                obs['track_evidence'] = proofs[row, col]
                tracks[key]['observations'].append(obs)
                tracks[key]['_frame_index'] = index
                tracks[key]['_frames'].add(obs['frame_key'])
            else:
                key = f'T{len(tracks)+1:04d}'
                obs['track_evidence'] = dict(method='new within-shot track', score=None)
                tracks[key] = dict(label=obs.get('label', ''), kind=_kind(obs),
                    appearance=obs.get('appearance', ''), shot_id=shot, observations=[obs],
                    _frame_index=index, _frames={obs['frame_key']})
                # Add births after evaluating this frame. Never make a new
                # same-frame track available to a subsequent detection.
        active.extend(key for key, track in tracks.items()
                      if track['_frame_index'] == index and key not in active)
    for track in tracks.values():
        track.pop('_frame_index')
        track.pop('_frames')
        # Non-person detector category corrections do not relabel a person.
        counts = Counter(_kind(o) for o in track['observations'])
        track['kind'] = counts.most_common(1)[0][0]
        best = max(track['observations'], key=lambda o: (
            float(o.get('face_quality', 0) or 0),
            float(o.get('detection_confidence', 0) or 0), _area(_box(o))))
        track['label'] = best.get('label', track['label'])
        track['appearance'] = best.get('appearance', track['appearance'])
    return tracks


def rank_candidates(entry, registry, limit=3):
    """Retrieve at most ``limit`` compatible global IDs, without joining them.

    Registry entries contain kind and observations, like the returned tracklets.
    The score is an appearance ranking, not a calibrated identity probability.
    Weak candidates remain eligible because clothes/pose can change; the caller
    must ask for evidence before merging. Missing features produce no candidate.
    """
    if limit <= 0:
        return []
    observed = entry.get('observations', [])
    occupied = {o['frame_key'] for o in observed}
    results = []
    for key, candidate in sorted(registry.items()):
        if not compatible(entry, candidate):
            continue
        samples = candidate.get('observations', [])
        if not samples or occupied & {o['frame_key'] for o in samples}:
            continue
        evidence = appearance_evidence(observed, samples, person=_kind(entry) == 'person')
        reliable = evidence.get('reliable_face')
        if reliable is not None and reliable['best'] < .20:
            continue
        base = evidence.get('body' if _kind(entry) == 'person' else 'object')
        face = evidence.get('face')
        visual = evidence.get('visual')
        scores = []
        if base:
            scores.append(base['score'])
        if visual:
            scores.append(.95*visual['score'])
        if face:
            # A .55 SFace cosine is already strong evidence. Map its useful
            # range to the same ranking scale as body/DINO, retaining raw values.
            scores.append(float(np.clip(.55+.45*(face['score']-.20)/.55, 0., 1.)))
        if not scores:
            continue
        results.append(dict(entity_id=key, score=float(max(scores)), evidence=evidence))
    return sorted(results, key=lambda r: (-r['score'], r['entity_id']))[:int(limit)]
