"""Deterministic appearance identities before any optional language-model review.

Track membership is decided from crop embeddings and physical exclusions. Labels,
detector class names, movie names and narrative roles never establish identity.
The result remains an estimate: an unlinked track is an unresolved singleton,
not evidence that a new physical individual exists.

Candidate edges are bounded to the union of each track's top-k appearance
neighbors. Edges are processed strongest first; every component join checks all
members for co-visibility and reliable-face contradiction. Larger components
also need representative support, several independent supporting track pairs,
or a particularly strong identity/appearance bridge. This permits changing views
without allowing an arbitrary chain of marginal similarities to absorb people.

No model loading, inference, file writes or mutation of input observations occurs
here. Similarity thresholds are configurable heuristics, not probabilities.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import math

from efficient_tracks import appearance_evidence, rank_candidates


VERSION = 1


@dataclass(frozen=True)
class VisualAssociationConfig:
    candidates: int = 5
    face_candidate_slots: int = 2
    reference_tracks: int = 4
    face: float = .55
    face_joint: float = .48
    joint_visual: float = .50
    joint_body: float = .60
    body: float = .86
    visual: float = .78
    object: float = .90
    clear_face_contradiction: float = .20
    coherence_min_size: int = 4
    coherence_supports: int = 2
    trusted_face: float = .62
    trusted_body: float = .93
    trusted_visual: float = .90
    trusted_object: float = .98

    def __post_init__(self):
        integer_fields = {'candidates', 'face_candidate_slots', 'reference_tracks', 'coherence_min_size',
                          'coherence_supports'}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in integer_fields:
                if type(value) is not int or value < 1:
                    raise ValueError(f'visual_association.{field.name} must be a positive integer')
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value) or not -1 <= value <= 1):
                raise ValueError(f'visual_association.{field.name} must be a finite cosine in [-1, 1]')
        if self.coherence_min_size < 3:
            raise ValueError('visual_association.coherence_min_size must be at least 3')
        if self.coherence_supports < 2:
            raise ValueError('visual_association.coherence_supports must be at least 2')
        if not self.clear_face_contradiction < self.face_joint <= self.face <= self.trusted_face:
            raise ValueError('visual_association face thresholds must increase from contradiction to trusted')
        if not self.joint_body <= self.body <= self.trusted_body:
            raise ValueError('visual_association body thresholds must increase from joint to trusted')
        if not self.joint_visual <= self.visual <= self.trusted_visual:
            raise ValueError('visual_association visual thresholds must increase from joint to trusted')
        if self.object > self.trusted_object:
            raise ValueError('visual_association.object must not exceed trusted_object')


def _config(config):
    if config is None:
        return VisualAssociationConfig()
    if isinstance(config, dict):
        return VisualAssociationConfig(**config)
    if not isinstance(config, VisualAssociationConfig):
        raise TypeError('config must be VisualAssociationConfig, a mapping, or None')
    return config


def _kind(entry):
    # A validated group-level correction takes precedence over raw detection kind.
    return entry.get('kind', entry.get('entity_kind', 'object'))


def _effective(entry):
    result = dict(entry)
    result['kind'] = _kind(entry)
    result.pop('entity_kind', None)
    return result


def _score(evidence, name):
    part = evidence.get(name)
    return None if part is None else float(part['score'])


def _above(value, threshold):
    return value is not None and value >= threshold


def visual_support(entry, other, config=None):
    """Return auditable positive evidence or an explicit reason not to link.

    ``accepted`` is a conservative automatic identity proposal. ``contradiction``
    identifies a hard person/non-person, same-frame, or reliable-face exclusion;
    ordinary missing or low-similarity evidence is *not* a contradiction. Callers
    must inspect ``accepted``; this function always returns a dictionary.
    """
    config = _config(config)
    person = _kind(entry) == 'person'
    other_person = _kind(other) == 'person'
    result = dict(accepted=False, contradiction=False, reason='insufficient_appearance',
                  rule=None, strength=None, trusted_bridge=False, evidence={})
    if person != other_person:
        result.update(contradiction=True, reason='person_nonperson_conflict')
        return result
    left = entry.get('observations', [])
    right = other.get('observations', [])
    shared = {o['frame_key'] for o in left if 'frame_key' in o} & {
        o['frame_key'] for o in right if 'frame_key' in o}
    if shared:
        result.update(contradiction=True, reason='co_visible', frames=sorted(shared))
        return result
    evidence = appearance_evidence(left, right, person=person)
    result['evidence'] = evidence
    reliable = evidence.get('reliable_face')
    if reliable is not None and reliable['best'] < config.clear_face_contradiction:
        result.update(contradiction=True, reason='reliable_face_conflict')
        return result
    if person:
        face, body, visual = (_score(evidence, field) for field in ('face', 'body', 'visual'))
        if _above(face, config.face):
            # Priority is for graph ordering, not an identity probability.
            result.update(accepted=True, rule='face', strength=3. + face)
        elif (_above(face, config.face_joint)
              and (_above(visual, config.joint_visual) or _above(body, config.joint_body))):
            result.update(accepted=True, rule='face_with_appearance', strength=2. + face)
        elif _above(body, config.body) and _above(visual, config.visual):
            result.update(accepted=True, rule='body_and_visual', strength=1. + min(body, visual))
        result['trusted_bridge'] = bool(
            _above(face, config.trusted_face)
            or (_above(body, config.trusted_body) and _above(visual, config.trusted_visual)))
    else:
        visual = _score(evidence, 'object')
        if _above(visual, config.object):
            result.update(accepted=True, rule='object_visual', strength=1. + visual)
        result['trusted_bridge'] = _above(visual, config.trusted_object)
    if result['accepted']:
        result['reason'] = 'positive_appearance'
    return result


def _track_order(item):
    key, entry = item
    times = []
    for obs in entry.get('observations', []):
        for field in ('full_time_sec', 'original_time_sec', 'time_sec', 'source_time_sec'):
            value = obs.get(field)
            if isinstance(value, (int, float)) and math.isfinite(value):
                times.append(float(value))
                break
    return (min(times, default=float('inf')), key)


def _quality(entry):
    observed = entry.get('observations', [])
    face = max((float(o.get('face_quality', 0) or 0) for o in observed), default=0.)
    confidence = max((float(o.get('detection_confidence', 0) or 0) for o in observed), default=0.)
    return face, len(observed), confidence


def build_visual_identities(groups, settings, config=None):
    """Return ``(identities, track_to_identity, audit)`` with total track coverage.

    ``settings`` is accepted for the pipeline interface; all association behavior
    is controlled by the validated ``config``. Each V identity owns copied actual
    observations and real reference track IDs. Singletons remain available for
    later review, explicitly marked unresolved instead of being discarded.
    Duplicate observations must be resolved before tracking; same-frame tracks
    are intentionally never joined here.
    """
    config = _config(config)
    entries = {key: _effective(value) for key, value in groups.items()}
    keys = sorted(entries)
    proofs = {}

    def proof(a, b):
        pair = tuple(sorted((a, b)))
        if pair not in proofs:
            proofs[pair] = visual_support(entries[pair[0]], entries[pair[1]], config)
        return proofs[pair]

    retrieval = []
    candidate_pairs = set()
    for key in keys:
        ranked = rank_candidates(entries[key],
            {other: value for other, value in entries.items() if other != key},
            limit=len(entries))
        # A plausible face bridge must not disappear behind many near-identical
        # coats in the body/DINO ranking. Reserve a few of the same fixed slots;
        # this does not increase the total number of candidate edges per track.
        face_ranked = sorted(
            (neighbor for neighbor in ranked
             if _above(_score(neighbor['evidence'], 'face'), config.face_joint)),
            key=lambda neighbor: (-_score(neighbor['evidence'], 'face'), neighbor['entity_id']))
        neighbors = face_ranked[:min(config.face_candidate_slots, config.candidates)]
        chosen = {neighbor['entity_id'] for neighbor in neighbors}
        for neighbor in ranked:
            if len(neighbors) >= config.candidates:
                break
            if neighbor['entity_id'] not in chosen:
                neighbors.append(neighbor)
                chosen.add(neighbor['entity_id'])
        retrieval.append(dict(track_id=key, candidates=[dict(track_id=n['entity_id'],
                              retrieval_score=n['score']) for n in neighbors]))
        for neighbor in neighbors:
            candidate_pairs.add(tuple(sorted((key, neighbor['entity_id']))))

    edges = []
    declined = []
    for a, b in sorted(candidate_pairs):
        evidence = proof(a, b)
        record = dict(left=a, right=b, **evidence)
        (edges if evidence['accepted'] else declined).append(record)
    edges.sort(key=lambda item: (-item['strength'], item['left'], item['right']))

    owner = {key: key for key in keys}
    members = {key: {key} for key in keys}
    joined, blocked = [], []

    def representatives(component):
        def score(key):
            support = [proof(key, other) for other in component if other != key]
            positive = [p for p in support if p['accepted']]
            return (len(positive), sum(p['strength'] for p in positive),
                    *_quality(entries[key]))
        return sorted(component, key=lambda key: tuple(-v for v in score(key)) + (key,))

    for edge in edges:
        a, b = edge['left'], edge['right']
        left_id, right_id = owner[a], owner[b]
        if left_id == right_id:
            continue
        left, right = members[left_id], members[right_id]
        cross = [(x, y, proof(x, y)) for x in sorted(left) for y in sorted(right)]
        conflicts = [dict(left=x, right=y, **p) for x, y, p in cross if p['contradiction']]
        if conflicts:
            blocked.append(dict(left=a, right=b, reason='component_contradiction',
                                conflicts=conflicts))
            continue
        positive = [(x, y, p) for x, y, p in cross if p['accepted']]
        left_core, right_core = representatives(left)[0], representatives(right)[0]
        representative_support = proof(left_core, right_core)['accepted']
        multiple_support = (
            len({x for x, _, _ in positive}) >= min(config.coherence_supports, len(left))
            and len({y for _, y, _ in positive}) >= min(config.coherence_supports, len(right)))
        small_component = len(left) + len(right) < config.coherence_min_size
        coherence = dict(left_reference=left_core, right_reference=right_core,
                         representative_support=representative_support,
                         mutual_support=multiple_support, trusted_bridge=edge['trusted_bridge'],
                         small_component=small_component,
                         supporting_pairs=[dict(left=x, right=y, rule=p['rule'])
                                           for x, y, p in positive])
        if not (small_component or representative_support or multiple_support or edge['trusted_bridge']):
            blocked.append(dict(left=a, right=b, reason='weak_chain_without_reference_support',
                                coherence=coherence))
            continue
        keep, remove = sorted((left_id, right_id))
        members[keep] |= members.pop(remove)
        for key in members[keep]:
            owner[key] = keep
        joined.append(dict(**edge, component_tracks=sorted(members[keep]), coherence=coherence))

    components = sorted(members.values(), key=lambda component:
        min(_track_order((key, entries[key])) for key in component))
    identities, mapping = {}, {}
    for component in components:
        identity_id = f'V{len(identities)+1:04d}'
        reference_keys = representatives(component)[:config.reference_tracks]
        reference = entries[reference_keys[0]]
        track_ids = sorted(component, key=lambda key: _track_order((key, entries[key])))
        observed = [deepcopy(obs) for key in track_ids for obs in entries[key].get('observations', [])]
        kind = Counter(_kind(entries[key]) for key in track_ids).most_common(1)[0][0]
        identities[identity_id] = dict(
            label=reference.get('label', ''), appearance=reference.get('appearance', ''),
            kind=kind, reference_track_ids=reference_keys, track_ids=track_ids,
            observations=observed, status='visual_estimated',
            support_status='linked' if len(component) > 1 else 'unresolved_singleton',
            identity_method='bounded appearance graph with physical exclusions')
        mapping.update({key: identity_id for key in track_ids})
    audit = dict(version=VERSION, config=asdict(config), tracks=len(entries),
                 identities=len(identities), linked_tracks=sum(
                     len(value['track_ids']) for value in identities.values()
                     if value['support_status'] == 'linked'),
                 unresolved_singletons=sum(value['support_status'] == 'unresolved_singleton'
                                           for value in identities.values()),
                 candidate_pairs=len(candidate_pairs), retrieval=retrieval,
                 accepted_edges=joined, blocked_edges=blocked, declined_candidates=declined,
                 notes=['Cosine similarities are heuristic evidence, not probabilities.',
                        'A singleton is unresolved; it is not a proven new physical identity.',
                        'No text label or expected entity count participates in association.'])
    return identities, mapping, audit
