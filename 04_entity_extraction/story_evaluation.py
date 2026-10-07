"""Evaluate partial identity annotations through immutable source observations.

Annotations are evaluation data, never production matching rules. Unannotated
entities are unknown. Incomplete and contradictory groups are reported rather
than converted to negatives or repaired by transitive union.
"""
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
import argparse
import hashlib
import json

from story_export import box_iou, read_json, write_json


def _observations(value):
    if isinstance(value, (str, Path)):
        value = read_json(value)
    return value['observations'] if isinstance(value, dict) else value


def pixel_box(observation):
    if 'bbox_pixels' in observation:
        return list(map(float, observation['bbox_pixels']))
    box = list(map(float, observation['bbox']))
    if 'normalized' in observation.get('bbox_format', ''):
        width, height = observation['source_size']
        return [x*bound/1000 for x, bound in zip(box, [width, height, width, height])]
    return box


def match_source_observations(old_observations, new_observations, *, min_iou=.25, frame_tolerance=0):
    """Map old labels onto new observations using shot + source frame + geometry.

    Multiple old duplicate boxes may legitimately match one new detection.
    No entity appearance/name/manual group is used in this geometric transfer.
    """
    by_frame = defaultdict(list)
    for row in new_observations:
        by_frame[(row['shot_id'], int(row['source_frame']))].append(row)
    matches, unmatched, ambiguous = [], [], []
    for old in old_observations:
        frame = int(old['source_frame']); box = pixel_box(old)
        candidates = []
        for offset in range(-frame_tolerance, frame_tolerance+1):
            for new in by_frame[(old['shot_id'], frame+offset)]:
                geometries = [('retained_box', new.get('detection_id'), pixel_box(new))]
                geometries.extend(('merged_source_box', source.get('detection_id'), source['bbox_pixels'])
                    for source in new.get('merged_source_observations', []) if source.get('bbox_pixels'))
                geometry = max(geometries, key=lambda item: box_iou(box, item[2]))
                overlap = box_iou(box, geometry[2])
                if overlap >= min_iou:
                    candidates.append((overlap, -abs(offset), new, geometry))
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if not candidates:
            unmatched.append(dict(old_id=old['entity_id'], observation_id=old.get('detection_id'),
                                  shot_id=old['shot_id'], source_frame=frame))
            continue
        best = candidates[0]
        different = next((c for c in candidates[1:] if c[2]['entity_id'] != best[2]['entity_id']), None)
        if different and best[0]-different[0] < .05:
            ambiguous.append(dict(old_id=old['entity_id'], observation_id=old.get('detection_id'),
                shot_id=old['shot_id'], source_frame=frame,
                candidates=[dict(entity_id=c[2]['entity_id'], iou=c[0]) for c in candidates[:3]]))
            continue
        new = best[2]
        matches.append(dict(old_id=old['entity_id'], new_id=new['entity_id'],
            old_observation_id=old.get('detection_id'), new_observation_id=new.get('detection_id'),
            shot_id=old['shot_id'], source_frame=frame, new_source_frame=int(new['source_frame']),
            iou=best[0], kind=new.get('kind', 'unknown'), matched_geometry=best[3][0],
            geometry_detection_id=best[3][1]))
    return dict(matches=matches, unmatched=unmatched, ambiguous=ambiguous,
                criteria=dict(min_iou=min_iou, frame_tolerance=frame_tolerance,
                              ambiguous_iou_margin=.05, coordinates='source pixels'))


def _prepare_groups(annotation, available_ids):
    groups = annotation['groups']
    membership = defaultdict(set)
    for group in groups:
        for eid in set(group['ids']):
            membership[eid].add(group['name'])
    explicit = set(annotation.get('ambiguous_ids', []))
    contradictory = {eid for eid, names in membership.items() if len(names) > 1}
    invalid = {eid for eid in membership if eid not in available_ids}
    excluded = explicit | contradictory | invalid
    clean = []
    for group in groups:
        unique = list(dict.fromkeys(group['ids']))
        clean.append(dict(group, ids=[eid for eid in unique if eid not in excluded],
                          excluded_ids=[eid for eid in unique if eid in excluded],
                          cross_group_eligible=not bool(set(unique) & contradictory)))
    return clean, dict(contradictory_ids=sorted(contradictory), explicit_ambiguous_ids=sorted(explicit),
                       unavailable_ids=sorted(invalid), excluded_ids=sorted(excluded))


def _identity_metrics(groups, assignments, all_assignments=None):
    all_assignments = all_assignments or {old: [new] for old, new in assignments.items()}
    group_rows, correct, total, covered_pairs = [], 0, 0, 0
    for group in groups:
        ids = group['ids']; pairs = list(combinations(ids, 2))
        matched_pairs = [(a, b) for a, b in pairs if a in assignments and b in assignments]
        true_pairs = [(a, b) for a, b in matched_pairs if assignments[a] == assignments[b]]
        counts = Counter(assignments[eid] for eid in ids if eid in assignments)
        fragments = sorted({new for eid in ids for new in all_assignments.get(eid, [])})
        group_rows.append(dict(name=group['name'], annotated_ids=ids, excluded_ids=group['excluded_ids'],
            mapped_ids=[eid for eid in ids if eid in assignments], unmapped_ids=[eid for eid in ids if eid not in assignments],
            target_identity_count=1 if ids else 0, dominant_identity_count=len(counts),
            all_supported_identity_count=len(fragments), supported_global_ids=fragments,
            dominant_assignments=dict(counts), same_id_pairs=len(pairs),
            covered_same_id_pairs=len(matched_pairs), correct_same_id_pairs=len(true_pairs),
            same_id_pair_recall=len(true_pairs)/len(pairs) if pairs else None,
            covered_pair_recall=len(true_pairs)/len(matched_pairs) if matched_pairs else None,
            cross_group_eligible=group['cross_group_eligible']))
        correct += len(true_pairs); total += len(pairs); covered_pairs += len(matched_pairs)
    negative_pairs, false_merges = [], []
    eligible = [g for g in groups if g['cross_group_eligible']]
    for left, right in combinations(eligible, 2):
        for a, b in product(left['ids'], right['ids']):
            if a not in assignments or b not in assignments:
                continue
            negative_pairs.append((a, b))
            if assignments[a] == assignments[b]:
                false_merges.append(dict(left_id=a, right_id=b, left_group=left['name'], right_group=right['name'],
                                         global_id=assignments[a]))
    predicted_same_pairs = correct + len(false_merges)
    return dict(groups=group_rows, same_id_pairs=total, covered_same_id_pairs=covered_pairs,
        correct_same_id_pairs=correct, same_id_pair_recall=correct/total if total else None,
        covered_same_id_pair_recall=correct/covered_pairs if covered_pairs else None,
        known_different_pairs_evaluated=len(negative_pairs), false_merge_pairs=len(false_merges),
        false_merge_rate=len(false_merges)/len(negative_pairs) if negative_pairs else None,
        partial_pair_precision=correct/predicted_same_pairs if predicted_same_pairs else None,
        false_merges=false_merges)


def evaluate_partial_identities(old_observations, new_observations, annotation, *, output=None, min_iou=.25, frame_tolerance=0):
    old = _observations(old_observations); new = _observations(new_observations)
    if isinstance(annotation, (str, Path)):
        annotation = read_json(annotation)
    available = {row['entity_id'] for row in old}
    groups, exclusions = _prepare_groups(annotation, available)
    annotated = {eid for group in groups for eid in group['ids']}
    annotated_old = [row for row in old if row['entity_id'] in annotated]
    transfer = match_source_observations(annotated_old, new, min_iou=min_iou, frame_tolerance=frame_tolerance)
    votes, kinds = defaultdict(Counter), defaultdict(Counter)
    for match in transfer['matches']:
        votes[match['old_id']][match['new_id']] += 1
        kinds[match['old_id']][match['kind']] += 1
    primary = {eid: counter.most_common(1)[0][0] for eid, counter in votes.items()}
    all_supported = {eid: sorted(counter) for eid, counter in votes.items()}
    current = _identity_metrics(groups, primary, all_supported)
    baseline = _identity_metrics(groups, {eid: eid for eid in annotated})
    classification = []
    for group in groups:
        expected = group.get('kind')
        if not expected:
            continue
        for eid in group['ids']:
            actual = kinds[eid].most_common(1)[0][0] if kinds[eid] else None
            if actual is not None:
                matches = actual == 'person' if expected == 'person' else actual in ('animal', 'object')
                classification.append(dict(old_id=eid, group=group['name'], expected=expected,
                                           observed=actual, correct=matches))
    signature = hashlib.sha256(json.dumps([dict(id=r.get('detection_id'), entity=r['entity_id'],
        shot=r['shot_id'], frame=r['source_frame'], box=pixel_box(r)) for r in annotated_old],
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = dict(schema_version=1, label_source=annotation.get('source', 'user-provided partial identity groups'),
        annotation_scope='Partial identity labels only; unlisted observations are unknown, never negatives.',
        limitations=['Geometric label transfer can miss changed boxes; transfer coverage is reported separately.',
                     'These labels cannot measure missed detections, IDF1 or HOTA without frame-level box annotation.',
                     'Dominant-ID recall alone can hide identity splits; all_supported_identity_count exposes them.',
                     'Contradictory groups are excluded from cross-group false-merge checks.'],
        exclusions=exclusions, source_observation_fingerprint=signature,
        annotation_ids=len(annotated), mapped_annotation_ids=len(primary),
        annotation_id_coverage=len(primary)/len(annotated) if annotated else None,
        labeled_observations=len(annotated_old), matched_observations=len(transfer['matches']),
        observation_coverage=len(transfer['matches'])/len(annotated_old) if annotated_old else None,
        assignments={eid: dict(primary=primary[eid], votes=dict(votes[eid]), kinds=dict(kinds[eid])) for eid in sorted(primary)},
        baseline=baseline, current=current, classification=classification,
        classification_accuracy=sum(row['correct'] for row in classification)/len(classification) if classification else None,
        transfer=transfer)
    if output is not None:
        write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Old observations.json with manually labeled IDs')
    parser.add_argument('--current', type=Path, required=True, help='New exported observations.json')
    parser.add_argument('--annotations', type=Path, required=True, help='Explicit evaluation-only JSON fixture')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-iou', type=float, default=.25)
    args = parser.parse_args()
    report = evaluate_partial_identities(args.baseline, args.current, args.annotations, output=args.output, min_iou=args.min_iou)
    print(json.dumps(dict(annotation_id_coverage=report['annotation_id_coverage'],
        same_id_pair_recall=report['current']['same_id_pair_recall'], false_merge_pairs=report['current']['false_merge_pairs']), indent=2))


if __name__ == '__main__':
    main()
