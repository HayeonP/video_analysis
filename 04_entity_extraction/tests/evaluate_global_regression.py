#!/usr/bin/env python3
"""Evaluation-only identity gate using frozen source boxes, never model labels.

Run after the global stage or export:
  python tests/evaluate_global_regression.py --run outputs/my_run
Returns nonzero for missing annotated identities, fragmentation, or false merges.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from story_evaluation import evaluate_partial_identities
from story_export import suppress_duplicate_boxes


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path=Path(path)
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path.resolve()),sha256=digest)


def catalog_partition(catalog):
    entities, mapping = {}, {}
    for index, entity in enumerate(catalog['entities'], 1):
        eid = str(entity.get('entity_id', f'P{index:04d}'))
        if eid in entities:
            raise ValueError(f'Duplicate global ID: {eid}')
        entities[eid] = entity
        for lid in entity.get('local_ids', []):
            if str(lid) in mapping:
                raise ValueError(f'Local ID assigned more than once: {lid}')
            mapping[str(lid)] = eid
    if 'mapping' in catalog:
        supplied = {str(lid): str(eid) for lid, eid in catalog['mapping'].items()}
        if mapping and supplied != mapping:
            raise ValueError('Catalog mapping disagrees with entity local_ids.')
        mapping = supplied
    if set(mapping.values()) - set(entities):
        raise ValueError('Mapping references an unknown global ID.')
    return entities, mapping


def source_observations(bundle, local, catalog):
    """Derive source-coordinate identity observations before expensive export."""
    globals_by_id, mapping = catalog_partition(catalog)
    local_by_shot = {shot['shot_id']: shot for shot in local['shots']}
    rejected = set(catalog.get('rejected_local_ids', []))
    rows = []
    for shot in bundle['shots']:
        sid = shot['shot_id']
        if sid not in local_by_shot:
            raise ValueError(f'Unreviewed shot: {sid}')
        owner = {}
        for entity in local_by_shot[sid]['entities']:
            lid = str(entity['local_id'])
            if lid in rejected:
                continue
            if lid not in mapping:
                raise ValueError(f'Local entity lacks a global mapping: {lid}')
            for track_id in entity['track_ids']:
                if str(track_id) in owner:
                    raise ValueError(f'Multiple local owners of track: {sid}/{track_id}')
                owner[str(track_id)] = entity
        for track in shot['tracks']:
            tid = str(track['track_id'])
            if tid not in owner:
                continue
            entity = owner[tid]
            lid = str(entity['local_id']); eid = mapping[lid]
            for observation in track.get('observations', []):
                frame = int(observation.get('frame', observation.get('source_frame', -1)))
                if not 0 <= frame < int(shot['frame_count']):
                    raise ValueError(f'Invalid source frame: {sid}/{frame}')
                box = [max(0., min(float(bound), float(value))) for value, bound in
                       zip(observation['bbox'], [shot['width'], shot['height']] * 2)]
                rows.append(dict(detection_id=f'{sid}_f{frame:08d}_{tid}', shot_id=sid,
                    source_frame=frame, bbox_pixels=box, tracklet_id=tid, local_id=lid, entity_id=eid,
                    kind=globals_by_id[eid].get('kind', entity.get('kind', 'unknown')),
                    confidence=observation.get('confidence', 0)))
    return suppress_duplicate_boxes(rows)[0]


def regression_gate(report):
    issues = []
    if not report['annotation_ids']:
        issues.append('No eligible annotated identities.')
    if report['mapped_annotation_ids'] != report['annotation_ids']:
        issues.append('Some annotated identities have no geometric observation match.')
    if report['exclusions']['excluded_ids']:
        issues.append('Annotations contain excluded, unavailable, or contradictory IDs.')
    for group in report['current']['groups']:
        if group['all_supported_identity_count'] != 1:
            issues.append(f"Identity group {group['name']} has {group['all_supported_identity_count']} supported global IDs.")
    # Check minority assignments too: dominant-vote metrics can hide contamination.
    false_merges = []
    for left, right in combinations(report['current']['groups'], 2):
        shared = sorted(set(left['supported_global_ids']) & set(right['supported_global_ids']))
        if shared:
            false_merges.append(dict(left=left['name'], right=right['name'], global_ids=shared))
    if false_merges:
        issues.append('Distinct annotated identity groups share global IDs.')
    if any(not row['correct'] for row in report['classification']):
        issues.append('An annotated identity has an incorrect person/animal/object kind.')
    return dict(passed=not issues, issues=issues, cross_group_supported_id_conflicts=false_merges)


def evaluate(observations, baseline, labels):
    report = evaluate_partial_identities(baseline, observations, labels)
    return dict(gate=regression_gate(report), evaluation=report)


def evaluate_run(run, baseline, labels, *, final_catalog=None, provisional_catalog=None, observations=None):
    run = Path(run)
    final_catalog = Path(final_catalog) if final_catalog else run / 'global_entities.json'
    provisional_catalog = Path(provisional_catalog) if provisional_catalog else run / 'global_provisional.json'
    config_path=run/'run_config.json'
    config=read(config_path) if config_path.is_file() else {}
    mask_enabled=config.get('mask_refinement') is not False
    paths=(('mask_refinement/bundle.json',) if mask_enabled else ())+('reviewed_tracking.json',)+(
        ('mask_track_graph/bundle.json',) if mask_enabled else ())+('dense_tracking/bundle.json',)
    bundle_path = next((run/path for path in paths if (run/path).is_file()),None)
    if bundle_path is None:
        raise FileNotFoundError('No completed tracking bundle found.')
    bundle = read(bundle_path)
    local_path=run/('mask_refinement/local.json' if bundle_path==run/'mask_refinement/bundle.json' else 'local_entities.json')
    local = read(local_path)
    current = read(observations) if observations else source_observations(bundle, local, read(final_catalog))
    result = dict(scope='Frozen partial source-box identity regression; not exhaustive detection/identity accuracy.',
                  labels=str(labels), baseline=str(baseline), bundle=str(bundle_path),
                  artifacts=dict(final_catalog=artifact(final_catalog),
                      local_entities=artifact(local_path),bundle=artifact(bundle_path),
                      labels=artifact(labels),baseline=artifact(baseline)),
                  final=evaluate(current, baseline, labels))
    if observations:result['artifacts']['exported_observations']=artifact(observations)
    if config_path.is_file():result['artifacts']['run_config']=artifact(config_path)
    if provisional_catalog.is_file():
        provisional = source_observations(bundle, local, read(provisional_catalog))
        result['provisional'] = evaluate(provisional, baseline, labels)
        result['artifacts']['provisional_catalog']=artifact(provisional_catalog)
    result['passed'] = result['final']['gate']['passed']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'tests/fixtures/sintel_global_observations.json')
    parser.add_argument('--labels', type=Path, default=ROOT / 'tests/fixtures/sintel_global_labels.json')
    parser.add_argument('--final-catalog', type=Path)
    parser.add_argument('--provisional-catalog', type=Path)
    parser.add_argument('--observations', type=Path, help='Use exported observations for the final catalog')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate_run(args.run, args.baseline, args.labels, final_catalog=args.final_catalog,
                          provisional_catalog=args.provisional_catalog, observations=args.observations)
    destination = args.output or args.run / 'regression_evaluation.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], final=result['final']['gate'], report=str(destination)), indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
