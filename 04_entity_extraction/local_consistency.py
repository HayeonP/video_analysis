"""Reconcile VLM whole-subject assignments with simultaneous foreground evidence."""
from copy import deepcopy
from itertools import combinations

LEGACY_GEOMETRY_REASON='VLM whole-subject roles plus repeated simultaneous separation:'
GEOMETRY_PROPOSAL_REASON='Geometry proposal only; physical identity remains unconfirmed'


def detector_separations(result, shot, aliases):
    """Repeated co-visible whole-subject boxes propose a visual comparison.

    Only observed detector boxes count. Contained face/body duplicate boxes,
    low-confidence continuations and isolated outliers do not create a split.
    """
    tracks = {}
    for track in shot.get('tracks', []):
        alias = aliases.get(track['track_id'])
        if result.track_roles.get(alias) not in ('whole_subject', 'handled_object'):
            continue
        rows = {}
        for row in track.get('observations', []):
            if row.get('evidence', 'observed_detection') != 'observed_detection':
                continue
            if row.get('tracker_confirmed') is False or float(row.get('confidence') or 0) < .24:
                continue
            box = row.get('bbox', [])
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            rows[int(row.get('frame', row.get('source_frame', -1)))] = box
        tracks[alias] = rows
    evidence = []
    for left, right in combinations(tracks, 2):
        frames = []
        for frame in tracks[left].keys() & tracks[right].keys():
            a, b = tracks[left][frame], tracks[right][frame]
            area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            iou = intersection / max(area_a + area_b - intersection, 1e-9)
            containment = intersection / max(min(area_a, area_b), 1e-9)
            if iou < .2 and containment < .85:
                frames.append(frame)
        if len(frames) >= 3:
            evidence.append(dict(left=left, right=right, frames=sorted(frames),
                                 source='observed whole-subject detector boxes'))
    return evidence


def separate_conflicting_subjects(result,shot,aliases,*,provisional_pairs=()):
    """Expose unresolved geometric regions as freely rejoinable local hypotheses.

    Body parts alone never activate this rule. Geometry and the model's explicit
    whole-subject role are both required. This uses no movie-specific labels.
    """
    from story_semantics import TrackRelation
    result=result.model_copy(deep=True)
    evidence=shot.get('mask_graph',{}).get('separation_frame_evidence',[])
    geometric=set()
    for relation in result.relations:
        if relation.relation=='different_individual' and relation.reason.startswith(LEGACY_GEOMETRY_REASON):
            relation.relation='uncertain';relation.reason=GEOMETRY_PROPOSAL_REASON+'; migrated legacy automatic veto'
        if relation.relation=='uncertain' and relation.reason.startswith(GEOMETRY_PROPOSAL_REASON):
            geometric.add(tuple(sorted((relation.left,relation.right))))
    for item in evidence:
        if len(set(item.get('frames',[])))<2:continue
        a=aliases.get(item['left']);b=aliases.get(item['right'])
        if a and b and a!=b and all(result.track_roles.get(k) in ('whole_subject','handled_object') for k in (a,b)):
            geometric.add(tuple(sorted((a,b))))
    for item in detector_separations(result, shot, aliases):
        geometric.add(tuple(sorted((item['left'], item['right']))))
    conflicts=set()
    for relation in result.relations:
        if relation.relation=='different_individual':
            conflicts.add(tuple(sorted((relation.left,relation.right))))
    verified_same={tuple(sorted((r.left,r.right))) for r in result.relations
        if r.relation in ('same_individual','part_of') and r.reason.startswith('Focused visual audit v3 of original frames ')}
    conflicts-=verified_same
    confirmed_conflicts=set(conflicts)
    provisional_pairs=({tuple(sorted(pair)) for pair in provisional_pairs}|geometric)-verified_same-confirmed_conflicts
    conflicts.update(provisional_pairs)
    for a,b in sorted(geometric-verified_same-confirmed_conflicts):
        if not any(r.relation=='uncertain' and {r.left,r.right}=={a,b} and r.reason.startswith(GEOMETRY_PROPOSAL_REASON) for r in result.relations):
            result.relations.append(TrackRelation(left=a,right=b,relation='uncertain',reason=GEOMETRY_PROPOSAL_REASON))
    used={e.local_id for e in result.entities};entities=[]
    for entity in result.entities:
        local_pairs={pair for pair in conflicts if set(pair)<=set(entity.track_ids)}
        if not local_pairs:entities.append(entity);continue
        groups=[]
        # Most constrained complete subjects first; unconnected fragments remain
        # with the original grouping rather than acquiring invented identities.
        order=sorted(entity.track_ids,key=lambda tid:(-sum(tid in p for p in local_pairs),entity.track_ids.index(tid)))
        for tid in order:
            group=next((g for g in groups if not any(tuple(sorted((tid,other))) in local_pairs for other in g)),None)
            if group is None:groups.append([tid])
            else:group.append(tid)
        groups.sort(key=lambda g:min(entity.track_ids.index(t) for t in g))
        for index,group in enumerate(groups):
            clone=entity.model_copy(deep=True)
            if index:
                serial=1
                while f'L{serial:02d}' in used:serial+=1
                clone.local_id=f'L{serial:02d}';used.add(clone.local_id)
            clone.track_ids=[t for t in entity.track_ids if t in group]
            clone.segments=[s for s in entity.segments if s.track_id in group]
            clone.appearance='; '.join(result.track_descriptions.get(t,'') for t in clone.track_ids) or entity.appearance
            if local_pairs & provisional_pairs:
                clone.uncertain=True
                clone.reason+='; 동일성 미확정 구간을 임시 후보로 분리했으며 전역 검토에서 재결합 가능'
            else:
                clone.reason+='; 명시적 시각 판정 또는 반복된 동시 분리 관측에 따라 병합 모순을 보정'
            entities.append(clone)
            for tid in clone.track_ids:
                if result.track_assignments:result.track_assignments[tid]=clone.local_id
    result.entities=entities
    return result
