"""Reconcile VLM whole-subject assignments with simultaneous foreground evidence."""
from copy import deepcopy


def separate_conflicting_subjects(result,shot,aliases):
    """Separate whole subjects when masks repeatedly show distinct regions.

    Body parts alone never activate this rule. Geometry and the model's explicit
    whole-subject role are both required. This uses no movie-specific labels.
    """
    from story_semantics import TrackRelation
    result=result.model_copy(deep=True)
    evidence=shot.get('mask_graph',{}).get('separation_frame_evidence',[])
    conflicts=set()
    for item in evidence:
        if len(set(item.get('frames',[])))<2:continue
        a=aliases.get(item['left']);b=aliases.get(item['right'])
        if a and b and a!=b and all(result.track_roles.get(k) in ('whole_subject','handled_object') for k in (a,b)):
            conflicts.add(tuple(sorted((a,b))))
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
            clone.reason+='; 완전한 대상이라는 VLM 판정과 반복된 동시 분리 마스크에 따라 병합 모순을 보정'
            entities.append(clone)
            for tid in clone.track_ids:
                if result.track_assignments:result.track_assignments[tid]=clone.local_id
        for a,b in sorted(local_pairs):
            result.relations.append(TrackRelation(left=a,right=b,relation='different_individual',
                reason='VLM whole-subject role plus at least two synchronous separated foreground observations'))
    result.entities=entities
    return result
