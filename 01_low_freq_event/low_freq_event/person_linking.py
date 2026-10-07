"""Offline person identities above local BoT-SORT track IDs.

No face is required. Independent appearances must support a join, co-visible
people can never merge, and similarly plausible competitors keep IDs separate.
All accepted joins and unresolved alternatives are exported for review.
"""
from itertools import combinations
from pathlib import Path

import numpy as np

from .detection import select_references
from .motion import camera_motion


def reliable_samples(samples, settings):
    result = []
    for sample in samples:
        x1,y1,x2,y2 = sample['bbox']
        width,height = sample.get('source_size',(1000,1000))
        aspect = (x2-x1)*width/max((y2-y1)*height,1.)
        clipped_fragment = (x1<=2 or x2>=998) and aspect<.20
        if sample.get('detection_confidence',0)>=settings.person_confidence and not clipped_fragment:
            result.append(sample)
    body_gallery=select_references(result, limit=16)
    faces=[sample for sample in result if sample.get('face_feature') and sample.get('face_evidence',{}).get('usable')]
    # Body diversity must not discard the useful face views. Sample face-space
    # separately and keep original observations, not substituted body vectors.
    face_views=select_references([dict(sample,body_feature=sample['face_feature']) for sample in faces],limit=8)
    selected={sample['detection_id'] for sample in body_gallery+face_views}
    return [sample for sample in result if sample['detection_id'] in selected]


def appearance_evidence(left, right, settings):
    """Multiple distinct observations on BOTH sides, not one lucky cosine pair."""
    a = [s for s in left if s.get('body_feature')]
    b = [s for s in right if s.get('body_feature')]
    body = -1.
    if a and b:
        similarities = np.clip(np.array([s['body_feature'] for s in a]) @ np.array([s['body_feature'] for s in b]).T,-1,1)
        count = min(3,len(a),len(b))
        body = float(min(np.sort(similarities.max(axis=1))[-count:].mean(),
                         np.sort(similarities.max(axis=0))[-count:].mean()))
    af = [s for s in left if s.get('face_feature') and s.get('face_evidence',{}).get('usable',False)]
    bf = [s for s in right if s.get('face_feature') and s.get('face_evidence',{}).get('usable',False)]
    face=-1.
    if af and bf:
        similarities=np.clip(np.array([s['face_feature'] for s in af]) @ np.array([s['face_feature'] for s in bf]).T,-1,1)
        count=min(2,len(af),len(bf))
        face=float(min(np.sort(similarities.max(axis=1))[-count:].mean(),np.sort(similarities.max(axis=0))[-count:].mean()))
    conflict = bool(af and bf and max(s['face_quality'] for s in af)>=settings.face_conflict_quality
                    and max(s['face_quality'] for s in bf)>=settings.face_conflict_quality
                    and max((float(np.dot(x['face_feature'],y['face_feature'])) for x in af for y in bf
                             if min(x['face_quality'],y['face_quality'])>=settings.face_conflict_quality),default=1.)<.25)
    # Put positive face evidence onto the same [0,1] decision scale as body cosine.
    face_score = .90 + .10*(face-settings.face_similarity)/(1-settings.face_similarity) if face>=settings.face_similarity else -1.
    supported = min(len(a),len(b))>=2
    score = max(body if supported else -1.,face_score if min(len(af),len(bf))>=2 else -1.)
    return dict(score=score if not conflict else -1.,body_similarity=body,face_similarity=face,
                face_supported=min(len(af),len(bf))>=2,face_conflict=conflict,support_left=len(a),support_right=len(b),method='multiview_reid')


def spatial_evidence(left, right, output, cache):
    """Short-gap endpoint geometry; never assume image position survives a cut."""
    a,b = sorted((left,right),key=lambda ss:min(s['time_sec'] for s in ss))
    previous = max(a,key=lambda s:s['time_sec']); current = min(b,key=lambda s:s['time_sec'])
    if current.get('identity_boundary')=='crowded_occlusion_recovery':
        return None
    gap = current['time_sec']-previous['time_sec']
    if not 0 < gap <= .4+1e-6:
        return None
    pair = previous['frame_key'],current['frame_key']
    if pair not in cache:
        def path(s):
            for folder in ('frames','tracking/frames'):
                candidate=Path(output)/folder/(s['frame_key']+'.jpg')
                if candidate.exists(): return candidate
            return Path(output)/'missing.jpg'
        # Estimate on the background, masking all people in these frames.
        boxes = cache.get('boxes',{})
        cache[pair] = camera_motion(path(previous),path(current),boxes.get(pair[0],[]),boxes.get(pair[1],[]))
    motion = cache[pair]
    if motion.get('valid'):
        matrix=np.array(motion['matrix'])
    elif previous.get('shot_segment')==current.get('shot_segment'):
        matrix=np.array([[1.,0.,0.],[0.,1.,0.]])
    else:
        return None
    x1,y1,x2,y2 = previous['bbox']
    corners=np.array([[x1,y1,1],[x2,y1,1],[x1,y2,1],[x2,y2,1]]) @ matrix.T
    lo,hi=corners.min(axis=0),corners.max(axis=0)
    qlo,qhi=np.array(current['bbox'][:2]),np.array(current['bbox'][2:])
    area=float(np.prod(hi-lo)); qarea=float(np.prod(qhi-qlo))
    if min(area,qarea)<=0: return None
    overlap=float(np.prod(np.maximum(0,np.minimum(hi,qhi)-np.maximum(lo,qlo))))
    iou=overlap/(area+qarea-overlap)
    displacement=float(np.linalg.norm((qlo+qhi-lo-hi)/2))/np.sqrt(area)
    body=float(np.dot(previous['body_feature'],current['body_feature']))
    if iou<.30 or displacement>.45 or not .5<=qarea/area<=2.0 or body<.60:
        return None
    return dict(score=.78+.12*iou+.10*max(0.,(body-.6)/.4),method='camera_motion_continuity',
                iou=iou,gap_sec=gap,body_similarity=body,camera_compensated=bool(motion.get('valid')))


def joint_matching(candidates, settings):
    """Resolve coupled ambiguities with a maximum-weight disjoint matching.

    An edge is accepted only if forbidding it reduces the GLOBAL objective by
    the required margin. This prevents two returnees from claiming one person.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    edges=[p for p in candidates if p['score']>=settings.person_relink_similarity]
    if not edges: return None
    keys=sorted({p[k] for p in edges for k in ('left','right')})
    rows={key:i for i,key in enumerate(keys)}
    incidence=np.zeros((len(keys),len(edges)))
    for j,p in enumerate(edges):
        incidence[rows[p['left']],j]=incidence[rows[p['right']],j]=1.
    weights=np.array([p['score']-settings.person_relink_similarity for p in edges])
    constraints=LinearConstraint(incidence,0,1)
    def solve(upper):
        return milp(-weights,integrality=np.ones(len(edges)),bounds=Bounds(0,upper),constraints=constraints)
    optimum=solve(np.ones(len(edges)))
    if not optimum.success: return None
    for index in np.flatnonzero(optimum.x>.5):
        upper=np.ones(len(edges));upper[index]=0
        alternative=solve(upper)
        if not alternative.success: continue
        margin=float(alternative.fun-optimum.fun)
        if margin>=settings.person_relink_margin:
            return dict(edges[index],competitor_margin=margin,decision='global_one_to_one_matching')
    return None


def reconcile_people(registry, visual, history, settings, output):
    tracks = visual['tracks']; assignments = visual['assignments']
    initial_count=len(history); joins=[]; cache={'boxes':{}}
    for samples in history.values():
        for sample in samples:
            cache['boxes'].setdefault(sample['frame_key'],[]).append(sample['bbox'])
    # Global IDs survive every local reset. Galleries are rebuilt from reliable
    # observations; faint detector continuations never become appearance anchors.
    galleries={eid:reliable_samples(samples,settings) for eid,samples in history.items()}

    def coexist(a,b): return bool(set(tracks[a]['frame_keys']) & set(tracks[b]['frame_keys']))

    def proposals():
        result=[]
        for a,b in combinations(history,2):
            if coexist(a,b) or not galleries[a] or not galleries[b]: continue
            proof=appearance_evidence(galleries[a],galleries[b],settings)
            if proof['face_conflict']: continue
            spatial=spatial_evidence(history[a],history[b],output,cache)
            if spatial and spatial['score']>proof['score']:
                proof=dict(proof,**spatial)
            result.append(dict(left=a,right=b,**proof))
        return result

    while True:
        candidates=proposals(); accepted=None
        for candidate in sorted(candidates,key=lambda p:p['score'],reverse=True):
            if candidate['score']<settings.person_relink_similarity: continue
            a,b=candidate['left'],candidate['right']
            # A competitor matters if it shares an endpoint and cannot itself
            # belong to the other endpoint (because both appear simultaneously).
            rivals=[]; face_rivals=[]
            for other in candidates:
                x,y=other['left'],other['right']
                if (x,y)==(a,b): continue
                common={a,b}&{x,y}
                if not common: continue
                anchor=next(iter(common)); peer=b if anchor==a else a
                rival=y if anchor==x else x
                if coexist(peer,rival):
                    rivals.append(other['score'])
                    if other.get('face_supported'): face_rivals.append(other['face_similarity'])
            margin=candidate['score']-max(rivals,default=-1.)
            face_margin=candidate['face_similarity']-max(face_rivals,default=-1.)
            strong_face=candidate.get('face_supported') and candidate['face_similarity']>=settings.face_similarity
            # Face cosine has its own margin; a compressed display score must
            # not turn a clear face winner into a permanently ambiguous pair.
            if strong_face and face_rivals and face_margin<0: continue
            if margin < settings.person_relink_margin and not (strong_face and face_margin>=.08): continue
            accepted=dict(candidate,competitor_margin=margin,face_competitor_margin=face_margin);break
        if accepted is None: accepted=joint_matching(candidates,settings)
        if accepted is None: break
        a,b=accepted['left'],accepted['right']
        target,old=sorted((a,b),key=lambda eid:(registry[eid]['first_seen_sec'],eid))
        joins.append(dict(accepted,target=target,retired=old))
        current,previous=registry[target],registry[old]
        current['first_seen_sec']=min(current['first_seen_sec'],previous['first_seen_sec'])
        current['last_seen_sec']=max(current['last_seen_sec'],previous['last_seen_sec'])
        current['identity_status']='matched';previous['alias_of']=target
        history[target]+=history.pop(old)
        galleries[target]=reliable_samples(history[target],settings);galleries.pop(old)
        current['gallery']=list(dict.fromkeys(s['crop'] for s in galleries[target]))[:3]
        tracks[target]['frame_keys']=sorted(set(tracks[target]['frame_keys'])|set(tracks[old]['frame_keys']))
        tracks[target]['samples']=galleries[target];del tracks[old]
        for assignment in assignments.values():
            if assignment['entity_id'] in (target,old):
                assignment.update(entity_id=target,identity_merge_method=accepted['method'],
                                  identity_uncertain=False,possible_matches=[])
    unresolved=sorted(proposals(),key=lambda p:p['score'],reverse=True)
    for candidate in unresolved:
        if candidate['score']<settings.person_relink_similarity: continue
        for eid,peer in ((candidate['left'],candidate['right']),(candidate['right'],candidate['left'])):
            registry[eid]['possible_matches']=list(dict.fromkeys(registry[eid].get('possible_matches',[])+[peer]))
    return dict(initial_person_tracklets=initial_count,final_person_ids=len(history),
                joins=joins,unresolved_candidates=unresolved,
                face_observations_used=sum(bool(s.get('face_feature')) for ss in history.values() for s in ss),
                note='ID count is not accuracy. Unresolved competing matches remain separate.')
