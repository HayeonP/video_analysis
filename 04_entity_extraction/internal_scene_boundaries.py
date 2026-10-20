"""Conservative tracking barriers inside supplied clips; source timing is unchanged."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION=2


def cuts_from_signals(rows):
    cuts=set();black=[];start=None
    for row in rows:
        i=row['frame'];dark=row['max_luma']<=16
        if dark and start is None:start=i
        if not dark and start is not None:
            if i-start>=2:black.append([start,i])
            start=None
        if i and row['content']>=20 and row['adaptive_ratio']>=2 and row['hist_distance']>=.45:
            cuts.add(i)
    if start is not None and len(rows)-start>=2:black.append([start,len(rows)])
    for a,b in black:cuts.update([a,b])
    return sorted(x for x in cuts if 0<x<len(rows)),black


def scan_video(path):
    import av
    with av.open(str(path)) as container:
        return scan_images(frame.to_ndarray(format='rgb24') for frame in container.decode(video=0))


def scan_images(images):
    import cv2
    maxima=[];hsv=[];hists=[]
    for rgb in images:
        maxima.append(int(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).max()))
        small=cv2.resize(rgb,(160,90),interpolation=cv2.INTER_AREA)
        value=cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
        hsv.append(value.astype(np.float32))
        hist=cv2.calcHist([value],[0,1],None,[16,8],[0,180,0,256])
        hists.append(cv2.normalize(hist,hist).flatten())
    content=np.r_[0,np.abs(np.diff(np.asarray(hsv),axis=0)).mean(axis=(1,2,3))]
    distance=[0]+[cv2.compareHist(hists[i-1],hists[i],cv2.HISTCMP_BHATTACHARYYA) for i in range(1,len(hists))]
    rows=[]
    for i,value in enumerate(content):
        near=[content[j] for j in range(max(1,i-2),min(len(content),i+3)) if j!=i]
        rows.append(dict(frame=i,max_luma=maxima[i],content=float(value),hist_distance=float(distance[i]),
                         adaptive_ratio=float(value/max(.01,np.mean(near))) if near else 0))
    return rows


def apply_internal_barriers(evidence,dense,output):
    """Split hypotheses at observed edits; no SAM-only propagation across an edit.

    A temporal segment needs original detector support for that raw hypothesis.
    All original observations remain, including explicit fallback detections.
    Distinct temporal fragments may later share a semantic Entity identity.
    """
    if evidence.get('internal_boundaries',{}).get('version')==VERSION:return evidence
    output=Path(output);folder=output/'internal_boundaries';folder.mkdir(parents=True,exist_ok=True)
    refined=deepcopy(evidence);source={s['shot_id']:s for s in dense['shots']};records=[]
    from shot_tracking import select_representatives
    from story_semantics import ensure_representatives
    for shot in refined['shots']:
        key=hashlib.sha256((str(VERSION)+source[shot['shot_id']].get('source_sha256',shot['source'])).encode()).hexdigest()
        cache=folder/(key+'.json')
        if cache.exists():signals=json.loads(cache.read_text())
        else:
            if shot.get('analysis_frames') == 'sampled':
                from sampled_frames import iter_shot_images
                signals=scan_images(np.asarray(im) for _,im in iter_shot_images(shot))
            else:signals=scan_video(shot['source'])
            cache.write_text(json.dumps(signals))
        sampled=shot.get('analysis_frames') == 'sampled'
        expected=len(shot['adaptive_frames']) if sampled else shot['frame_count']
        if len(signals)!=expected:raise ValueError('Boundary scan frame count differs from source')
        cuts,black=cuts_from_signals(signals)
        if sampled:
            frames=shot['adaptive_frames']
            cuts=[frames[i] for i in cuts]
            black=[[frames[a],frames[b] if b<len(frames) else shot['frame_count']] for a,b in black]
        edges=[0]+cuts+[shot['frame_count']]
        spans=list(zip(edges,edges[1:]));original={t['track_id']:t for t in source[shot['shot_id']]['tracks']}
        tracks=[];dropped=[]
        for track in shot['tracks']:
            support={o['frame'] for tid in track.get('source_track_ids',[track['track_id']])
                     for o in original.get(tid,{}).get('observations',[])}
            if not support:raise ValueError('Raw hypothesis lacks original detector provenance')
            children=[]
            for index,(a,b) in enumerate(spans):
                observations=[o for o in track['observations'] if a<=o['frame']<b]
                if not observations:continue
                if not any(a<=f<b for f in support):
                    dropped.extend(dict(track_id=track['track_id'],frame=o['frame'],reason='no_original_detection_in_edit_segment') for o in observations)
                    continue
                child=deepcopy(track);child['observations']=observations
                child['representatives']=[o for o in track.get('representatives',[]) if a<=o['frame']<b]
                occupied=sum(any(a<=o['frame']<b for o in track['observations']) for a,b in spans)
                if occupied>1:
                    child['track_id']=track['track_id']+f':B{index+1:03d}'
                    child['internal_boundary_parent']=track['track_id']
                    child['temporal_candidate']=dict(first_frame=a,last_frame=b-1,same_identity_allowed=True,
                        reason='source hard cut or near-black interval, not a claimed identity change')
                child['first_frame']=observations[0]['frame'];child['last_frame']=observations[-1]['frame']
                child['observed_frames']=len(observations)
                child['observed_duration_sec']=len(observations)/shot['fps']
                child['span_duration_sec']=(observations[-1]['frame']-observations[0]['frame']+1)/shot['fps']
                child['observation_coverage']=len(observations)/(observations[-1]['frame']-observations[0]['frame']+1)
                for row in child['observations']+child['representatives']:row['track_id']=child['track_id']
                ensure_representatives(refined,shot,child,output)
                child['representatives']=select_representatives(child['representatives'],8)
                children.append(child)
            retained={o['frame'] for t in children for o in t['observations']}
            if not support<=retained:raise ValueError('Boundary filtering lost an original detector observation')
            tracks.extend(children)
        record=dict(shot_id=shot['shot_id'],cut_frames=cuts,black_intervals=black,
                    before_tracks=len(shot['tracks']),after_tracks=len(tracks),removed_sam_observations=dropped)
        shot['tracks']=tracks;shot['internal_boundaries']=record;records.append(record)
    summary=dict(version=VERSION,shots=records,removed_sam_observations=sum(len(r['removed_sam_observations']) for r in records),
        note='Conservative heuristic barriers; gradual nonblack dissolves may remain. Original frames and detector observations are preserved.')
    refined['internal_boundaries']=summary
    refined['fingerprint']=hashlib.sha256(json.dumps([evidence['fingerprint'],summary],sort_keys=True).encode()).hexdigest()
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (folder/'bundle.json').write_text(json.dumps(refined,ensure_ascii=False))
    return refined
