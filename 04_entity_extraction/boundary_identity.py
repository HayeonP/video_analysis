"""Actual adjacent-shot boundary images propose identity comparisons only."""
from pathlib import Path
import numpy as np
from PIL import Image


def _asset(bundle,output,value):
    if not value:return None
    path=Path(value)
    path=path if path.is_absolute() else Path(bundle.get('asset_root',output))/path
    return path if path.is_file() else None


def _pixels(path,size=(96,64)):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize(size),dtype=np.float32)/255.


def _similarity(left,right):
    # Bounded pixel agreement, deliberately not an identity embedding. The VLM
    # must inspect actual source evidence before accepting a proposed match.
    return float(np.clip(1-np.mean(np.abs(left-right)),0,1))


def boundary_candidates(bundle,entries,output,*,allow_unconfirmed=False):
    """Return compatible observed subjects at consecutive shot endpoints.

    Both full source images and crops must already exist for observed detections
    within two frames of the endpoints. No interpolation or labels are used.
    """
    from shot_tracking import box_iou
    shots=bundle['shots'];cache={};result=[]
    owners={tid:(lid,entry) for lid,entry in entries.items() for tid in entry['track_ids']}
    def endpoints(shot,side):
        rows=[];limit=int(shot['frame_count'])
        for track in shot['tracks']:
            if track['track_id'] not in owners:continue
            lid,entry=owners[track['track_id']]
            if entry.get('_part_or_worn'):continue
            for row in track['observations']:
                frame=int(row['frame'])
                if not (frame<=2 if side=='start' else frame>=limit-3):continue
                if row.get('evidence')!='observed_detection':continue
                if row.get('tracker_confirmed') is False and (not allow_unconfirmed or row.get('confidence',0)<.6):continue
                if row.get('confidence',0)<.24:continue
                image=_asset(bundle,output,row.get('image'));crop=_asset(bundle,output,row.get('crop'))
                if not image or not crop:continue
                try:
                    with Image.open(image) as source:
                        if source.size!=(shot['width'],shot['height']):continue
                    for path in (image,crop):
                        if path not in cache:cache[path]=_pixels(path)
                except (OSError,ValueError):continue
                if float(cache[image].std())<.04:continue
                bounds=[shot['width'],shot['height']]*2
                box=[float(x)/bound for x,bound in zip(row['bbox'],bounds)]
                time=row.get('full_time_sec',row.get('original_time_sec'))
                if time is None:time=shot.get('full_start_sec',0)+frame/float(shot['fps'])
                rows.append(dict(local_id=lid,kind=entry.get('kind','unknown'),frame=frame,
                    time_sec=float(time),bbox=box,image=image,crop=crop,track_id=track['track_id']))
        return rows
    for left_shot,right_shot in zip(shots,shots[1:]):
        left_rows=endpoints(left_shot,'end');right_rows=endpoints(right_shot,'start');best={}
        for left in left_rows:
            for right in right_rows:
                if left['kind']!=right['kind'] and 'unknown' not in (left['kind'],right['kind']):continue
                gap=right['time_sec']-left['time_sec']
                if not 0<gap<=max(.25,3/min(left_shot['fps'],right_shot['fps'])):continue
                overlap=box_iou(left['bbox'],right['bbox'])
                if overlap<.5:continue
                scene=_similarity(cache[left['image']],cache[right['image']])
                crop=_similarity(cache[left['crop']],cache[right['crop']])
                if scene<.90 or crop<.75:continue
                pair=(left['local_id'],right['local_id']);rank=overlap*scene*crop
                item=dict(left_local_id=pair[0],right_local_id=pair[1],
                    left_frame=left['frame'],right_frame=right['frame'],
                    left_shot_id=left_shot['shot_id'],right_shot_id=right_shot['shot_id'],
                    left_time_sec=left['time_sec'],right_time_sec=right['time_sec'],
                    bbox_iou=overlap,scene_similarity=scene,crop_similarity=crop,
                    left_track_id=left['track_id'],right_track_id=right['track_id'],
                    left_image=str(left['image']),right_image=str(right['image']),
                    left_crop=str(left['crop']),right_crop=str(right['crop']),
                    status='proposal_requires_visual_identity_review',rank=rank)
                if pair not in best or rank>best[pair]['rank']:best[pair]=item
        result.extend(sorted(best.values(),key=lambda r:-r['rank'])[:4])
    return sorted(result,key=lambda r:-r['rank'])
