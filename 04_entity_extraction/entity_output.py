"""Representative images, entity catalog, and a full-length annotated VFR video."""
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
from pathlib import Path
import shutil

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from low_freq_event.identity import canonical
from low_freq_event.storage import atomic_json
from media import VideoWriter

ROOT=Path(__file__).resolve().parent


def font(size):
    path=ROOT/'assets/NanumGothic-Regular.ttf'
    if not path.exists(): raise FileNotFoundError(f'한국어 폰트가 필요합니다: {path}')
    return ImageFont.truetype(str(path),size)


def color(eid):
    digest=hashlib.sha256(eid.encode()).digest()
    return tuple(90+int(c)%166 for c in digest[:3])


def representative_score(obs,output):
    image=Image.open(output/obs['crop']).convert('L')
    pixels=np.asarray(image.resize((128,128)))
    sharpness=float(cv2.Laplacian(pixels,cv2.CV_64F).var())
    visible=min(1.,float(pixels.mean())/55.)
    size=np.sqrt(image.width*image.height)
    score=size*(1+min(sharpness,800)/800)*visible
    if obs.get('detection_uncertain'): score*=.4
    x1,y1,x2,y2=obs['bbox']
    if min(x1,y1)<2 or max(x2,y2)>998: score*=.8
    return score


def export_results(output,frames,timeline,raw_all,registry,visual,*,mode='with_qwen'):
    by_entity=defaultdict(list);by_frame=defaultdict(list);clean=[]
    for raw in raw_all:
        proof=visual['assignments'][raw['detection_id']]
        eid=canonical(registry,proof['entity_id'])
        obs={k:v for k,v in raw.items() if not k.endswith('_feature')}
        obs.update(proof,entity_id=eid)
        obs['kind']=raw['entity_kind']
        obs['bbox_format']='xyxy normalized 0..1000'
        clean.append(obs);by_entity[eid].append(obs);by_frame[obs['frame_key']].append(obs)
    for values in by_frame.values():
        ids=[o['entity_id'] for o in values]
        if len(set(ids))!=len(ids): raise RuntimeError('같은 프레임의 서로 다른 관찰이 하나의 ID로 병합되었습니다.')
    entities=[];tiles=[]
    label_font=font(18)
    for eid,observed in sorted(by_entity.items()):
        ranked=sorted(observed,key=lambda o:representative_score(o,output),reverse=True)
        best=ranked[0];label=best['label']
        folder=output/'entities'/eid;folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(output/best['crop'],folder/'representative.jpg')
        gallery=[]
        used_shots=set()
        for o in ranked:
            if o['shot_id'] in used_shots: continue
            name=f'view_{len(gallery)+1:02d}.jpg';shutil.copyfile(output/o['crop'],folder/name)
            gallery.append(f'entities/{eid}/{name}');used_shots.add(o['shot_id'])
            if len(gallery)>=6: break
        aliases=sorted(k for k in registry if k!=eid and canonical(registry,k)==eid)
        possible=sorted({canonical(registry,p) for o in observed for p in o.get('possible_matches',[]) if canonical(registry,p)!=eid})
        intervals=[]
        fps=Fraction(timeline['fps'])
        for o in observed:
            intervals.append(dict(shot_id=o['shot_id'],original_start_frame=o['original_frame'],
                original_end_frame=o['original_frame']+o['hold_frames'],
                original_start_sec=o['original_time_sec'],original_end_sec=o['original_time_sec']+float(o['hold_frames']/fps),
                full_start_sec=o['full_time_sec'],full_end_sec=o['full_time_sec']+float(o['hold_frames']/fps)))
        entity=dict(entity_id=eid,label=label,kind=Counter(o['kind'] for o in observed).most_common(1)[0][0],
            identity_status=registry[eid].get('identity_status','estimated'),reference_key=registry[eid].get('reference_key'),
            appearance=best['appearance'],label_variants=list(dict.fromkeys(o['label'] for o in observed)),
            representative=f'entities/{eid}/representative.jpg',gallery=gallery,
            representative_observation_id=best['detection_id'],merged_ids=aliases,
            color_rgb=list(color(eid)),observations=len(observed),shots=sorted({o['shot_id'] for o in observed}),
            first_seen_sec=min(o['original_time_sec'] for o in observed),
            last_seen_sec=max(o['original_time_sec'] for o in observed),possible_matches=possible,
            uncertain_observations=sum(o.get('detection_uncertain',False) for o in observed),
            display_intervals=intervals,
            interval_note='Display intervals follow held adaptive images; they do not establish continuous visibility in omitted source frames.')
        entities.append(entity);atomic_json(folder/'entity.json',entity)
        tile=Image.new('RGB',(360,310),(24,26,30))
        crop=ImageOps.contain(Image.open(folder/'representative.jpg').convert('RGB'),(344,230))
        tile.paste(crop,((360-crop.width)//2,8+(230-crop.height)//2))
        draw=ImageDraw.Draw(tile);draw.text((8,243),eid,fill=color(eid),font=label_font)
        draw.text((8,267),label[:18],fill='white',font=label_font)
        tiles.append(tile)
    gallery=Image.new('RGB',(360*4,310*max(1,(len(tiles)+3)//4)),(24,26,30))
    for i,tile in enumerate(tiles): gallery.paste(tile,((i%4)*360,(i//4)*310))
    gallery.save(output/'entity_gallery.jpg',quality=94)
    atomic_json(output/'entities.json',dict(entities=entities,entity_count=len(entities),mode=mode,
        identification='01 YOLO11s + SFace/OSNet-AIN/DINOv2' +
            (' + Qwen3-VL foreground grounding and gallery review' if mode=='with_qwen' else '') +
            (' + temporal scene review' if (output/'temporal_review').exists() else '') +
            (' + sequential memory / independent reverse / final global audit' if (output/'sequential').exists() else ''),
        note='Automatic visual identity estimates; uncertain matches remain separate. See per-observation evidence.'))
    shot_ids=[s['id'] for s in timeline['shots']]
    presence=[dict(entity_id=e['entity_id'],label=e['label'],
        **{sid:int(sid in e['shots']) for sid in shot_ids}) for e in entities]
    with (output/'entity_shot_presence.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['entity_id','label',*shot_ids])
        writer.writeheader();writer.writerows(presence)
    atomic_json(output/'entity_shot_presence.json',dict(shots=shot_ids,rows=presence,
        semantics={'1':'Observed in at least one adaptive sample of this shot',
                   '0':'Not confirmed in analyzed samples; not proof of absence in unsampled source frames'}))
    atomic_json(output/'shot_entities.json',[dict(shot_id=sid,entities=[dict(entity_id=e['entity_id'],
        label=e['label'],representative=e['representative'],
        observation_ids=[o['detection_id'] for o in by_entity[e['entity_id']] if o['shot_id']==sid])
        for e in entities if sid in e['shots']]) for sid in shot_ids])
    atomic_json(output/'observations.json',dict(bbox_format='xyxy normalized 0..1000',observations=clean))
    annotated_frames=[dict(f,observations=by_frame[f['frame_key']]) for f in frames]
    atomic_json(output/'frame_annotations.json',annotated_frames)
    labels={e['entity_id']:e['label'] for e in entities}
    fps=Fraction(timeline['fps']);width,height=timeline['size']
    holds={f['full_frame']:f['hold_frames'] for f in frames}
    pending=output/'.adaptive_entities_full.mp4'
    writer=VideoWriter(pending,width,height,fps,holds,crf=18)
    box_font=font(max(18,round(width/65)));header_font=font(max(16,round(width/85)))
    try:
        for f in frames:
            image=Image.open(output/f['image']).convert('RGB');draw=ImageDraw.Draw(image)
            for o in by_frame[f['frame_key']]:
                eid=o['entity_id'];c=color(eid)
                x1,y1,x2,y2=np.array(o['bbox'])*np.array([width,height]*2)/1000
                draw.rectangle((x1,y1,x2,y2),outline=c,width=max(2,width//450))
                text=f'{eid} | {labels[eid]}' + (' ?' if o.get('detection_uncertain') else '')
                tw=draw.textbbox((0,0),text,font=box_font)[2]+12
                tx=max(0,min(int(x1),width-tw));ty=max(45,int(y1)-box_font.size-12)
                draw.rectangle((tx,ty,tx+tw,ty+box_font.size+10),fill=(15,18,24))
                draw.text((tx+5,ty+3),text,font=box_font,fill=c)
            header=f'{mode} | {f["shot_id"]} | original sample {f["original_time_sec"]:.3f}s | full {f["full_time_sec"]:.3f}s | entities {len(by_frame[f["frame_key"]])}'
            draw.rectangle((0,0,width,40),fill=(15,18,24));draw.text((10,7),header,font=header_font,fill='white')
            writer.write(f['full_frame'],np.asarray(image)[:,:,::-1].copy())
    finally: writer.close()
    with av.open(str(pending)) as container:
        stream=container.streams.video[0];duration=stream.duration*stream.time_base
        stamps=[f.pts*f.time_base for f in container.decode(stream)]
    expected=[Fraction(f['full_frame'],1)/fps for f in frames]
    if stamps!=expected or duration!=Fraction(timeline['total_frames'],1)/fps:
        raise RuntimeError('시각화 영상 PTS/길이가 입력 타임라인과 다릅니다.')
    pending.replace(output/'adaptive_entities_full.mp4')
    validation=dict(passed=True,samples=len(frames),video_frames=len(stamps),
        original_total_frames=timeline['total_frames'],duration_sec=float(duration),
        exact_sample_timestamps=True,duration_error_sec=0,audio=False,
        hold_intervals_cover_full_timeline=sum(holds.values())==timeline['total_frames'],
        distinct_entities_per_frame=True)
    atomic_json(output/'video_validation.json',validation)
    return dict(mode=mode,entity_count=len(entities),observations=len(clean),sampled_frames=len(frames),
        video='adaptive_entities_full.mp4',duration_sec=float(duration),catalog='entities.json',gallery='entity_gallery.jpg',
        presence='entity_shot_presence.csv',shot_entities='shot_entities.json')
