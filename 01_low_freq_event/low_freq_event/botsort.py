"""Ultralytics BoT-SORT with cached OSNet/DINOv2 embeddings and dense source frames.

The library owns Kalman prediction, GMC, lost-track recovery and association.
This adapter preserves detection indices, separates classes and exports global IDs.
"""
from copy import deepcopy
import logging
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np

from .detection import ObjectEncoder, reconcile_tracklets, select_references
from .storage import atomic_json, read_json
from .person_linking import reconcile_people
from .motion import camera_motion

LOG = logging.getLogger(__name__)


def tracking_frames(source, fps, samples):
    targets = sorted(set(round(float(t), 9) for t in np.arange(0, source.duration-1e-9, 1/fps))
                     | {s['time_sec'] for s in samples})
    index = 0
    with av.open(str(source.path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time = float(frame.pts * stream.time_base) - source.origin
            if index >= len(targets):
                break
            if time + 1e-7 < targets[index]:
                continue
            while index < len(targets) and targets[index] <= time + 1e-7:
                index += 1
            yield time, frame.to_image().convert('RGB')


class DetectionBatch:
    """Carry original indices through BoT-SORT's high/low score slicing."""
    def __init__(self, boxes, indices=None, observations=None):
        self.boxes = boxes
        self.observations = observations
        self.indices = np.arange(len(boxes)) if indices is None else indices

    def __len__(self): return len(self.boxes)
    def __getitem__(self, index): return DetectionBatch(self.boxes[index], self.indices[index], self.observations)
    @property
    def xyxy(self): return self.boxes.xyxy
    @property
    def xywh(self): return self.boxes.xywh
    @property
    def conf(self): return self.boxes.conf
    @property
    def cls(self): return self.boxes.cls


def new_tracker(settings, fps, kind):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.basetrack import BaseTrack
    class IndexedBoTSORT(BOTSORT):
        def init_track(self, results, img=None):
            tracks = super().init_track(results, img)
            for track, index in zip(tracks, results.indices):
                track.idx = int(index)
                if results.observations is not None:
                    observation=results.observations[int(index)]
                    track.face_feature=observation.get('face_feature')
                    track.face_usable=observation.get('face_evidence',{}).get('usable',False)
            return tracks

        def get_dists(self, tracks, detections):
            if kind != 'person' or not tracks or not detections:
                return super().get_dists(tracks, detections)
            from ultralytics.trackers.utils import matching
            geometry = matching.iou_distance(tracks, detections)
            # Geometry is already Kalman-predicted and GMC-corrected. Unlike min(IoU,
            # embedding), the combined cost keeps clothing useful at crossings.
            appearance = matching.embedding_distance(tracks, detections)
            costs = .55*geometry + .45*appearance
            impossible = (geometry > .95) | ((appearance > .55) & (geometry > .45))
            costs[impossible] = 1.
            # Good faces are optional anchors, never a requirement for continuity.
            # Use a clear best candidate to disambiguate people in a crossing.
            for j,detection in enumerate(detections):
                if not getattr(detection,'face_usable',False): continue
                feature=getattr(detection,'face_feature',None)
                if feature is None: continue
                similarities=[]
                for i,track in enumerate(tracks):
                    refs=self.person_face_anchors.get(track.track_id,[])
                    value=max((float(np.dot(feature,r)) for r in refs),default=-1.)
                    similarities.append(value if geometry[i,j]<=.95 else -1.)
                ranked=sorted(enumerate(similarities),key=lambda x:x[1],reverse=True)
                if ranked and ranked[0][1]>=settings.face_similarity:
                    best,value=ranked[0]
                    if len(ranked)==1 or value-ranked[1][1]>=.08:
                        costs[best,j]=min(costs[best,j],.10+.15*(1-value))
                        for i,similarity in enumerate(similarities):
                            if i!=best and 0<=similarity<.35: costs[i,j]=1.
            return costs
    high = settings.person_confidence if kind=='person' else settings.object_confidence
    args = SimpleNamespace(track_high_thresh=high, track_low_thresh=settings.tracking_low_confidence,
        new_track_thresh=high, track_buffer=round(settings.track_buffer_sec*30), match_thresh=.8,
        fuse_score=True, gmc_method='sparseOptFlow', proximity_thresh=.3,
        appearance_thresh=.85, with_reid=True, model='auto')
    count = BaseTrack._count
    tracker = IndexedBoTSORT(args, frame_rate=fps)
    # Each class has its own tracker; construction must not reset other classes' IDs.
    BaseTrack._count = count
    tracker.encoder = lambda features, boxes: [f for f in features]
    tracker.person_face_anchors = {}
    return tracker


class SceneCuts:
    """Conservative abrupt-change guard, separate from smooth GMC."""
    def __init__(self): self.previous = self.histogram = None
    def update(self, image):
        import cv2
        small = cv2.resize(image, (96,54))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256])
        cv2.normalize(histogram,histogram)
        difference = 0. if self.previous is None else float(np.abs(small.astype(float)-self.previous).mean()/255)
        distance = 0. if self.histogram is None else float(cv2.compareHist(histogram,self.histogram,cv2.HISTCMP_BHATTACHARYYA))
        self.previous, self.histogram = small.astype(float), histogram
        return difference > .20 and distance > .35, dict(pixel_change=difference,histogram_distance=distance)


class BoTSORTPerception:
    def __init__(self, settings, output):
        self.settings, self.output = settings, Path(output)
        self.encoder = None
        self.sampled = {}

    def _extract(self, image, key, time):
        import torch
        if self.encoder is None:
            # Low-score boxes may continue an existing track, never initiate one.
            settings = self.settings.model_copy(update={
                'person_confidence': self.settings.tracking_low_confidence,
                'object_confidence': self.settings.tracking_low_confidence})
            self.encoder = ObjectEncoder(settings)
        try:
            return self.encoder.extract(image,key,time,self.output)
        except torch.cuda.OutOfMemoryError:
            if self.settings.perception_device != 'auto' or self.encoder.device.type != 'cuda':
                raise
        self.encoder.offload()
        LOG.warning('BoT-SORT 특징 추출 GPU 메모리 부족: CPU로 계속합니다.')
        return self.encoder.extract(image,key,time,self.output)

    def prepare_video(self, source, samples, registry, visual_state):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.basetrack import BaseTrack
        result_path = self.output/'tracking'/'result.json'
        if result_path.exists():
            data = read_json(result_path)
            for eid, entry in data['registry'].items(): registry.setdefault(eid,entry)
            visual_state.update(data['visual_state'])
            self.sampled = {round(f['time_sec']*1_000_000):f for f in data['sampled_frames']}
            return data['sampled_frames']
        # Kalman state is replayed from cached detections after interruption; no pickle.
        trackers, local_ids, local_registry = {}, {}, {}
        visual = dict(tracks={}, assignments={})
        frames = []; cuts = []; suppressed_cuts = []; scene = SceneCuts(); segment = 0
        person_history = {}
        revisions = {}; last_seen = {}; crowded_pairs = {}; crowded_recoveries = []
        previous_path = None; previous_boxes = []
        fps = min(self.settings.tracking_fps, source.fps)
        sample_keys = {round(s['time_sec']*1_000_000) for s in samples}
        try:
            for number,(time,image) in enumerate(tracking_frames(source,fps,samples)):
                key=f't{round(time*1_000_000):012d}'
                raw_path=self.output/'tracking'/'detections'/f'{key}.json'
                image_path=f'frames/{key}.jpg' if round(time*1_000_000) in sample_keys else f'tracking/frames/{key}.jpg'
                destination=self.output/image_path; destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists(): image.save(destination,quality=95)
                if raw_path.exists():
                    raw=read_json(raw_path)['observations']
                else:
                    raw=self._extract(image,key,time)
                    atomic_json(raw_path,dict(time_sec=time,observations=raw,
                        inference_device=str(getattr(self.encoder,'device','cached'))))
                # Coordinate units match the actual GMC image (source pixels).
                bgr=np.asarray(image)[:,:,::-1].copy()
                cut,cut_evidence=scene.update(bgr)
                person_boxes = [o['bbox'] for o in raw if o['kind']=='person']
                if cut and previous_path is not None:
                    motion = camera_motion(previous_path, destination, previous_boxes, person_boxes)
                    if motion.get('valid'):
                        suppressed_cuts.append(dict(time_sec=time, **cut_evidence, motion=motion))
                        cut = False
                previous_path, previous_boxes = destination, person_boxes
                if cut:
                    segment+=1
                    count=BaseTrack._count
                    for tracker in trackers.values(): tracker.reset()
                    BaseTrack._count=count
                    cuts.append(dict(time_sec=time,segment=segment,**cut_evidence))
                grouped={}
                for observation in raw:
                    grouped.setdefault(observation['detector_class'],[]).append(observation)
                for cls,observations in grouped.items():
                    if cls not in trackers: trackers[cls]=new_tracker(self.settings,fps,observations[0]['kind'])
                output=[]
                for cls,tracker in trackers.items():
                    observations=grouped.get(cls,[])
                    rows=[]; features=[]
                    for o in observations:
                        rows.append([v*(image.width if i%2==0 else image.height)/1000 for i,v in enumerate(o['bbox'])]
                                    +[o['detection_confidence'],0.])
                        features.append(o['body_feature'] if o['kind']=='person' else o['object_feature'])
                    batch=DetectionBatch(Boxes(np.array(rows,dtype=np.float32).reshape(-1,6),(image.height,image.width)),observations=observations)
                    tracker.update(batch,img=bgr,feats=np.asarray(features,dtype=np.float32))
                    visible=[t for t in tracker.tracked_stracks if t.frame_id==tracker.frame_id]
                    if cls=='person':
                        # When an occluded track returns inside a crowd, its local
                        # Kalman ID is not sufficient proof of person identity.
                        # Split BOTH participants and let future appearances relink.
                        boundary=set()
                        for track in visible:
                            tid=int(track.track_id); last=last_seen.get((segment,tid),number)
                            if 1<number-last<=max(2,round(fps*.5)):
                                for (shot,a,b),at in crowded_pairs.items():
                                    if shot==segment and tid in (a,b) and 0<=last-at<=3:
                                        boundary.update((a,b))
                        for tid in boundary: revisions[(segment,tid)]=revisions.get((segment,tid),0)+1
                        if boundary: crowded_recoveries.append(dict(time_sec=time,tracker_ids=sorted(boundary)))
                        for i,left in enumerate(visible):
                            a=np.array(observations[int(left.idx)]['bbox'])
                            for right in visible[i+1:]:
                                b=np.array(observations[int(right.idx)]['bbox'])
                                intersection=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
                                smaller=min(float(np.prod(a[2:]-a[:2])),float(np.prod(b[2:]-b[:2])))
                                if smaller>0 and intersection/smaller>.6:
                                    crowded_pairs[(segment,int(left.track_id),int(right.track_id))]=number
                        for track in visible: last_seen[(segment,int(track.track_id))]=number
                    for track in tracker.tracked_stracks:
                        if track.frame_id!=tracker.frame_id: continue
                        observation=observations[int(track.idx)]
                        revision=revisions.get((segment,int(track.track_id)),0) if cls=='person' else 0
                        local=(segment,cls,int(track.track_id),revision)
                        new_identity=local not in local_ids
                        if local not in local_ids:
                            prefix='person' if observation['kind']=='person' else 'object'
                            n=1+sum(e['kind']==observation['kind'] for e in local_registry.values())
                            eid=f'{prefix}_{n:04d}'; local_ids[local]=eid
                            local_registry[eid]=dict(entity_id=eid,kind=observation['kind'],
                                label=f'인물 {n}' if prefix=='person' else observation['label'],
                                appearance='BoT-SORT · OSNet' if prefix=='person' else f"BoT-SORT · DINOv2 · {cls}",
                                gallery=[],observations=[],first_seen_sec=time,last_seen_sec=time,
                                possible_matches=[],identity_status='new')
                            visual['tracks'][eid]=dict(samples=[],frame_keys=[])
                        eid=local_ids[local]; entry=local_registry[eid]
                        entry['last_seen_sec']=time
                        entry['identity_status']='matched' if track.is_activated else 'uncertain'
                        entry['gallery']=list(dict.fromkeys(entry['gallery']+[observation['crop']]))[-3:]
                        proof=dict(entity_id=eid,identity_method='BoT-SORT',identity_similarity=None,
                            identity_uncertain=not track.is_activated,possible_matches=[],
                            tracker_id=int(track.track_id),shot_segment=segment,
                            reid_model='OSNet-AIN' if observation['kind']=='person' else 'DINOv2-small')
                        visual['assignments'][observation['detection_id']]=proof
                        track_data=visual['tracks'][eid]
                        track_data['frame_keys'].append(key)
                        enriched = dict(observation, shot_segment=segment)
                        if new_identity and revision:
                            enriched['identity_boundary']='crowded_occlusion_recovery'
                        if observation['kind']=='person' and observation.get('face_feature') and observation.get('face_evidence',{}).get('usable'):
                            anchors=tracker.person_face_anchors.setdefault(int(track.track_id),[])
                            feature=observation['face_feature']
                            # Preserve early anchors; contradictory new faces cannot
                            # silently overwrite an identity after an ID switch.
                            if not anchors or max(float(np.dot(feature,a)) for a in anchors)>=.4:
                                if len(anchors)<12: anchors.append(feature)
                                else: anchors[6:]=anchors[7:]+[feature]
                        track_data['samples']=select_references(track_data['samples']+[enriched])
                        if observation['kind']=='person':
                            person_history.setdefault(eid, []).append(enriched)
                        output.append({k:v for k,v in dict(observation,**proof).items() if not k.endswith('_feature')})
                assert len({o['entity_id'] for o in output})==len(output)
                frames.append(dict(time_sec=time,image=image_path,observations=output))
                if number%10==0:
                    LOG.info('BoT-SORT %.1ffps · %.2f/%.2f초 · 현재 %d개 추적',fps,time,source.duration,len(output))
                atomic_json(self.output/'tracking'/'progress.json',dict(processed_frames=number+1,time_sec=time,duration_sec=source.duration))
        finally:
            if self.encoder is not None: self.encoder.offload()
        if visual['tracks']:
            # Keep generic-object merging separate: person identity has temporal,
            # multi-view and competing-person constraints rather than one best pair.
            object_ids = {eid for eid in visual['tracks'] if local_registry[eid]['kind']!='person'}
            objects = dict(tracks={eid:visual['tracks'][eid] for eid in object_ids},
                assignments={key:value for key,value in visual['assignments'].items() if value['entity_id'] in object_ids})
            reconcile_tracklets(local_registry,objects,self.settings)
            for eid in object_ids: visual['tracks'].pop(eid,None)
            visual['tracks'].update(objects['tracks'])
        linking = reconcile_people(local_registry, visual, person_history, self.settings, self.output)
        atomic_json(self.output/'tracking'/'person_linking.json',linking)
        for frame in frames:
            for o in frame['observations']: o.update(visual['assignments'][o['detection_id']])
        sampled=[frame for frame in frames if round(frame['time_sec']*1_000_000) in sample_keys]
        if {round(f['time_sec']*1_000_000) for f in sampled}!=sample_keys:
            raise ValueError('일부 1fps 분석 샘플의 추적 결과가 없습니다.')
        data=dict(tracker='BoT-SORT',implementation='ultralytics==8.3.221',tracking_fps=fps,
            processed_frames=len(frames),scene_cuts=cuts,suppressed_cuts=suppressed_cuts,crowded_recoveries=crowded_recoveries,person_linking=linking,registry=local_registry,visual_state=visual,sampled_frames=sampled)
        atomic_json(result_path,data)
        atomic_json(self.output/'tracking'/'frames.json',frames)
        registry.update(local_registry);visual_state.update(visual)
        self.sampled={round(f['time_sec']*1_000_000):f for f in sampled}
        return sampled

    def prepare(self, source, clip, registry, visual_state):
        if not self.sampled:
            raise RuntimeError('BoT-SORT의 전체 영상 추적을 먼저 실행하세요.')
        return [dict(deepcopy(self.sampled[round(f.time_sec*1_000_000)]),frame_index=i)
                for i,f in enumerate(clip.frames)]
