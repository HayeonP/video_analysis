"""Global sampled-frame tracking with appearance and camera-aware continuity."""
from copy import deepcopy
import logging
from pathlib import Path

import numpy as np

from .perception import PersonEncoder, normalize
from .storage import atomic_json, read_json

LOG = logging.getLogger(__name__)


def filter_detections(observations, settings):
    """Gate detection evidence before assigning IDs, including cached frames."""
    accepted, rejected = [], []
    for observation in observations:
        threshold = settings.person_confidence if observation['kind'] == 'person' else settings.object_confidence
        score = observation.get('detection_confidence')
        if isinstance(score, (int, float)) and np.isfinite(score) and threshold <= score <= 1:
            accepted.append(observation)
        else:
            rejected.append(dict(detection_id=observation['detection_id'], detector_class=observation.get('detector_class'),
                detection_confidence=score if isinstance(score,(int,float)) and np.isfinite(score) else None,
                bbox=observation['bbox'], threshold=threshold, reason='insufficient_detection_confidence'))
    return accepted, rejected


class ObjectEncoder(PersonEncoder):
    """One YOLO pass; person features and generic object features are separate."""
    def __init__(self, settings):
        super().__init__(settings)
        self.object_processor = self.object_model = None

    def object_features(self, crops):
        if not crops:
            return []
        import torch
        from transformers import AutoImageProcessor, AutoModel
        if self.object_model is None:
            kwargs = dict(revision=self.settings.object_model_revision)
            self.object_processor = AutoImageProcessor.from_pretrained(self.settings.object_model_id, use_fast=False, **kwargs)
            self.object_model = AutoModel.from_pretrained(self.settings.object_model_id, **kwargs).to(self.device).eval()
        result = []
        for start in range(0, len(crops), 8):
            inputs = self.object_processor(images=crops[start:start+8], return_tensors='pt').to(self.device)
            with torch.inference_mode():
                vectors = self.object_model(**inputs).last_hidden_state[:, 0].cpu().numpy()
            result.extend(normalize(v).tolist() for v in vectors)
        return result

    def extract(self, image, frame_key, time_sec, output):
        result = self.detector.predict(image, conf=min(self.settings.person_confidence, self.settings.object_confidence),
            iou=.45, imgsz=self.settings.detector_size, device=self.device, verbose=False)[0]
        raw = list(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()))
        people = [(box, conf) for box, conf, cls in raw if int(cls) == 0 and conf >= self.settings.person_confidence]
        observations = super().extract(image, frame_key, time_sec, output, detections=people) if people else []
        for obs in observations:
            obs.update(kind='person', detector_class='person', label='인물')
        crops, objects = [], []
        for index, (box, conf, cls) in enumerate(raw):
            if int(cls) == 0 or conf < self.settings.object_confidence:
                continue
            x1, y1, x2, y2 = box.clip([0, 0, 0, 0], [image.width, image.height, image.width, image.height])
            crop = image.crop((int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))))
            path = f'crops/{frame_key}_object_{index:03d}.jpg'
            (output / 'crops').mkdir(exist_ok=True)
            crop.save(output / path, quality=95)
            crops.append(crop)
            label = self.detector.names[int(cls)]
            objects.append(dict(detection_id=f'{frame_key}_object_{index:03d}', frame_key=frame_key, time_sec=time_sec,
                bbox=[float(x1/image.width*1000), float(y1/image.height*1000), float(x2/image.width*1000), float(y2/image.height*1000)],
                crop=path, detection_confidence=float(conf), grounding_source='YOLO11s', source_size=list(image.size),
                kind='object', detector_class=label, label=label))
        for obs, feature in zip(objects, self.object_features(crops)):
            obs['object_feature'] = feature
        return observations + objects


def similarity(query, samples, settings):
    """Return an affinity and its underlying evidence; never use box overlap."""
    def best(key, minimum_quality=0):
        q = query.get(key)
        refs = [s[key] for s in samples if s.get(key) and s.get('face_quality', 0) >= minimum_quality]
        return max((float(np.dot(q, v)) for v in refs), default=-1.) if q else -1.
    conflict = fragment = False
    if query['kind'] == 'person':
        face, body = best('face_feature', 20), best('body_feature')
        has_faces = query.get('face_quality', 0) >= 20 and any(s.get('face_quality', 0) >= 20 and s.get('face_feature') for s in samples)
        if has_faces and face >= settings.face_similarity:
            threshold, value, method = settings.face_similarity, face, 'SFace'
            affinity = .5 + .5 * (value-threshold)/(1-threshold)
        else:
            threshold, value, method = settings.body_similarity, body, 'OSNet-AIN'
            affinity = .5 * (value-threshold)/(1-threshold) if value >= threshold else -1.
            # Strong contradictory face evidence and clipped clothing fragments
            # must not be overridden by similar clothes.
            # A tiny/profile face is not reliable *negative* evidence. Requiring
            # high-quality references too avoids vetoing a good body match on zoom.
            quality = settings.face_conflict_quality
            reliable_faces = query.get('face_quality', 0) >= quality and any(
                s.get('face_feature') and s.get('face_quality', 0) >= quality for s in samples)
            conflict = reliable_faces and best('face_feature', quality) < .25
            x1, y1, x2, y2 = query['bbox']
            width, height = query.get('source_size', (1, 1))
            aspect = (x2-x1)*width / max((y2-y1)*height, 1e-9)
            # Touching a border alone does not mean this is an unrecognizable fragment.
            fragment = (x1 <= 2 or x2 >= 998) and aspect < .20
            if conflict or fragment:
                affinity = -1.
    else:
        value, method = best('object_feature'), 'DINOv2-small'
        # A class is a weak compatibility hint, never sufficient identity proof.
        same_class = any(s.get('detector_class') == query.get('detector_class') for s in samples)
        threshold = settings.object_similarity if same_class else min(.98, settings.object_similarity + .05)
        affinity = (value-threshold)/(1-threshold) if value >= threshold else -1.
    return affinity, dict(identity_method=method, identity_similarity=value,
        identity_face_conflict=conflict, identity_fragment=fragment)


def select_references(samples, limit=8):
    """Retain a clear face, recent samples and diverse front/side/back appearances."""
    unique = {s['frame_key']: s for s in samples}
    pool = sorted(unique.values(), key=lambda s:s.get('time_sec', 0.))
    if len(pool) <= limit:
        return pool
    selected = list(dict.fromkeys([max(range(len(pool)), key=lambda i: pool[i].get('face_quality', 0)),
                                   len(pool)-1, len(pool)-2]))
    def distance(i):
        key = 'body_feature' if pool[i]['kind'] == 'person' else 'object_feature'
        vector = pool[i].get(key)
        refs = [pool[j][key] for j in selected if pool[j].get(key)]
        return min((1-float(np.dot(vector, v)) for v in refs), default=0.) if vector else 0.
    while len(selected) < limit:
        selected.append(max((i for i in range(len(pool)) if i not in selected), key=distance))
    return [pool[i] for i in sorted(selected)]


def associate_frame(observations, registry, visual_state, settings, *, motion=None):
    """One-to-one assignments; repeated source detections reuse their saved IDs."""
    from scipy.optimize import linear_sum_assignment
    tracks = visual_state.setdefault('tracks', {})
    saved = visual_state.setdefault('assignments', {})
    occupied = {saved[o['detection_id']]['entity_id'] for o in observations if o['detection_id'] in saved}
    fresh = [o for o in observations if o['detection_id'] not in saved]
    candidates = [eid for eid in tracks if eid not in occupied]
    scores = np.full((len(fresh), len(candidates)+len(fresh)), -1.)
    scores[:, len(candidates):] = 0.
    evidence = {}
    for i, query in enumerate(fresh):
        feature_key = 'body_feature' if query['kind']=='person' else 'object_feature'
        active_similarities = []
        if motion and motion.get('valid') and query.get(feature_key):
            for eid in candidates:
                recent = max(tracks[eid]['samples'], key=lambda s:s.get('time_sec',-1), default={})
                if (registry[eid]['kind']==query['kind'] and recent.get(feature_key)
                        and abs(recent.get('time_sec',-100)-motion['previous_time_sec']) < 1e-6):
                    active_similarities.append(float(np.dot(query[feature_key], recent[feature_key])))
        best_recent = max(active_similarities, default=-1.)
        for j, eid in enumerate(candidates):
            if query['kind'] != registry[eid]['kind']:
                continue
            score, proof = similarity(query, tracks[eid]['samples'], settings)
            from .motion import continuity
            candidate = continuity(query, tracks[eid]['samples'], motion, settings,
                blocked=proof['identity_face_conflict'] or proof['identity_fragment'])
            # Crossing people must not inherit an ID just because they occupy
            # its previous position when another active body is a better match.
            if candidate and candidate[1]['identity_similarity'] + .05 < best_recent:
                candidate = None
            if candidate and proof['identity_method'] != 'SFace' and candidate[0] > score:
                score, proof = candidate
            scores[i, j] = score
            evidence[i, j] = proof
    chosen = dict(zip(*linear_sum_assignment(-scores))) if fresh else {}
    for i, query in enumerate(fresh):
        j = chosen[i]
        uncertain = False
        possible = [eid for c, eid in enumerate(candidates) if scores[i, c] > 0]
        proof = dict(identity_method='new', identity_similarity=None)
        accepted = j < len(candidates) and scores[i, j] > 0
        if accepted:
            other_row = np.delete(scores[i, :len(candidates)], j)
            other_col = np.delete(scores[:, j], i)
            rival = max(0., float(other_row.max()) if other_row.size else 0., float(other_col.max()) if other_col.size else 0.)
            accepted = scores[i, j] - rival >= settings.identity_margin
        if accepted:
            eid = candidates[j]
            proof = evidence[i, j]
            registry[eid]['identity_status'] = 'matched'
        else:
            prefix = 'person' if query['kind'] == 'person' else 'object'
            number = 1 + max((int(k.split('_')[1]) for k in registry if k.startswith(prefix+'_')), default=0)
            eid = f'{prefix}_{number:04d}'
            uncertain = any(registry[c]['kind'] == query['kind'] for c in candidates)
            registry[eid] = dict(entity_id=eid, kind=query['kind'], label=f'인물 {number}' if prefix == 'person' else query['label'],
                appearance='SFace·OSNet-AIN 비교 대상' if prefix == 'person' else f"YOLO 검출 종류: {query['label']}",
                gallery=[], observations=[], first_seen_sec=query['time_sec'], last_seen_sec=query['time_sec'],
                possible_matches=possible, identity_status='uncertain' if uncertain else 'new')
            tracks[eid] = {'samples': [], 'frame_keys': []}
        proof.update(entity_id=eid, identity_uncertain=not accepted and uncertain, possible_matches=possible if not accepted else [])
        saved[query['detection_id']] = proof
        sample = {k: v for k, v in query.items() if k in {'face_feature','body_feature','object_feature','face_quality','detector_class','crop','frame_key','bbox','kind','source_size','time_sec'}}
        tracks[eid].setdefault('frame_keys', []).append(query['frame_key'])
        samples = tracks[eid]['samples']
        samples.append(sample)
        if len(samples) > 8:
            # Keep clear face anchors plus recent appearances; provisional links
            # never contaminate a different entity's reference gallery.
            tracks[eid]['samples'] = select_references(samples)
    result = []
    for query in observations:
        proof = saved[query['detection_id']]
        eid = proof['entity_id']
        entry = registry[eid]
        entry['first_seen_sec'] = min(entry['first_seen_sec'], query['time_sec'])
        entry['last_seen_sec'] = max(entry['last_seen_sec'], query['time_sec'])
        gallery = list(dict.fromkeys(entry['gallery'] + [query['crop']]))
        entry['gallery'] = gallery[:1] + gallery[-2:] if len(gallery) > 3 else gallery
        result.append({k: v for k, v in dict(query, **proof).items() if not k.endswith('_feature')})
    assert len({o['entity_id'] for o in result}) == len(result), 'A frame cannot assign one ID to two detections'
    return result


def merge_affinity(query, samples, settings):
    """Separate group-merge confidence from the deliberately face-prioritized online score."""
    score, proof = similarity(query, samples, settings)
    if query['kind'] != 'person' or proof['identity_method'] != 'OSNet-AIN' or score <= 0:
        return score
    value = proof['identity_similarity']
    if value < settings.body_merge_similarity:
        return -1.
    # Online body scores never exceed .5: comparing them against .65 made
    # face-free fragment recovery impossible, even for identical embeddings.
    return .65 + .05 * (min(value, 1.) - settings.body_merge_similarity) / (1. - settings.body_merge_similarity)


def reconcile_tracklets(registry, visual_state, settings):
    """Use all observations in the window to join clear, non-co-visible fragments."""
    from .identity import canonical
    tracks = visual_state['tracks']
    changed = True
    while changed:
        changed = False
        proposals = []
        keys = list(tracks)
        for i, left in enumerate(keys):
            for right in keys[i+1:]:
                if registry[left]['kind'] != registry[right]['kind']:
                    continue
                if set(tracks[left]['frame_keys']) & set(tracks[right]['frame_keys']):
                    continue
                samples_a, samples_b = tracks[left]['samples'], tracks[right]['samples']
                scores = [merge_affinity(q, samples_b, settings) for q in samples_a]
                reverse = [merge_affinity(q, samples_a, settings) for q in samples_b]
                score = min(max(scores, default=-1), max(reverse, default=-1))
                # Higher bar for joining existing groups than adding one sample.
                if score >= .65:
                    proposals.append((score, left, right))
        for score, left, right in sorted(proposals, reverse=True):
            rivals = [s for s,a,b in proposals if (a,b)!=(left,right) and ({a,b}&{left,right})]
            if rivals and score - max(rivals) < settings.identity_margin:
                continue
            target, old = sorted([left, right])
            current, previous = registry[target], registry[old]
            current['observations'].extend(previous['observations'])
            current['first_seen_sec'] = min(current['first_seen_sec'], previous['first_seen_sec'])
            current['last_seen_sec'] = max(current['last_seen_sec'], previous['last_seen_sec'])
            current['gallery'] = list(dict.fromkeys(current['gallery'] + previous['gallery']))[:3]
            current['identity_status'] = 'matched'
            current['possible_matches'] = []
            previous['observations'] = []
            previous['alias_of'] = target
            samples = tracks[target]['samples'] + tracks[old]['samples']
            tracks[target]['samples'] = select_references(samples)
            tracks[target]['frame_keys'] = sorted(set(tracks[target]['frame_keys']) | set(tracks[old]['frame_keys']))
            del tracks[old]
            for assignment in visual_state['assignments'].values():
                if canonical(registry, assignment['entity_id']) == target:
                    assignment.update(entity_id=target, identity_uncertain=False, possible_matches=[],
                                      identity_merge_method='mutual_window_appearance')
            changed = True
            break


class WindowPerception:
    def __init__(self, settings, output):
        self.settings, self.output = settings, Path(output)
        self.encoder = None

    def extract(self, image, frame_key, time_sec):
        import torch
        try:
            return self.encoder.extract(image, frame_key, time_sec, self.output)
        except torch.cuda.OutOfMemoryError:
            if self.settings.perception_device != 'auto' or self.encoder.device.type != 'cuda':
                raise
            LOG.warning('GPU 메모리 부족 · %.3fs부터 이번 윈도 검출은 CPU로 재시도', time_sec)
        self.encoder.offload()
        return self.encoder.extract(image, frame_key, time_sec, self.output)

    def prepare(self, source, clip, registry, visual_state):
        folder = self.output / 'perception' / 'frames'
        folder.mkdir(parents=True, exist_ok=True)
        needed = {round(f.time_sec*1_000_000) for f in clip.frames if not (folder/f't{round(f.time_sec*1_000_000):012d}.json').exists()}
        if needed:
            if self.encoder is None:
                self.encoder = ObjectEncoder(self.settings)
                import hashlib
                atomic_json(self.output / 'perception' / 'model_sources.json', {
                    'weights_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in self.encoder.root.iterdir() if p.suffix in {'.pt','.pth','.onnx'}},
                    'object_model': self.settings.object_model_id,
                    'object_revision': self.settings.object_model_revision,
                    'device_policy': self.settings.perception_device,
                    'preferred_device': str(self.encoder.preferred_device),
                    'face_device': 'cpu'})
            elif hasattr(self.encoder, 'activate'):
                self.encoder.activate()
            try:
                for key, image in source.original_images(clip):
                    if key not in needed:
                        continue
                    frame_key = f't{key:012d}'
                    path = f'frames/{frame_key}.jpg'
                    image.save(self.output / path, quality=95)
                    device = str(getattr(self.encoder, 'device', 'custom'))
                    LOG.info('YOLO 검출·사람/물체 특징 추출 · %.3fs · %s', key/1_000_000, device)
                    observations = self.extract(image, frame_key, key/1_000_000)
                    atomic_json(folder/f'{frame_key}.json', dict(frame_key=frame_key, time_sec=key/1_000_000,
                        image=path, observations=observations, inference_device=str(getattr(self.encoder, 'device', 'custom'))))
            finally:
                if hasattr(self.encoder, 'offload'):
                    self.encoder.offload()
        frames = []
        for index, frame in enumerate(clip.frames):
            key = f't{round(frame.time_sec*1_000_000):012d}'
            raw = read_json(folder/f'{key}.json')
            accepted, rejected = filter_detections(raw['observations'], self.settings)
            atomic_json(self.output / 'perception' / 'filters' / f'{key}.json', {
                'time_sec': frame.time_sec, 'accepted_detection_ids': [o['detection_id'] for o in accepted],
                'rejected': rejected, 'person_threshold': self.settings.person_confidence,
                'object_threshold': self.settings.object_confidence})
            # Commit a frame atomically so interruption cannot half-assign it.
            next_registry, next_visual = deepcopy(registry), deepcopy(visual_state)
            motion = None
            if any(o['detection_id'] not in next_visual.get('assignments', {}) for o in accepted):
                from .motion import camera_motion
                previous = next_visual.get('motion_previous')
                if previous and 0 < frame.time_sec - previous['time_sec'] <= 1.5:
                    motion = camera_motion(self.output / previous['image'], self.output / frame.path,
                        previous['boxes'], [o['bbox'] for o in accepted if o['kind']=='person'])
                    motion['previous_time_sec'] = previous['time_sec']
                    atomic_json(self.output / 'perception' / 'motion' / f'{key}.json', motion)
                next_visual['motion_previous'] = dict(time_sec=frame.time_sec, image=frame.path,
                    boxes=[o['bbox'] for o in accepted if o['kind']=='person'])
            observations = associate_frame(accepted, next_registry, next_visual, self.settings, motion=motion)
            registry.clear(); registry.update(next_registry)
            visual_state.clear(); visual_state.update(next_visual)
            frames.append(dict(frame_index=index, time_sec=frame.time_sec, image=frame.path, observations=observations))
        reconcile_tracklets(registry, visual_state, self.settings)
        for f in frames:
            for observation in f['observations']:
                observation.update(visual_state['assignments'][observation['detection_id']])
        return frames
