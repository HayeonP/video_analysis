"""Person detection and appearance evidence, shared by all overlapping windows.

Coordinates are used to crop a detection and associate its face, never to link
people across frames. Global identity uses appearance and co-visibility only.
"""
from pathlib import Path
import gc
import hashlib
import logging
import urllib.request

import numpy as np
from PIL import Image

from .config import ROOT
from .storage import atomic_json, read_json
from .video import sliding_windows

LOG = logging.getLogger(__name__)
OSNET_FILE = 'osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
OSNET_REVISION = 'a5c5cc037c24235cda3b21085b93ad77c9616224'


def model_files():
    from huggingface_hub import hf_hub_download
    root = ROOT / 'models'
    root.mkdir(exist_ok=True)
    urls = {
        'yolo11s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt',
        'yunet.onnx': 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        'sface.onnx': 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
    }
    for name, url in urls.items():
        path = root / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.download')
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
    if not (root / OSNET_FILE).exists():
        hf_hub_download('kaiyangzhou/osnet', OSNET_FILE, revision=OSNET_REVISION, local_dir=root)
    return root


def normalize(array):
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    return array / max(float(np.linalg.norm(array)), 1e-12)


def perception_device(policy):
    import torch
    if policy == 'auto':
        policy = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(policy)


def face_evidence_quality(bgr, face, settings):
    """Gate recognition BEFORE SFace; dimensions refer to original source pixels."""
    import cv2
    x, y, w, h = map(float, face[:4])
    eye_distance = float(np.linalg.norm(face[4:6] - face[6:8]))
    eye_mid = (face[4:6] + face[6:8]) / 2
    nose_offset = abs(float(face[8] - eye_mid[0])) / max(eye_distance, 1.)
    quality = min(w,h)*float(face[14])*min(1., eye_distance/max(w*.3,1.))
    patch = bgr[max(0,int(y)):min(bgr.shape[0],int(y+h)), max(0,int(x)):min(bgr.shape[1],int(x+w))]
    sharpness = 0.
    if patch.size:
        gray = cv2.cvtColor(cv2.resize(patch,(64,64)),cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray,cv2.CV_64F).var())
    reason = ('too_small' if min(w,h)<settings.face_min_size else
              'low_confidence' if face[14]<.9 else
              'profile_or_small_eyes' if eye_distance<settings.face_min_eye_distance or eye_distance/max(w,1)<.20 or nose_offset>.6 else
              'blurred' if sharpness<settings.face_min_sharpness else
              'clipped' if x<0 or y<0 or x+w>bgr.shape[1] or y+h>bgr.shape[0] else 'accepted')
    return dict(usable=reason=='accepted',reason=reason,min_size_px=min(w,h),
                eye_distance_px=eye_distance,sharpness=sharpness,quality=quality)


class PersonEncoder:
    def __init__(self, settings):
        import cv2
        import torch
        from torchvision import transforms
        from ultralytics import YOLO
        from .third_party.osnet.osnet_ain import osnet_ain_x1_0
        self.settings = settings
        # Ultralytics select_device('cpu') hides every GPU process-wide. A
        # torch.device bypasses that environment mutation so Qwen can use CUDA
        # after CPU detection, and an explicit user GPU mask stays intact.
        self.preferred_device = perception_device(settings.perception_device)
        self.device = torch.device('cpu')
        self.root = model_files()
        self.detector = YOLO(str(self.root / 'yolo11s.pt'))
        state = torch.load(self.root / OSNET_FILE, map_location='cpu', weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.body = osnet_ain_x1_0(num_classes=state['classifier.weight'].shape[0], pretrained=False)
        self.body.load_state_dict(state)
        self.body.eval()
        self.transform = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(),
                                             transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        cv2.setNumThreads(2)
        self.face_detector = cv2.FaceDetectorYN.create(str(self.root / 'yunet.onnx'), '', (1280, 720), .8)
        self.face_encoder = cv2.FaceRecognizerSF.create(str(self.root / 'sface.onnx'), '')
        self.activate()
        LOG.info('검출·외형 모델 장치: %s (설정 %s) · YuNet/SFace: CPU', self.device, settings.perception_device)

    def to_device(self, device):
        """Keep weights in RAM between windows; never change the GPU mask."""
        import torch
        device = torch.device(device)
        # YOLO.to also resets its predictor, releasing cached CUDA tensors.
        self.detector.to(device)
        self.body.to(device)
        if getattr(self, 'object_model', None) is not None:
            self.object_model.to(device)
        self.device = device

    def offload(self):
        """Release perception VRAM before Qwen's video/token generation."""
        import torch
        self.to_device('cpu')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def activate(self):
        import torch
        if self.device == self.preferred_device:
            return
        try:
            self.to_device(self.preferred_device)
            return
        except torch.cuda.OutOfMemoryError:
            if self.settings.perception_device != 'auto':
                raise
            LOG.warning('검출 모델을 GPU에 올릴 메모리가 부족하여 이번 윈도는 CPU 사용')
        # Exit the except block first to release traceback-held CUDA tensors.
        self.offload()

    def extract(self, image, frame_key, time_sec, output, *, detections=None):
        import cv2
        import torch
        from scipy.optimize import linear_sum_assignment
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # NMS removes duplicate detections in ONE image. It is not an identity
        # association and never links subjects across images by box overlap.
        if detections is None:
            result = self.detector.predict(image, classes=[0], conf=self.settings.person_confidence, iou=.45,
                                           imgsz=self.settings.detector_size, device=self.device, verbose=False)[0]
            detections = list(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()))
        boxes = sorted(detections, key=lambda b: float(b[0][0]))
        scale = min(1., 1280 / image.width)
        small = cv2.resize(bgr, (round(image.width * scale), round(image.height * scale)))
        self.face_detector.setInputSize((small.shape[1], small.shape[0]))
        faces = self.face_detector.detect(small)[1]
        faces = np.asarray(faces if faces is not None else [], dtype=np.float32).reshape(-1, 15)
        faces[:, :14] /= scale
        costs = np.full((len(boxes), len(faces)), 1e6)
        for i, (box, _) in enumerate(boxes):
            x1, y1, x2, y2 = box
            for j, face in enumerate(faces):
                x, y, w, h = face[:4]
                cx, cy = x + w/2, y + h/2
                if x1 <= cx <= x2 and y1 <= cy <= y1 + .55 * (y2-y1):
                    costs[i,j] = abs(y-y1)/(y2-y1) * 3 + abs(cx-(x1+x2)/2)/(x2-x1)
        assigned = {int(i): int(j) for i, j in zip(*linear_sum_assignment(costs)) if costs[i,j] < 1}
        observations, tensors = [], []
        folder = output / 'crops'
        folder.mkdir(exist_ok=True)
        for i, (box, confidence) in enumerate(boxes):
            x1, y1, x2, y2 = box.clip([0, 0, 0, 0], [image.width, image.height, image.width, image.height])
            crop = image.crop((int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))))
            path = f'crops/{frame_key}_person_{i:03d}.jpg'
            crop.save(output / path, quality=95)
            tensors.append(self.transform(crop))
            observation = {'detection_id': f'{frame_key}_person_{i:03d}', 'frame_key': frame_key,
                           'time_sec': time_sec, 'bbox': [float(x1/image.width*1000), float(y1/image.height*1000),
                                                        float(x2/image.width*1000), float(y2/image.height*1000)],
                           'crop': path, 'detection_confidence': float(confidence), 'face_feature': None,
                           'face_quality': 0., 'face_evidence': {'usable': False, 'reason': 'not_detected'}, 'grounding_source': 'YOLO11s', 'source_size': list(image.size)}
            if i in assigned:
                face = faces[assigned[i]]
                evidence = face_evidence_quality(bgr, face, self.settings)
                observation['face_evidence'] = evidence
                if evidence['usable']:
                    aligned = self.face_encoder.alignCrop(bgr, face)
                    observation['face_feature'] = normalize(self.face_encoder.feature(aligned)).tolist()
                    observation['face_quality'] = evidence['quality']
            observations.append(observation)
        if tensors:
            with torch.inference_mode():
                features = self.body(torch.stack(tensors).to(self.device)).cpu().numpy()
            for observation, feature in zip(observations, features):
                observation['body_feature'] = normalize(feature).tolist()
        return observations


def extract_people(source, output, settings):
    """Decode unique PTS once at source resolution; retain resumable evidence."""
    import av
    times = {}
    for start, end in sliding_windows(source.duration, settings.window_sec, settings.stride_sec):
        for frame in source.sample(start, end, settings.sample_fps).frames:
            times[round(frame.time_sec * 1_000_000)] = frame.time_sec
    for frame in source.sample(0, source.duration, settings.tracking_fps or 6.).frames:
        times[round(frame.time_sec * 1_000_000)] = frame.time_sec
    folder = output / 'perception' / 'frames'
    folder.mkdir(parents=True, exist_ok=True)
    (output / 'frames').mkdir(exist_ok=True)
    missing = {key for key in times if not (folder / f't{key:012d}.json').exists()}
    if missing:
        encoder = PersonEncoder(settings)
        with av.open(str(source.path)) as container:
            stream = container.streams.video[0]
            for decoded in container.decode(stream):
                if decoded.pts is None:
                    continue
                timestamp = float(decoded.pts * stream.time_base) - source.origin
                key = round(timestamp * 1_000_000)
                if key not in missing:
                    continue
                frame_key = f't{key:012d}'
                LOG.info('전용 검출·외형 추출 · %.3fs (%d/%d)', timestamp, len(times)-len(missing)+1, len(times))
                image = decoded.to_image().convert('RGB')
                image.save(output / 'frames' / f'{frame_key}.jpg', quality=95)
                observations = encoder.extract(image, frame_key, timestamp, output)
                atomic_json(folder / f'{frame_key}.json', {'time_sec': timestamp, 'frame_key': frame_key,
                                                         'image': f'frames/{frame_key}.jpg', 'observations': observations})
                missing.remove(key)
                if not missing:
                    break
        del encoder
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if missing:
            raise ValueError(f'원본 프레임을 찾을 수 없습니다: {sorted(missing)}')
    frames = [read_json(folder / f't{key:012d}.json') for key in sorted(times)]
    return frames


def cluster_people(frames, *, face_threshold=.5, body_threshold=.75):
    """Constrained appearance clustering with no position/IoU identity links.

    Strong face links are established first. Clothing links cannot override
    contradictory face evidence or merge any co-visible detections.
    """
    observations = [o for f in frames for o in f['observations']]
    n = len(observations)
    body = np.asarray([o['body_feature'] for o in observations], dtype=np.float32).reshape(n, 512)
    face = np.asarray([o['face_feature'] or [0.] * 128 for o in observations], dtype=np.float32).reshape(n, 128)
    body_scores, face_scores = body @ body.T, face @ face.T
    parent = list(range(n))
    members = {i: {i} for i in range(n)}
    visible = {i: {o['frame_key']} for i, o in enumerate(observations)}
    links = []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j, score, method):
        a, b = root(i), root(j)
        if a == b or visible[a] & visible[b]:
            return
        faces_a = [k for k in members[a] if observations[k]['face_feature'] is not None]
        faces_b = [k for k in members[b] if observations[k]['face_feature'] is not None]
        if method == 'body' and len(faces_a) >= 3 and len(faces_b) >= 3:
            if float(face_scores[np.ix_(faces_a, faces_b)].max()) < face_threshold:
                return
        if a > b:
            a, b = b, a
        parent[b] = a
        members[a] |= members.pop(b)
        visible[a] |= visible.pop(b)
        links.append({'a': observations[i]['detection_id'], 'b': observations[j]['detection_id'],
                      'method': method, 'cosine_similarity': float(score)})

    for method, scores, threshold in [('face', face_scores, face_threshold), ('body', body_scores, body_threshold)]:
        rows, cols = np.where(np.triu(scores >= threshold, k=1))
        for i, j in sorted(zip(rows, cols), key=lambda ij: (-float(scores[ij]), int(ij[0]), int(ij[1]))):
            join(int(i), int(j), scores[i, j], method)
    groups = sorted(members.values(), key=lambda group: min(group))
    tracks = {}
    for number, group in enumerate(groups, 1):
        entity_id = f'person_{number:04d}'
        ordered = sorted(group)
        # Prefer sharp, large faces for the displayed representative, not the
        # first arbitrary crop. Keep full-resolution source crops unchanged.
        gallery = sorted(ordered, key=lambda k: (-observations[k]['face_quality'], k))[:3]
        tracks[entity_id] = {'entity_id': entity_id, 'label': f'인물 {number}', 'kind': 'person',
                             'appearance': f'영상에서 {entity_id}로 표시된 인물',
                             'gallery': [observations[k]['crop'] for k in gallery], 'observations': [],
                             'first_seen_sec': min(observations[k]['time_sec'] for k in group),
                             'last_seen_sec': max(observations[k]['time_sec'] for k in group),
                             'possible_matches': [], 'identity_status': 'matched' if len(group)>1 else 'uncertain'}
        outside = [k for k in range(n) if k not in group]
        for k in ordered:
            o = observations[k]
            peers = [j for j in ordered if j != k]
            same_face = float(face_scores[k, peers].max()) if peers else 0.
            same_body = float(body_scores[k, peers].max()) if peers else 0.
            other_face = float(face_scores[k, outside].max()) if outside else 0.
            method = 'face' if same_face >= face_threshold else 'body' if same_body >= body_threshold else 'unresolved'
            uncertain = method == 'unresolved' or (method == 'face' and same_face-other_face < .05)
            o.update(entity_id=entity_id, identity_method=method, identity_uncertain=uncertain,
                     face_similarity=same_face, body_similarity=same_body,
                     identity_reasons=[f'{method} 외형 비교; 동일 프레임 동시 등장 병합 금지'])
    return tracks, links


def associate_people(frames):
    """Anchor identities with repeated face evidence, then assign whole frames.

    Assignment is one-to-one, with an explicit unmatched alternative. Body
    appearance bridges rear views; it cannot change the stable face anchors.
    """
    from scipy.optimize import linear_sum_assignment
    seed_tracks, _ = cluster_people(frames, face_threshold=.65, body_threshold=1.1)
    observations = [o for f in frames for o in f['observations']]
    groups = {eid: [i for i, o in enumerate(observations) if o['entity_id'] == eid] for eid in seed_tracks}
    anchors = [group for group in groups.values() if len(group) >= 6]
    # Short clips still work: seed from their strongest face groups, preserving
    # co-visibility rather than assuming any particular number of performers.
    if not anchors:
        return cluster_people(frames, face_threshold=.65, body_threshold=.82)[0]
    anchors.sort(key=min)
    body = np.asarray([o['body_feature'] for o in observations])
    face = np.asarray([o['face_feature'] or [0.] * 128 for o in observations])
    bs, fs = body @ body.T, face @ face.T
    assigned = {i: t for t, group in enumerate(anchors) for i in group}
    galleries = [list(group) for group in anchors]
    reason = {i: ('face_anchor', float(fs[i, [j for j in anchors[t] if j != i]].max()), False)
              for i, t in assigned.items()}
    index = {o['detection_id']: i for i, o in enumerate(observations)}
    pending = set(range(len(observations))) - assigned.keys()
    # Forward and reverse passes propagate only clear appearance decisions;
    # camera cuts and screen coordinates do not imply identity.
    for sweep in range(3):
        for frame in (frames if sweep % 2 == 0 else frames[::-1]):
            ids = [index[o['detection_id']] for o in frame['observations']]
            queries = [i for i in ids if i in pending]
            occupied = {assigned[i] for i in ids if i in assigned}
            candidates = [t for t in range(len(anchors)) if t not in occupied]
            if not queries or not candidates:
                continue
            scores = np.full((len(queries), len(candidates) + len(queries)), .0)
            evidence = {}
            for r, i in enumerate(queries):
                for c, t in enumerate(candidates):
                    body_score = float(bs[i, galleries[t]].max())
                    face_score = float(fs[i, anchors[t]].max())
                    x1, y1, x2, y2 = observations[i]['bbox']
                    clipped = x1 <= 2 or x2 >= 998
                    # Clothing fragments at the edge are not enough to claim
                    # a person identity, even when their fabric looks similar.
                    if clipped and (observations[i]['face_quality'] < 20 or face_score < .6):
                        scores[r,c] = -1.
                        evidence[r,c] = (body_score, face_score)
                        continue
                    if observations[i]['face_feature'] is not None and observations[i]['face_quality'] >= 20:
                        affinity = .75 * (face_score-.4)/.6 + .25 * (body_score-.65)/.35
                    else:
                        affinity = (body_score-.72)/.28
                    scores[r, c] = affinity
                    evidence[r,c] = (body_score, face_score)
            for r, c in zip(*linear_sum_assignment(-scores)):
                if c >= len(candidates) or scores[r,c] <= 0:
                    continue
                i, t = queries[r], candidates[c]
                alternatives = np.delete(scores[r,:len(candidates)], c)
                margin = float(scores[r,c] - max(0., float(alternatives.max()) if len(alternatives) else 0.))
                # Hold ambiguous crops back during prototype expansion.
                if sweep < 2 and margin < .08:
                    continue
                assigned[i] = t
                pending.remove(i)
                b, f = evidence[r,c]
                method = 'face' if f >= .5 else 'body'
                uncertain = margin < .08
                reason[i] = (method, f if method == 'face' else b, uncertain)
                if not uncertain:
                    galleries[t].append(i)
    # Keep unresolved evidence visible under provisional IDs. Never silently
    # force a partial body crop onto one of the established people.
    for i in sorted(pending):
        assigned[i] = len(galleries)
        galleries.append([i])
        reason[i] = ('unresolved', 0., True)
    tracks = {}
    for t in sorted(set(assigned.values())):
        group = [i for i in range(len(observations)) if assigned[i] == t]
        eid = f'person_{t+1:04d}'
        stable = t < len(anchors)
        good = sorted(group, key=lambda i: (-observations[i]['face_quality'], i))[:3]
        tracks[eid] = {'entity_id': eid, 'label': f'인물 {t+1}' if stable else f'동일인 미확정 {t+1}',
                       'kind': 'person', 'appearance': f'영상에서 P{t+1}로 표시된 인물',
                       'gallery': [observations[i]['crop'] for i in good], 'observations': [],
                       'first_seen_sec': min(observations[i]['time_sec'] for i in group),
                       'last_seen_sec': max(observations[i]['time_sec'] for i in group),
                       'possible_matches': [], 'identity_status': 'matched' if stable else 'uncertain'}
        for i in group:
            method, score, uncertain = reason[i]
            observations[i].update(entity_id=eid, identity_method=method, identity_similarity=score,
                                   identity_uncertain=uncertain,
                                   identity_reasons=[f'{method} 외형 비교; 프레임별 일대일 연결'])
    return tracks


def prepare_people(source, output, settings):
    path = output / 'perception' / 'tracks.json'
    if path.exists():
        data = read_json(path)
        return {f['frame_key']: f for f in data['frames']}, data['tracks']
    frames = extract_people(source, output, settings)
    tracks = associate_people(frames)
    sampled = {f't{round(f.time_sec*1_000_000):012d}'
               for start, end in sliding_windows(source.duration, settings.window_sec, settings.stride_sec)
               for f in source.sample(start, end, settings.sample_fps).frames}
    used = {o['entity_id'] for f in frames if f['frame_key'] in sampled for o in f['observations']}
    tracks = {eid: e for eid, e in tracks.items() if eid in used}
    # Embeddings stay in the evidence cache. Export only auditable decisions.
    for f in frames:
        for o in f['observations']:
            o.pop('body_feature', None)
            o.pop('face_feature', None)
    root = model_files()
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ['yolo11s.pt', 'yunet.onnx', 'sface.onnx', OSNET_FILE]}
    atomic_json(path, {'tracks': tracks, 'frames': frames, 'model_sha256': sources,
                      'identity_policy': 'Face anchors + OSNet appearance, one-to-one assignments; no spatial identity links',
                      'tracking_fps': settings.tracking_fps})
    LOG.info('사람 연결 완료 · 확정 그룹 %d · 미확정 관찰 그룹 %d',
             sum(e['identity_status'] == 'matched' for e in tracks.values()),
             sum(e['identity_status'] != 'matched' for e in tracks.values()))
    return {f['frame_key']: f for f in frames}, tracks


def annotated_clip(clip, grounded):
    """Make detector IDs visible in Qwen's video, leaving source images intact."""
    from PIL import ImageDraw, ImageFont
    from .video import Clip, Frame
    font = ImageFont.load_default(size=15)
    frames = []
    for frame, ground in zip(clip.frames, grounded):
        image = frame.image.copy()
        draw = ImageDraw.Draw(image)
        for box in ground['boxes']:
            if not box['local_id'].startswith('person_'):
                continue
            number = int(box['local_id'].split('_')[1])
            x1, y1, x2, y2 = box['bbox']
            coords = (x1*image.width/1000, y1*image.height/1000, x2*image.width/1000, y2*image.height/1000)
            color = ['#ffff00', '#00ffff', '#ff9999', '#99ff99', '#ff99ff'][number % 5]
            draw.rectangle(coords, outline=color, width=2)
            x, y = coords[:2]
            draw.rectangle((x, y, x+42, y+18), fill='black')
            draw.text((x+1,y), f'P{number}', fill=color, font=font)
        frames.append(Frame(frame.time_sec, image, frame.path))
    return Clip(clip.start_sec, clip.end_sec, clip.source_fps, frames)
