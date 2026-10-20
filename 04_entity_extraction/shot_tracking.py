#!/usr/bin/env python3
"""Dense, class-agnostic shot tracking on the *original* source frames.

The detector, DINO appearance encoder and Ultralytics BoT-SORT have separate
responsibilities. Detector category changes do not create new tracker classes.
Only observed detector boxes are exported; an occluded Kalman prediction is
never reported as a visible object. All track births, including brief ones,
remain available for the later visual review.

Assets are relative to ``bundle['asset_root']``. Dense observations have null
``image``/``crop`` unless materialized. Every representative and scene frame
has an existing image, and each representative has an existing crop.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent / '01_low_freq_event') not in sys.path:
    sys.path.insert(0, str(ROOT.parent / '01_low_freq_event'))

from low_freq_event.storage import atomic_json, read_json

LOG = logging.getLogger('entities.dense_tracking')
VERSION = 2
# Association changes must replay the existing detector/feature cache instead
# of running the expensive perception stage again.
DETECTOR_CACHE_VERSION = 1


@dataclass(frozen=True)
class DenseTrackingConfig:
    detector_size: int = 1280
    detector_confidence: float = .08
    detector_iou: float = .72
    high_confidence: float = .24
    new_track_confidence: float = .40
    birth_tile_confidence: float = .60
    track_buffer_sec: float = 1.0
    match_threshold: float = .90
    association_proximity: float = .02
    association_fuse_score: bool = False
    camera_motion_method: str = 'none'
    appearance: bool = True
    appearance_model: str = 'facebook/dinov2-small'
    appearance_revision: str = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
    tile_size: int = 640
    tile_overlap: float = .25
    tile_probe_stride: int = 6
    tile_max_calls: int = 6
    tile_target_height_ratio: float = .18
    tile_enabled: bool = True
    representatives: int = 8
    scene_frames: int = 10

    def __post_init__(self):
        if not 320 <= self.detector_size <= 1920:
            raise ValueError('detector_size must be in 320..1920')
        if not 0 < self.detector_confidence < self.high_confidence <= self.new_track_confidence < 1:
            raise ValueError('confidence thresholds must satisfy 0 < low < high <= birth < 1')
        if not self.new_track_confidence <= self.birth_tile_confidence < 1:
            raise ValueError('tile birth confidence must be >= full-frame birth confidence and < 1')
        if self.camera_motion_method not in {'none', 'sparseOptFlow', 'orb', 'sift', 'ecc'}:
            raise ValueError('unsupported camera motion method')
        if self.tile_probe_stride < 1 or self.tile_size < 128 or self.tile_max_calls < 1:
            raise ValueError('tile settings must be positive')
        if not 0 <= self.tile_overlap < .75 or not 0 < self.tile_target_height_ratio <= 1:
            raise ValueError('invalid tile overlap or relative target height')
        if self.representatives < 1 or self.scene_frames < 1 or self.track_buffer_sec <= 0:
            raise ValueError('representative, scene and buffer settings must be positive')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _compatible_detector_cache(output, weights, detector_config, requested_key):
    """Reuse a moving revision alias only after checking its resolved commit.

    Older runs recorded ``main`` in configuration but the actual model commit
    in every frame cache. Pinning that same model must not rerun YOLO/DINO.
    """
    cache_root = output / 'dense_tracking' / 'cache'
    if (cache_root / requested_key).is_dir():
        return requested_key
    previous_path = output / 'dense_tracking' / 'bundle.json'
    if not previous_path.is_file():
        return requested_key
    previous = read_json(previous_path)
    old_config = {key: value for key, value in previous.get('config', {}).items()
                  if key.startswith(('detector_', 'appearance', 'tile_'))}
    compare_old, compare_new = dict(old_config), dict(detector_config)
    compare_old.pop('appearance_revision', None)
    compare_new.pop('appearance_revision', None)
    if compare_old != compare_new or previous.get('detector_model') != str(weights):
        return requested_key
    old_key = previous.get('detector_cache_key') or _digest(
        dict(version=DETECTOR_CACHE_VERSION, weights=_file_digest(weights), config=old_config))
    if not (cache_root / old_key).is_dir():
        return requested_key
    if old_config.get('appearance_revision') == detector_config.get('appearance_revision'):
        return old_key
    expected = detector_config.get('appearance_revision')
    checked = 0
    for path in (cache_root / old_key).glob('*/f*.json'):
        record = read_json(path)
        if any(row.get('feature') for row in record['detections']):
            if record.get('appearance_revision') != expected:
                return requested_key
            checked += 1
    if checked:
        LOG.info('[CACHE VERIFIED] DINO revision alias resolved to requested commit %s (%d frame records)', expected, checked)
        return old_key
    return requested_key


def _resolve_video(raw, metadata, source):
    from input_media import resolve_media_path
    return resolve_media_path(raw, metadata, source)


def discover_shots(source, *, original_video=None, cache_dir=None):
    """Read stage-03 metadata; raw pre-cut videos are also supported."""
    import av
    source = Path(source).expanduser().resolve()
    summary_path = source / 'summary.json' if source.is_dir() else None
    rows = []
    materializers = {}
    original_digests = {}
    if summary_path is not None and summary_path.is_file():
        summary = read_json(summary_path)
        for item in summary['videos']:
            metadata = source / item['samples']
            meta = read_json(metadata)
            recovered = False
            try:
                video = _resolve_video(meta['source'], metadata, source)
            except FileNotFoundError:
                from input_media import ShotMaterializer
                original = original_video
                if original is None and meta.get('original_video'):
                    original = _resolve_video(meta['original_video'], metadata, source)
                if original is None:
                    raise
                original = Path(original).expanduser().resolve()
                if original not in materializers:
                    materializers[original] = ShotMaterializer(original, cache_dir or source / '.recovered_media')
                materializer = materializers[original]
                video = materializer.recover(meta)
                recovered = True
            rows.append(dict(shot_id=item['id'], source=str(video), metadata=str(metadata),
                             fps=float(Fraction(meta['source_fps'])), frame_count=int(meta['source_frames']),
                             original_start_frame=int(meta.get('original_start_frame', 0)),
                             original_fps=float(Fraction(meta.get('original_fps', meta['source_fps']))),
                             original_video=str(materializer.video) if recovered else meta.get('original_video', str(video)),
                             source_metadata={key:meta[key] for key in ('source_sha256','source_fps','source_frames',
                                 'original_start_frame','original_fps','output_timing') if key in meta},
                             adaptive_frames=[int(s['source_frame']) for s in meta.get('samples', [])],
                             adaptive_samples=meta.get('samples', []), source_sha256=_file_digest(video)))
    else:
        videos = [source] if source.is_file() else sorted(
            p for p in source.iterdir() if p.suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi', '.webm'})
        for number, video in enumerate(videos, 1):
            rows.append(dict(shot_id=f'shot_{number:04d}', source=str(video), metadata=None,
                             original_start_frame=0, original_video=str(video),
                             adaptive_frames=[], adaptive_samples=[], source_sha256=_file_digest(video)))
    if not rows:
        raise ValueError(f'No original shot videos found at {source}')
    full_start = 0.
    seen = set()
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['shot_id']) or row['shot_id'] in seen:
            raise ValueError(f'Unsafe or duplicate shot ID: {row["shot_id"]}')
        seen.add(row['shot_id'])
        original_path=Path(original_video or row.get('original_video',row['source'])).expanduser()
        if original_path.is_file():
            original_path=original_path.resolve()
            if original_path not in original_digests:original_digests[original_path]=_file_digest(original_path)
            row['original_sha256']=original_digests[original_path]
        with av.open(row['source']) as container:
            stream = container.streams.video[0]
            rate = float(stream.average_rate or stream.base_rate or 0)
            if rate <= 0:
                raise ValueError(f'Video has no usable frame rate: {row["source"]}')
            row.setdefault('fps', rate)
            if abs(row['fps'] - rate) > .01:
                raise ValueError(f'Original video FPS differs from metadata: {row["source"]}')
            row.setdefault('original_fps', rate)
            row.setdefault('frame_count', int(stream.frames or sum(1 for _ in container.decode(stream))))
            row['width'], row['height'] = stream.width, stream.height
        if row['frame_count'] < 1:
            raise ValueError(f'Empty shot: {row["source"]}')
        row['full_start_sec'] = full_start
        row['duration_sec'] = row['frame_count'] / row['fps']
        full_start += row['duration_sec']
    return rows


def box_iou(left, right):
    intersection = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(
        0., min(left[3], right[3]) - max(left[1], right[1]))
    area_a = max(0., left[2] - left[0]) * max(0., left[3] - left[1])
    area_b = max(0., right[2] - right[0]) * max(0., right[3] - right[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


def fuse_duplicate_boxes(rows, threshold=.86):
    """Merge nearly identical boxes across labels; keep nested separate subjects."""
    kept = []
    for source in sorted(rows, key=lambda row: -row['confidence']):
        row = dict(source)
        duplicate = next((old for old in kept if box_iou(old['bbox'], row['bbox']) >= threshold), None)
        if duplicate is None:
            kept.append(row)
            continue
        categories = set(duplicate.get('alternative_categories', [])) | set(row.get('alternative_categories', []))
        categories.add(row['category'])
        categories.discard(duplicate['category'])
        duplicate['alternative_categories'] = sorted(categories)
        duplicate['duplicate_sources'] = sorted(set(duplicate.get('duplicate_sources', [duplicate['source']]))
                                                | {row['source']})
    return sorted(kept, key=lambda row: (row['bbox'][0], row['bbox'][1], -row['confidence']))


def _tiles(width, height, config):
    def positions(length, side):
        if side >= length:
            return [0]
        number = math.ceil((length - side) / (side * (1 - config.tile_overlap))) + 1
        return [round(i * (length - side) / (number - 1)) for i in range(number)]
    side = min(config.tile_size, max(width, height))
    while True:
        xs, ys = positions(width, side), positions(height, side)
        if len(xs) * len(ys) <= config.tile_max_calls:
            return [(x, y, min(x + side, width), min(y + side, height)) for y in ys for x in xs]
        side = min(max(width, height), max(side + 1, math.ceil(side * 1.1)))


class DinoEncoder:
    """A single generic appearance space avoids class-flip discontinuities."""
    def __init__(self, config, device):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(config.appearance_model, revision=config.appearance_revision,
                                                           use_fast=False)
        self.model = AutoModel.from_pretrained(config.appearance_model, revision=config.appearance_revision).to(self.device).eval()
        self.revision = getattr(self.model.config, '_commit_hash', None) or config.appearance_revision

    def __call__(self, image, rows):
        import numpy as np
        import torch
        if not rows:
            return []
        crops = [image.crop(tuple(int(round(v)) for v in row['bbox'])) for row in rows]
        vectors = []
        for start in range(0, len(crops), 16):
            batch = self.processor(images=crops[start:start + 16], return_tensors='pt').to(self.device)
            with torch.inference_mode():
                features = self.model(**batch).last_hidden_state[:, 0].float()
                features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
            vectors.extend(np.round(features, 7).tolist())
        return vectors


class DetectionBatch:
    """Carry original indices through BoT-SORT high/low confidence slicing."""
    def __init__(self, boxes, indices=None):
        import numpy as np
        self.boxes = boxes
        self.indices = np.arange(len(boxes)) if indices is None else indices

    def __len__(self): return len(self.boxes)
    def __getitem__(self, index): return DetectionBatch(self.boxes[index], self.indices[index])
    @property
    def xyxy(self): return self.boxes.xyxy
    @property
    def xywh(self): return self.boxes.xywh
    @property
    def conf(self): return self.boxes.conf
    @property
    def cls(self): return self.boxes.cls


def make_tracker(fps, config):
    from ultralytics.trackers.bot_sort import BOTSORT

    class IndexedBoTSORT(BOTSORT):
        def init_track(self, results, img=None):
            tracks = super().init_track(results, img)
            for track, index in zip(tracks, results.indices):
                track.idx = int(index)
            return tracks

    args = SimpleNamespace(track_high_thresh=config.high_confidence,
                           track_low_thresh=config.detector_confidence - 1e-6,
                           new_track_thresh=config.new_track_confidence,
                           track_buffer=max(1, round(config.track_buffer_sec * 30)),
                           match_thresh=config.match_threshold, fuse_score=config.association_fuse_score,
                           gmc_method=config.camera_motion_method, proximity_thresh=config.association_proximity,
                           appearance_thresh=.80, with_reid=config.appearance, model='auto')
    tracker = IndexedBoTSORT(args, frame_rate=fps)
    if config.appearance:
        tracker.encoder = lambda features, boxes: list(features)
    return tracker


def tracking_confidence(row, config):
    """Tile context is weaker: its low scores may continue, but not start, a track.

    Original detector confidence remains intact in cache and exported evidence.
    This decision is independent of detector class (including human/nonhuman).
    """
    value = row['confidence']
    if row['source'] == 'yolo_small_target_tile' and value < config.birth_tile_confidence:
        return min(value, config.high_confidence - .001)
    return value


def _predict(detector, image, config, device, *, rect=None, full_size=None):
    width, height = full_size or image.size
    dx, dy = rect[:2] if rect is not None else (0, 0)
    result = detector.predict(image, imgsz=config.detector_size, conf=config.detector_confidence,
                              iou=config.detector_iou, agnostic_nms=True, device=device,
                              half=str(device).startswith('cuda'), max_det=200, verbose=False)[0]
    rows = []
    for bbox, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                     result.boxes.cls.cpu().tolist()):
        box = [max(0., min(width, bbox[0] + dx)), max(0., min(height, bbox[1] + dy)),
               max(0., min(width, bbox[2] + dx)), max(0., min(height, bbox[3] + dy))]
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        if rect is not None:
            if (box[3] - box[1]) / height > config.tile_target_height_ratio:
                continue
            if ((rect[0] > 0 and bbox[0] < 3) or (rect[1] > 0 and bbox[1] < 3)
                    or (rect[2] < width and bbox[2] > image.width - 3)
                    or (rect[3] < height and bbox[3] > image.height - 3)):
                continue
        rows.append(dict(bbox=[round(v, 3) for v in box], confidence=float(confidence),
                         class_id=int(cls), category=str(detector.names[int(cls)]),
                         source='yolo_full' if rect is None else 'yolo_small_target_tile'))
    return rows


def select_representatives(observations, limit=8):
    """Choose endpoints, a clear view, and temporal/appearance diverse views."""
    import numpy as np
    if len(observations) <= limit:
        return list(observations)
    best = max(range(len(observations)), key=lambda i: observations[i]['confidence'] * math.sqrt(
        max(1., (observations[i]['bbox'][2] - observations[i]['bbox'][0]) *
            (observations[i]['bbox'][3] - observations[i]['bbox'][1]))))
    chosen = list(dict.fromkeys([best, 0, len(observations) - 1]))[:limit]
    span = max(1, observations[-1]['frame'] - observations[0]['frame'])
    while len(chosen) < limit:
        def merit(index):
            row = observations[index]
            temporal = min(abs(row['frame'] - observations[j]['frame']) / span for j in chosen)
            feature = row.get('_feature') or row.get('feature')
            refs = [observations[j].get('_feature') or observations[j].get('feature') for j in chosen]
            appearance = min((max(0., 1. - float(np.dot(feature, ref))) for ref in refs if ref), default=0.) if feature else 0.
            return .65 * temporal + .35 * appearance + .015 * row['confidence']
        chosen.append(max((i for i in range(len(observations)) if i not in chosen), key=merit))
    return [observations[i] for i in sorted(chosen)]


def _finish_tracks(groups, fps, config):
    tracks = []
    for number, (raw_id, observations) in enumerate(sorted(groups.items(), key=lambda item: (item[1][0]['frame'], item[0])), 1):
        categories = Counter()
        for obs in observations:
            categories[obs['category']] += obs['confidence']
        track_id = f'{observations[0]["shot_id"]}:T{number:04d}'
        for obs in observations:
            obs['track_id'] = track_id
        flags = []
        if len(observations) == 1:
            flags.append('single_observed_frame')
        if max(o['confidence'] for o in observations) < .4:
            flags.append('weak_detector_evidence')
        if 'person' in categories and len(categories) > 1:
            flags.append('human_nonhuman_class_conflict')
        gaps = [dict(start_frame=a['frame'] + 1, end_frame_exclusive=b['frame'],
                     duration_sec=(b['frame'] - a['frame'] - 1) / fps)
                for a, b in zip(observations, observations[1:]) if b['frame'] > a['frame'] + 1]
        span = observations[-1]['frame'] - observations[0]['frame'] + 1
        tracks.append(dict(track_id=track_id, raw_tracker_id=int(raw_id), observations=observations,
                           representatives=select_representatives(observations, config.representatives),
                           category=categories.most_common(1)[0][0], category_evidence=dict(categories),
                           first_frame=observations[0]['frame'], last_frame=observations[-1]['frame'],
                           observed_frames=len(observations), observed_duration_sec=len(observations) / fps,
                           span_duration_sec=span / fps, observation_coverage=len(observations) / span,
                           mean_confidence=sum(o['confidence'] for o in observations) / len(observations),
                           occlusion_gaps=gaps, quality_flags=flags, uncertain=bool(flags),
                           classification_status='provisional_detector_label'))
    return tracks


def _materialize_assets(shot, output, config):
    import av
    import numpy as np
    uniform = {int(round(v)) for v in np.linspace(0, shot['frame_count'] - 1,
                                                min(config.scene_frames, shot['frame_count']))}
    if shot.get('analysis_frames') == 'sampled':
        frames = shot['adaptive_frames']
        uniform = {frames[round(i)] for i in np.linspace(0, len(frames)-1, min(config.scene_frames,len(frames)))}
    by_frame = defaultdict(list)
    wanted = uniform | set(shot['adaptive_frames'])
    for track in shot['tracks']:
        for obs in track['representatives']:
            wanted.add(obs['frame'])
            by_frame[obs['frame']].append(obs)
    scenes = []
    saved = set()
    from sampled_frames import iter_shot_images
    for index, image in iter_shot_images(shot, wanted):
        image_rel = f'dense_tracking/frames/{shot["shot_id"]}/f{index:08d}.jpg'
        destination = output / image_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=94)
        saved.add(index)
        if index in uniform:
            scenes.append(dict(frame=index, source_frame=index, time_sec=index / shot['fps'],
                               original_time_sec=shot['original_start_frame'] / shot['original_fps'] + index / shot['fps'],
                               image=image_rel))
        for obs in by_frame[index]:
            crop_rel = f'dense_tracking/crops/{shot["shot_id"]}/{obs["track_id"].split(":")[-1]}_f{index:08d}.jpg'
            destination = output / crop_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = obs['bbox']
            image.crop((int(x1), int(y1), math.ceil(x2), math.ceil(y2))).save(destination, quality=96)
            obs['image'], obs['crop'] = image_rel, crop_rel
            if obs.get('_feature'):
                obs['feature'] = obs['_feature']
    if saved != wanted:
        raise ValueError(f'Could not materialize requested frames in {shot["shot_id"]}: {sorted(wanted - saved)}')
    shot['scene_frames'] = scenes
    shot['adaptive_scene_frames'] = [dict(frame=i, time_sec=i / shot['fps'],
                                        image=f'dense_tracking/frames/{shot["shot_id"]}/f{i:08d}.jpg')
                                     for i in shot['adaptive_frames']]
    for track in shot['tracks']:
        for obs in track['observations']:
            if obs['frame'] in saved:
                obs['image'] = f'dense_tracking/frames/{shot["shot_id"]}/f{obs["frame"]:08d}.jpg'
            obs.pop('_feature', None)


def _assets_valid(bundle, output):
    return all((output / row['image']).is_file() for shot in bundle['shots']
               for row in shot['scene_frames'] + shot.get('adaptive_scene_frames', [])) and all(
        row.get('crop') and (output / row['crop']).is_file() and row.get('image') and (output / row['image']).is_file()
        for shot in bundle['shots'] for track in shot['tracks'] for row in track['representatives'])


def build_dense_tracks(source: Path, output: Path, *, model='yolo11x.pt', device='cuda:0', force=False,
                       config: DenseTrackingConfig | dict | None = None, original_video=None, analysis_frames="dense"):
    """Build/reuse dense detector cache and class-agnostic, per-shot BoT-SORT tracks."""
    import av
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.engine.results import Boxes
    from strong_detector import _weights_path

    config = DenseTrackingConfig(**config) if isinstance(config, dict) else config or DenseTrackingConfig()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    from sampled_frames import discover_sampled_shots, iter_shot_images, SparseTracker
    shots = (discover_sampled_shots(source) if analysis_frames == 'sampled' else
             discover_shots(source, original_video=original_video, cache_dir=output / 'input_media'))
    weights = _weights_path(model)
    detector_config = {key: value for key, value in asdict(config).items()
                       if key.startswith(('detector_', 'appearance', 'tile_'))}
    detector_key = _digest(dict(version=DETECTOR_CACHE_VERSION, weights=_file_digest(weights), config=detector_config))
    if analysis_frames=='sampled':detector_key=_digest(dict(sampled_version=1,key=detector_key))
    bundle_key = _digest(dict(version=VERSION, shots=shots, detector=detector_key, config=asdict(config)))
    bundle_path = output / 'dense_tracking' / 'bundle.json'
    if bundle_path.is_file() and not force:
        cached = read_json(bundle_path)
        if cached.get('fingerprint') == bundle_key and _assets_valid(cached, output):
            LOG.info('[CACHE HIT] dense original-frame tracking: %d shots', len(cached['shots']))
            return cached
    detector_cache_key = detector_key if analysis_frames=='sampled' else _compatible_detector_cache(output, weights, detector_config, detector_key)
    started = time.perf_counter()
    detector = encoder = None
    totals = Counter()
    built_shots = []
    try:
        for shot_number, shot in enumerate(shots, 1):
            shot_cache_key = _digest(dict(bundle=bundle_key, shot=shot['shot_id']))
            track_path = output / 'dense_tracking' / 'tracks' / f'{shot["shot_id"]}.json'
            if not force and track_path.is_file():
                cached = read_json(track_path)
                if cached.get('fingerprint') == shot_cache_key and _assets_valid({'shots': [cached]}, output):
                    LOG.info('[CACHE HIT] dense shot %s', shot['shot_id'])
                    built_shots.append(cached)
                    totals.update(cached['stats'])
                    continue
            LOG.info('[TRACK %d/%d] %s: %d source frames, %.3f fps', shot_number, len(shots),
                     shot['shot_id'], shot['frame_count'], shot['fps'])
            tracker = (SparseTracker(shot['fps'], config) if analysis_frames == 'sampled' else make_tracker(shot['fps'], config))
            groups = defaultdict(list)
            candidates = []
            frame_stats = Counter()
            decoded = 0
            previous_tiny = False
            cache_root = output / 'dense_tracking' / 'cache' / detector_cache_key / shot['source_sha256']
            from tqdm.auto import tqdm
            expected = len(shot['adaptive_frames']) if analysis_frames == 'sampled' else shot['frame_count']
            for index, image in tqdm(iter_shot_images(shot), total=expected,
                                     desc=f"검출 {shot_number}/{len(shots)} {shot['shot_id']}", unit='frame'):
                if image.size != (shot['width'], shot['height']):
                    raise ValueError(f'Frame size changed inside shot {shot["shot_id"]}')
                cache_path = cache_root / f'f{index:08d}.json'
                if not force and cache_path.is_file():
                    cached = read_json(cache_path)
                    rows = cached['detections']
                    frame_stats['detector_cache_frames'] += 1
                    frame_stats['tile_passes'] += cached.get('tile_passes', 0)
                else:
                    if detector is None:
                        LOG.info('Loading dense detector: %s on %s', weights.name, device)
                        detector = YOLO(str(weights))
                    rows = _predict(detector, image, config, device)
                    tile_calls = 0
                    tiny = any((r['bbox'][3] - r['bbox'][1]) / image.height < config.tile_target_height_ratio
                               and r['confidence'] >= config.high_confidence for r in rows)
                    if config.tile_enabled and (index % config.tile_probe_stride == 0 or tiny or previous_tiny):
                        for rect in _tiles(image.width, image.height, config):
                            if rect == (0, 0, image.width, image.height):
                                continue
                            rows.extend(_predict(detector, image.crop(rect), config, device, rect=rect, full_size=image.size))
                            tile_calls += 1
                    rows = fuse_duplicate_boxes(rows)
                    if config.appearance and rows:
                        if encoder is None:
                            LOG.info('Loading generic tracking appearance encoder: %s', config.appearance_model)
                            encoder = DinoEncoder(config, device)
                        for row, feature in zip(rows, encoder(image, rows)):
                            row['feature'] = feature
                    atomic_json(cache_path, dict(frame=index, detections=rows, tile_passes=tile_calls,
                                                 appearance_revision=getattr(encoder, 'revision', None)))
                    frame_stats['detector_inferred_frames'] += 1
                    frame_stats['tile_passes'] += tile_calls
                previous_tiny = any((r['bbox'][3] - r['bbox'][1]) / image.height < config.tile_target_height_ratio
                                    and r['confidence'] >= config.high_confidence for r in rows)
                if analysis_frames == 'sampled':
                    visible = tracker.update(rows, index)
                else:
                    raw = np.asarray([r['bbox'] + [tracking_confidence(r, config), 0.] for r in rows],
                                     dtype=np.float32).reshape(-1, 6)
                    batch = DetectionBatch(Boxes(raw, (image.height, image.width)))
                    features = np.asarray([r['feature'] for r in rows], dtype=np.float32) if config.appearance and rows else None
                    tracker.update(batch, img=np.asarray(image)[:, :, ::-1].copy(), feats=features)
                    visible = [t for t in tracker.tracked_stracks if t.frame_id == tracker.frame_id]
                associated = {int(t.idx) for t in visible}
                # Preserve unassociated proposals as explicit recovery evidence;
                # they are not each promoted to a physical entity identity.
                unassociated = [dict(bbox=row['bbox'], confidence=row['confidence'], category=row['category'],
                                     source=row['source'], reason='not_associated_or_insufficient_birth_evidence')
                                for i, row in enumerate(rows) if i not in associated]
                if unassociated:
                    candidates.append(dict(frame=index, time_sec=index / shot['fps'], detections=unassociated))
                for track in visible:
                    row = rows[int(track.idx)]
                    obs = dict(frame=index, source_frame=index, shot_id=shot['shot_id'],
                               time_sec=index / shot['fps'],
                               full_time_sec=shot['full_start_sec'] + index / shot['fps'],
                               original_time_sec=shot['original_start_frame'] / shot['original_fps'] + index / shot['fps'],
                               original_frame=shot['original_start_frame'] + round(index / shot['fps'] * shot['original_fps']),
                               bbox=row['bbox'], confidence=row['confidence'], category=row['category'],
                               alternative_categories=row.get('alternative_categories', []),
                               detector_class_id=row['class_id'], detection_source=row['source'],
                               evidence='observed_detection', tracker_confirmed=bool(track.is_activated),
                               image=None, crop=None, _feature=row.get('feature'))
                    groups[int(track.track_id)].append(obs)
                frame_stats['analyzed_frames'] += 1
                frame_stats['sampled_frames' if analysis_frames == 'sampled' else 'dense_frames'] += 1
                frame_stats['detections'] += len(rows)
                frame_stats['observed_track_boxes'] += len(visible)
                frame_stats['unassociated_low_confidence_boxes'] += len(rows) - len(visible)
                decoded += 1
                if (index + 1) % 50 == 0:
                    LOG.info('  %s %d/%d frames; %d track births', shot['shot_id'], index + 1,
                             shot['frame_count'], len(groups))
            if decoded != expected:
                raise ValueError(f'Original frame count mismatch: {shot["shot_id"]}: {decoded} != {shot["frame_count"]}')
            shot['tracks'] = _finish_tracks(groups, shot['fps'], config)
            proposal_path = output / 'dense_tracking' / 'candidates' / f'{shot["shot_id"]}.json'
            atomic_json(proposal_path, dict(shot_id=shot['shot_id'], frames=candidates,
                        description='Observed detector proposals not assigned to a track; retained for scene-driven recovery, not asserted entities.'))
            shot['candidate_evidence'] = dict(path=str(proposal_path.relative_to(output)),
                detector_cache=str(cache_root.relative_to(output)),
                count=sum(len(item['detections']) for item in candidates),
                full_frame_birth_confidence=config.new_track_confidence,
                tile_birth_confidence=config.birth_tile_confidence,
                policy='Full scene review may request recovery from these retained proposals. Low-score tile boxes can continue existing tracks.')
            _materialize_assets(shot, output, config)
            frame_stats['tracks'] = len(shot['tracks'])
            frame_stats['singleton_tracks'] = sum(len(t['observations']) == 1 for t in shot['tracks'])
            shot.update(stats=dict(frame_stats), fingerprint=shot_cache_key)
            atomic_json(track_path, shot)
            built_shots.append(shot)
            totals.update(frame_stats)
            LOG.info('  %s complete: %d tracks, %d singleton tracks, %d observed boxes', shot['shot_id'],
                     len(shot['tracks']), frame_stats['singleton_tracks'], frame_stats['observed_track_boxes'])
    finally:
        detector = encoder = None
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    bundle = dict(version=VERSION, fingerprint=bundle_key, asset_root=str(output), source=str(Path(source).resolve()), analysis_frames=analysis_frames,
                  config=asdict(config), detector_model=str(weights), tracker='Sparse time-aware association' if analysis_frames == 'sampled' else 'Ultralytics BoT-SORT',
                  detector_cache_key=detector_cache_key,
                  tracker_class_policy='one class-agnostic tracker per shot; detector classes are provisional evidence',
                  appearance_model=config.appearance_model if config.appearance else None,
                  timing_policy=f'{analysis_frames} frames; observations retain original and concatenated timestamps',
                  visibility_policy='only detector-observed boxes; gaps are unknown, never fabricated visibility',
                  shots=built_shots, stats=dict(totals), elapsed_sec=time.perf_counter() - started)
    atomic_json(bundle_path, bundle)
    LOG.info('Dense tracking complete: %d frames, %d tracks, %.1f sec', totals['analyzed_frames'], totals['tracks'],
             bundle['elapsed_sec'])
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--original-video', type=Path, help='Recover unavailable shots from original frame intervals')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='yolo11x.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-appearance', action='store_true')
    parser.add_argument('--no-tiles', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    bundle = build_dense_tracks(args.source, args.output, model=args.model, device=args.device, force=args.force,
                                original_video=args.original_video,
                                config=DenseTrackingConfig(appearance=not args.no_appearance, tile_enabled=not args.no_tiles))
    print(json.dumps(dict(bundle=str(args.output / 'dense_tracking' / 'bundle.json'), stats=bundle['stats']), ensure_ascii=False))


if __name__ == '__main__':
    main()
