"""Bounded YOLO detection with a configurable small-target tile pass.

The detector never calls Qwen. It makes one full-image pass per sample, then
optionally at most ``max_tiles_per_frame`` tile calls per sample. Tile classes
are configurable. Extra tile boxes must have box height divided by full-frame
height <= ``small_target_height_ratio``; full-image detections
have no height restriction. All returned boxes are directly observed
detections, not interpolated track boxes.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import tempfile
import time
from urllib.request import urlopen

from low_freq_event.storage import atomic_json, read_json


LOG = logging.getLogger('entities.detector')
VERSION = 3
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StrongDetectorConfig:
    model: str = 'yolo11x.pt'
    imgsz: int = 1280
    person_confidence: float = .25
    object_confidence: float = .25
    iou: float = .55
    half: bool = True
    tile_mode: str = 'adaptive'  # off, adaptive, or all
    tile_size: int = 768
    tile_overlap: float = .20
    tile_imgsz: int = 1280
    max_tiles_per_frame: int = 6
    small_target_height_ratio: float = .09
    tile_categories: tuple[str, ...] = ('person',)
    duplicate_iou: float = .55
    max_detections: int = 100

    def __post_init__(self):
        if self.tile_mode not in {'off', 'adaptive', 'all'}:
            raise ValueError('tile_mode must be off, adaptive, or all')
        for key in ('imgsz', 'tile_imgsz'):
            value = getattr(self, key)
            if not isinstance(value, int) or not 320 <= value <= 1920:
                raise ValueError(f'{key} must be an integer in 320..1920')
        if not isinstance(self.tile_size, int) or self.tile_size < 256:
            raise ValueError('tile_size must be an integer >= 256')
        if (not isinstance(self.tile_overlap, (int, float))
                or not math.isfinite(self.tile_overlap) or not 0 <= self.tile_overlap < .75):
            raise ValueError('tile_overlap must be in [0, .75)')
        if not isinstance(self.max_tiles_per_frame, int) or not 1 <= self.max_tiles_per_frame <= 6:
            raise ValueError('max_tiles_per_frame must be an integer in 1..6')
        if not isinstance(self.max_detections, int) or not 1 <= self.max_detections <= 1000:
            raise ValueError('max_detections must be an integer in 1..1000')
        if (not isinstance(self.small_target_height_ratio, (int, float))
                or not math.isfinite(self.small_target_height_ratio)
                or not 0 < self.small_target_height_ratio <= 1):
            raise ValueError('small_target_height_ratio must be in (0, 1]')
        categories = self.tile_categories
        if isinstance(categories, str):
            categories = categories.split(',')
        if (not isinstance(categories, (tuple, list)) or not categories
                or any(not isinstance(value, str) or not value.strip() for value in categories)):
            raise ValueError('tile_categories must contain detector class names, or only all')
        categories = tuple(sorted({value.strip().casefold() for value in categories}))
        if 'all' in categories and len(categories) != 1:
            raise ValueError('tile_categories all cannot be combined with individual classes')
        object.__setattr__(self, 'tile_categories', categories)
        for key in ('person_confidence', 'object_confidence', 'iou', 'duplicate_iou'):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{key} must be between zero and one')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _weights_path(name):
    """Only known official names auto-download; custom paths must exist."""
    supplied = Path(name).expanduser()
    if supplied.is_file():
        return supplied.resolve()
    if supplied.name != name or not re.fullmatch(r'yolo11[nslmx]\.pt', name):
        raise FileNotFoundError(f'Detector weights not found: {supplied}')
    for path in (ROOT / 'assets' / 'models' / name, ROOT.parent / '01_low_freq_event' / 'models' / name):
        if path.is_file():
            return path.resolve()
    destination = ROOT / 'assets' / 'models' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}'
    LOG.info('Downloading official Ultralytics detector weights: %s', url)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f'.{name}.', delete=False) as target:
            temporary = Path(target.name)
            with urlopen(url, timeout=120) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(block)
        if temporary.stat().st_size < 1_000_000:
            raise ValueError('Downloaded detector weights are unexpectedly small')
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination.resolve()


def tile_rectangles(width, height, config):
    """Cover the whole image with bounded, overlapping tiles, including edges.

    Very large inputs enlarge the tiles instead of silently dropping a spatial
    region when the configured maximum number of calls would be exceeded.
    """
    def starts(length, side):
        side = min(side, length)
        if side == length:
            return [0]
        stride = max(1, int(side * (1 - config.tile_overlap)))
        count = math.ceil((length - side) / stride) + 1
        return sorted({round(i * (length - side) / (count - 1)) for i in range(count)})

    side = min(config.tile_size, max(width, height))
    while True:
        xs, ys = starts(width, side), starts(height, side)
        if len(xs) * len(ys) <= config.max_tiles_per_frame:
            return [(x, y, min(width, x + side), min(height, y + side)) for y in ys for x in xs]
        side = min(max(width, height), max(side + 1, math.ceil(side * 1.1)))


def box_iou(left, right):
    overlap = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(
        0., min(left[3], right[3]) - max(left[1], right[1]))
    area_a = max(0., left[2] - left[0]) * max(0., left[3] - left[1])
    area_b = max(0., right[2] - right[0]) * max(0., right[3] - right[1])
    return overlap / max(area_a + area_b - overlap, 1e-12)


def suppress_duplicates(rows, threshold=.55):
    """Suppress same-class duplicate geometry, retaining nested individuals.

    Deliberately do not use intersection-over-smaller-area: a smaller target
    inside another target's rectangle must remain a separate detection.
    Near-identical non-person boxes with conflicting detector labels retain
    the strongest geometry and record the alternative labels. This does not
    suppress a smaller target merely because it is inside a larger rectangle.
    """
    result = []
    for original in sorted(rows, key=lambda r: (-r['confidence'], r['category'], r['bbox'])):
        row = dict(original)
        duplicate = next((old for old in result if
                          (row['category'] == old['category'] and box_iou(row['bbox'], old['bbox']) >= threshold)
                          or (row['category'] != 'person' and old['category'] != 'person'
                              and box_iou(row['bbox'], old['bbox']) >= .90)), None)
        if duplicate is not None:
            alternatives = set(duplicate.get('alternative_categories', [])) | set(row.get('alternative_categories', []))
            if row['category'] != duplicate['category']:
                alternatives.add(row['category'])
            alternatives.discard(duplicate['category'])
            if alternatives:
                duplicate['alternative_categories'] = sorted(alternatives)
            continue
        result.append(row)
    return sorted(result, key=lambda r: (r['bbox'][0], r['bbox'][1], r['category']))


def _tile_class_ids(model, config):
    """Resolve configured names against the selected model, including custom models."""
    if config.tile_categories == ('all',):
        return None
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    available = {str(name).casefold(): int(index) for index, name in items}
    missing = set(config.tile_categories) - set(available)
    if missing:
        raise ValueError('Tile categories absent from the selected detector: ' + ', '.join(sorted(missing)))
    return [available[name] for name in config.tile_categories]


def _tile_target(row, config):
    return config.tile_categories == ('all',) or row['category'].casefold() in config.tile_categories


def _predict(model, image, config, device, *, source, rect=None, full_size=None):
    """Return normalized boxes and provenance using the actual image geometry."""
    import torch
    tiled = rect is not None
    kwargs = dict(conf=min(config.person_confidence, config.object_confidence),
                  imgsz=config.tile_imgsz if tiled else config.imgsz, iou=config.iou,
                  device=device, verbose=False, half=config.half and device.type == 'cuda',
                  agnostic_nms=False, max_det=config.max_detections)
    if tiled:
        class_ids = _tile_class_ids(model, config)
        if class_ids is not None:
            kwargs['classes'] = class_ids
    with torch.inference_mode():
        result = model.predict(image, **kwargs)[0]
    width, height = full_size or image.size
    dx, dy = rect[:2] if tiled else (0, 0)
    records = []
    names = model.names
    for box, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                    result.boxes.cls.cpu().tolist()):
        category = names[int(cls)]
        minimum = config.person_confidence if category == 'person' else config.object_confidence
        if confidence < minimum or not all(math.isfinite(v) for v in [*box, confidence]):
            continue
        xy = [min(width, max(0., box[0] + dx)), min(height, max(0., box[1] + dy)),
              min(width, max(0., box[2] + dx)), min(height, max(0., box[3] + dy))]
        if xy[2] - xy[0] < 2 or xy[3] - xy[1] < 2:
            continue
        # Use relative size so the same policy applies at different resolutions.
        # Medium and large targets remain the full-frame pass's responsibility.
        if tiled and (xy[3] - xy[1]) / height > config.small_target_height_ratio:
            continue
        # A cut-off crop edge can produce a partial target as an extra box. A
        # neighboring overlapping tile owns that subject's complete extent.
        if tiled and ((rect[0] > 0 and box[0] < 2) or (rect[1] > 0 and box[1] < 2)
                      or (rect[2] < width and box[2] > image.width - 2)
                      or (rect[3] < height and box[3] > image.height - 2)):
            continue
        record = dict(category=category, bbox=[v / scale * 1000 for v, scale in zip(xy, [width, height] * 2)],
                      confidence=float(confidence), source=source)
        if tiled:
            record['tile_xyxy'] = list(rect)
        records.append(record)
    return records


def detect_proposals(frames, output, settings, config=None):
    """Detect full-frame entities and recover configured small targets with tiles.

    ``config`` accepts a ``StrongDetectorConfig`` or a dictionary of its fields.
    Outputs use the original pipeline's ``frame_key -> list[proposal]`` schema.
    The metadata sidecar hashes the configuration, weights and input images so
    proposals from an older detector cannot be silently reused.
    """
    import torch
    import ultralytics
    from ultralytics import YOLO
    from PIL import Image

    if config is None:
        config = StrongDetectorConfig()
    elif isinstance(config, dict):
        config = StrongDetectorConfig(**config)
    if not isinstance(config, StrongDetectorConfig):
        raise TypeError('config must be a StrongDetectorConfig or its field dictionary')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = [f['frame_key'] for f in frames]
    if len(set(keys)) != len(keys):
        raise ValueError('Detector input has duplicate frame keys')
    weights = _weights_path(config.model)
    inputs = [dict(frame_key=f['frame_key'], shot_id=f['shot_id'], sha256=_sha256(output / f['image'])) for f in frames]
    manifest = dict(version=VERSION, config=asdict(config), weights_sha256=_sha256(weights),
                    ultralytics_version=ultralytics.__version__, inputs=inputs)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cache_path = output / 'yolo_proposals.json'
    metadata_path = output / 'yolo_proposals.meta.json'
    if cache_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get('fingerprint') == fingerprint and metadata.get('proposals_sha256') == _sha256(cache_path):
            cached = read_json(cache_path)
            if set(cached) == set(keys):
                LOG.info('[CACHE HIT] %s / %s detector boxes reused (no model inference)', output.name, weights.stem)
                return cached
    LOG.info('[CACHE MISS] %s / %s, %d full frames; at most %d extra tiles per frame; categories=%s',
             output.name, weights.stem, len(frames), config.max_tiles_per_frame if config.tile_mode != 'off' else 0,
             ','.join(config.tile_categories))
    policy = getattr(settings, 'perception_device', 'auto')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if policy == 'auto' else policy)
    started = time.perf_counter()
    model = None
    proposals = {}
    per_shot = defaultdict(list)
    rescued_frames = []
    tile_calls = 0
    try:
        model = YOLO(str(weights))
        if config.tile_mode != 'off':
            _tile_class_ids(model, config)  # Reject unsupported classes before any inference.
        for index, frame in enumerate(frames):
            with Image.open(output / frame['image']) as source:
                image = source.convert('RGB')
            proposals[frame['frame_key']] = _predict(model, image, config, device, source=f'{weights.stem}/full')
            per_shot[frame['shot_id']].append(frame)
            if (index + 1) % 25 == 0 or index + 1 == len(frames):
                LOG.info('%s full-image detection %d/%d', weights.stem, index + 1, len(frames))
        for shot_id, shot_frames in per_shot.items():
            shot_small = any(_tile_target(row, config) and (row['bbox'][3] - row['bbox'][1]) /
                             1000 <= config.small_target_height_ratio
                             for f in shot_frames for row in proposals[f['frame_key']])
            for frame in shot_frames:
                key = frame['frame_key']
                targets = [p for p in proposals[key] if _tile_target(p, config)]
                needs_tiles = config.tile_mode == 'all' or (config.tile_mode == 'adaptive' and (shot_small or not targets))
                if not needs_tiles:
                    continue
                with Image.open(output / frame['image']) as source:
                    image = source.convert('RGB')
                extrema = image.getextrema()
                if all(high - low < 2 for low, high in extrema):
                    continue  # Uniform images have no spatial evidence.
                rows = list(proposals[key])
                tiles = tile_rectangles(*image.size, config)
                # One whole-image tile at the same input size brings no extra resolution.
                if len(tiles) == 1 and tiles[0] == (0, 0, *image.size) and config.tile_imgsz <= config.imgsz:
                    continue
                for rect in tiles:
                    rows.extend(_predict(model, image.crop(rect), config, device,
                                         source=f'{weights.stem}/small_target_tile', rect=rect, full_size=image.size))
                    tile_calls += 1
                proposals[key] = suppress_duplicates(rows, config.duplicate_iou)
                rescued_frames.append(dict(frame_key=key, shot_id=shot_id, tile_calls=len(tiles),
                                           targets_before=len(targets),
                                           targets_after=sum(_tile_target(p, config) for p in proposals[key]),
                                           categories=list(config.tile_categories),
                                           maximum_target_height_pixels=image.height * config.small_target_height_ratio))
            LOG.info('%s small-target tiles / %s complete; cumulative %d tiles', weights.stem, shot_id, tile_calls)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    proposals = {key: suppress_duplicates(proposals[key], config.duplicate_iou) for key in keys}
    atomic_json(cache_path, proposals)
    atomic_json(metadata_path, dict(fingerprint=fingerprint, proposals_sha256=_sha256(cache_path), **manifest))
    atomic_json(output / 'detector_summary.json', dict(
        model=weights.name, model_path=str(weights), config=asdict(config), fingerprint=fingerprint,
        sampled_frames=len(frames), full_image_calls=len(frames), tile_calls=tile_calls,
        inference_call_ceiling=len(frames) * (1 + (config.max_tiles_per_frame if config.tile_mode != 'off' else 0)),
        elapsed_sec=time.perf_counter() - started,
        detections=sum(len(value) for value in proposals.values()), rescued_frames=rescued_frames,
        note='All boxes are model detections at sampled times. Tile categories are configurable and tile boxes are limited by their height relative to the full frame. Full-image boxes have no height restriction. Custom targets require detector weights that support those classes.'))
    return proposals
