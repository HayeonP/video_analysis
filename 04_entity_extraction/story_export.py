"""Export a shot-first identity story without changing its source time axis.

Model-free module: all identity decisions are supplied by the caller.  Box
coordinates in the tracking bundle are pixels; legacy viewer observations are
also exported with explicit normalized coordinates.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def _rate(value):
    # Tracker metadata may serialize a rational rate (30000/1001) as float.
    return Fraction(str(value)).limit_denominator(1_000_000)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def box_iou(a, b):
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-12)


def _box_area(box):
    return max(0., box[2]-box[0])*max(0., box[3]-box[1])


def _intersection_over_smaller(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    return intersection/max(1e-12, min(_box_area(a), _box_area(b)))


def entity_color(entity_id):
    return tuple(90 + byte % 150 for byte in hashlib.sha256(entity_id.encode()).digest()[:3])


def _confidence(row):
    for key in ('confidence', 'detection_confidence'):
        value = row.get(key)
        if isinstance(value, (float, int)) and math.isfinite(value):
            return float(value)
    return 0.


def _observation_provenance(observation):
    """Keep detector evidence distinct from a propagated SAM mask's score."""
    kind = next((observation[key] for key in ('observation_type', 'evidence', 'detection_source', 'source')
                 if isinstance(observation.get(key), str) and observation[key]), 'detected')
    fields = ('evidence', 'detection_source', 'confidence_semantics', 'mask', 'masked_crop',
              'masked_crop_background', 'mask_quality', 'detector_supported', 'nearest_detector_frame',
              'source_track_ids', 'source_track_id', 'tracker_confirmed', 'mask_refinement_status',
              'original_box_candidates', 'feature_provenance')
    result = {key: deepcopy(observation[key]) for key in fields if key in observation}
    result['observation_type'] = kind
    score = _confidence(observation)
    result['confidence'] = score
    is_mask = 'sam2_mask' in (kind, observation.get('evidence'), observation.get('detection_source'), observation.get('source'))
    # Do not rename an uncalibrated mask stability score as YOLO confidence.
    result['detection_confidence'] = observation.get('detection_confidence', None if is_mask else score)
    if 'confidence_semantics' not in result:
        result['confidence_semantics'] = 'uncalibrated mask score' if is_mask else 'source observation confidence'
    return result


def _font(size):
    from PIL import ImageFont
    for path in (ROOT/'assets/NanumGothic-Regular.ttf', Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _asset_path(value, bundle, output):
    if not value:
        return None
    path = Path(value)
    roots = [Path(bundle.get('asset_root', output)), output, ROOT]
    for candidate in ([path] if path.is_absolute() else [root/path for root in roots]):
        if candidate.is_file():
            return candidate
    return None


def _track_id(value):
    return str(value.get('track_id', value.get('id', ''))) if isinstance(value, dict) else str(value)


def validate_story_contract(bundle, local, global_result):
    """Return complete coverage accounting; malformed identity partitions fail."""
    shots = {str(s['shot_id']): s for s in bundle['shots']}
    locals_by_shot = {str(s['shot_id']): s for s in local['shots']}
    if set(shots) != set(locals_by_shot):
        raise ValueError('Local review must cover every shot exactly once.')
    if len(locals_by_shot) != len(local['shots']) or len(shots) != len(bundle['shots']):
        raise ValueError('Duplicate shot IDs are not allowed.')
    global_entities = {str(e['entity_id']): e for e in global_result['entities']}
    if len(global_entities) != len(global_result['entities']):
        raise ValueError('Duplicate global Entity IDs.')
    mapping = global_result['mapping']
    seen_local = set()
    assigned = rejected = unreviewed = 0
    unreviewed_rows = []
    for shot_id, shot in shots.items():
        tracks = {str(t['track_id']) for t in shot['tracks']}
        if len(tracks) != len(shot['tracks']):
            raise ValueError(f'Duplicate track IDs in {shot_id}.')
        local_shot = locals_by_shot[shot_id]
        owners = {}
        for entity in local_shot.get('entities', []):
            lid = str(entity['local_id'])
            if lid in seen_local:
                raise ValueError(f'Local IDs must be globally unique: {lid}')
            seen_local.add(lid)
            if lid not in mapping or str(mapping[lid]) not in global_entities:
                raise ValueError(f'No valid global assignment for {lid}.')
            for tid in map(str, entity.get('track_ids', [])):
                if tid not in tracks:
                    raise ValueError(f'{lid} refers to unknown track {tid}.')
                if tid in owners:
                    raise ValueError(f'{shot_id}/{tid} is assigned to multiple local entities.')
                owners[tid] = lid
        rejected_ids = {_track_id(t) for t in local_shot.get('rejected_tracks', [])}
        if rejected_ids - tracks:
            raise ValueError(f'{shot_id} rejects unknown tracks.')
        if rejected_ids & owners.keys():
            raise ValueError(f'{shot_id} both rejects and assigns the same track.')
        missing = sorted(tracks - owners.keys() - rejected_ids)
        assigned += len(owners)
        rejected += len(rejected_ids)
        unreviewed += len(missing)
        unreviewed_rows.extend({'shot_id': shot_id, 'track_id': tid} for tid in missing)
    extra = set(mapping) - seen_local
    if extra:
        raise ValueError(f'Global mapping refers to missing local IDs: {sorted(extra)}')
    return dict(total_tracks=assigned+rejected+unreviewed, assigned_tracks=assigned,
                rejected_tracks=rejected, unreviewed_tracks=unreviewed,
                unreviewed=unreviewed_rows, complete_track_review=unreviewed == 0)


def suppress_duplicate_boxes(observations, threshold=.70, containment_threshold=.95):
    """Suppress geometric duplicates only within the SAME global ID/frame.

    Separate boxes sharing an ID survive and are reported for identity review.
    A carrier and carried creature with different IDs can never suppress one
    another here, regardless of their overlap.
    """
    groups = defaultdict(list)
    for row in observations:
        groups[(row['shot_id'], int(row['source_frame']), row['entity_id'])].append(row)
    kept, suppressed, conflicts = [], [], []
    def suppress(row, other, method):
        suppressed.append(dict(shot_id=row['shot_id'], frame=row['source_frame'], entity_id=row['entity_id'],
            removed=row['detection_id'], retained=other['detection_id'], method=method,
            iou=box_iou(row['bbox_pixels'], other['bbox_pixels']),
            intersection_over_smaller=_intersection_over_smaller(row['bbox_pixels'], other['bbox_pixels']),
            removed_observation=deepcopy(row)))
    for (sid, frame, eid), rows in groups.items():
        # A face/hand/body box nested in the reviewed same physical identity is
        # represented by the outer extent, even when the smaller box scored higher.
        outer_boxes = []
        for row in sorted(rows, key=lambda r: (_box_area(r['bbox_pixels']), _confidence(r)), reverse=True):
            enclosing = next((other for other in outer_boxes if
                _intersection_over_smaller(row['bbox_pixels'], other['bbox_pixels']) >= containment_threshold), None)
            if enclosing is None:
                outer_boxes.append(row)
            else:
                suppress(row, enclosing, 'same_id_containment')
        survivors = []
        for row in sorted(outer_boxes, key=_confidence, reverse=True):
            duplicate = next((other for other in survivors if box_iou(row['bbox_pixels'], other['bbox_pixels']) >= threshold), None)
            if duplicate is None:
                survivors.append(row)
            else:
                suppress(row, duplicate, 'same_id_high_iou')
        if len(survivors) > 1:
            conflicts.append(dict(shot_id=sid, frame=frame, entity_id=eid,
                track_ids=[r['tracklet_id'] for r in survivors],
                observation_ids=[r['detection_id'] for r in survivors],
                reason='Distinct simultaneous boxes share a global identity; retained for review.'))
        kept.extend(survivors)
    aliases = {row['removed']: row['retained'] for row in suppressed}
    survivors = {row['detection_id']: row for row in kept}
    for record in suppressed:
        final = record['retained']
        while final in aliases:
            final = aliases[final]
        record['final_retained'] = final
        source = record['removed_observation']
        provenance = {key: deepcopy(source[key]) for key in ('detection_id', 'tracklet_id', 'local_id',
            'bbox_pixels', 'observation_type', 'evidence', 'detection_source', 'confidence',
            'confidence_semantics', 'source_track_ids', 'detector_supported', 'mask_quality') if key in source}
        survivors[final].setdefault('merged_source_observations', []).append(provenance)
    return kept, dict(suppressed_count=len(suppressed), suppressed=suppressed,
                      conflict_count=len(conflicts), remaining_conflicts=conflicts,
                      policy='Same global identity only. Nested boxes keep outer extent; distinct global identities never suppress each other.')


def _timeline_and_samples(bundle, adaptive_input):
    rates = {_rate(s['fps']) for s in bundle['shots']}
    if len(rates) != 1:
        raise ValueError('Exact concatenation currently requires one common source FPS.')
    fps = rates.pop()
    dimensions = {(int(s['width']), int(s['height'])) for s in bundle['shots']}
    if len(dimensions) != 1:
        raise ValueError('Exact overlay concatenation requires equal shot dimensions.')
    sample_metadata = {}
    if adaptive_input is not None:
        adaptive_input = Path(adaptive_input)
        summary = read_json(adaptive_input/'summary.json')
        sample_metadata = {str(row['id']): read_json(adaptive_input/row['samples']) for row in summary['videos']}
    timeline_shots, samples, total = [], [], 0
    for shot in bundle['shots']:
        sid, count = str(shot['shot_id']), int(shot['frame_count'])
        original_start = int(shot.get('original_start_frame', total))
        original_fps = _rate(shot.get('original_fps', shot['fps']))
        original_start_sec = float(original_start/original_fps)
        if count <= 0:
            raise ValueError(f'{sid} has no frames.')
        meta = sample_metadata.get(sid)
        if meta is None and adaptive_input is not None:
            raise ValueError(f'Adaptive metadata is missing shot {sid}.')
        if meta is not None:
            if int(meta['source_frames']) != count or _rate(meta['source_fps']) != fps:
                raise ValueError(f'Adaptive timeline disagrees with tracking source: {sid}')
            if int(meta.get('original_start_frame', original_start)) != original_start:
                raise ValueError(f'Original source offset disagrees: {sid}')
        picks = meta['samples'] if meta else [dict(source_frame=i, hold_until_frame=i+1) for i in range(count)]
        expected_start = 0
        for sample in picks:
            start, end = int(sample['source_frame']), int(sample['hold_until_frame'])
            if start != expected_start or not start < end <= count:
                raise ValueError(f'Gap/overlap in adaptive hold intervals: {sid}')
            expected_start = end
            key = f'{sid}_f{start:08d}'
            samples.append(dict(sample, frame_key=key, shot_id=sid, source_frame=start,
                source_time_sec=float(start/fps),
                original_frame=int(sample.get('original_frame', original_start+round(start/fps*original_fps))),
                original_time_sec=float(sample.get('original_time_sec', original_start_sec+float(start/fps))), full_frame=total+start,
                full_time_sec=float((total+start)/fps), hold_frames=end-start,
                image=f'frames/{key}.jpg', original_video=shot.get('original_video', shot['source'])))
        if expected_start != count:
            raise ValueError(f'Adaptive samples do not cover whole shot: {sid}')
        timeline_shots.append(dict(id=sid, source=str(shot['source']), full_start_frame=total,
            full_start_sec=float(total/fps), source_frames=count, duration_sec=float(count/fps),
            original_start_frame=original_start, original_start_sec=original_start_sec, original_fps=str(original_fps),
            original_video=shot.get('original_video', shot['source'])))
        total += count
    return dict(fps=str(fps), size=list(dimensions.pop()), total_frames=total,
                duration_sec=float(total/fps), shots=timeline_shots), samples


def _copy_representatives(entities, observations, bundle, output):
    from PIL import Image, ImageDraw, ImageOps
    by_entity = defaultdict(list)
    for obs in observations:
        by_entity[obs['entity_id']].append(obs)
    tiles = []
    for entity in entities:
        eid = entity['entity_id']
        rows = by_entity[eid]
        ranked = sorted(rows, key=lambda o: math.sqrt(max(0., (o['bbox_pixels'][2]-o['bbox_pixels'][0]) *
                        (o['bbox_pixels'][3]-o['bbox_pixels'][1]))) * _confidence(o), reverse=True)
        chosen, chosen_shots = [], set()
        for row in ranked:
            if row['shot_id'] not in chosen_shots and _asset_path(row.get('crop'), bundle, output):
                chosen.append(row); chosen_shots.add(row['shot_id'])
        for row in ranked:
            if len(chosen) >= 6:
                break
            if row not in chosen and _asset_path(row.get('crop'), bundle, output) and all(
                row['shot_id'] != old['shot_id'] or abs(row['source_frame']-old['source_frame']) >= 8 for old in chosen):
                chosen.append(row)
        chosen = chosen[:6]
        folder = output/'entities'/eid
        folder.mkdir(parents=True, exist_ok=True)
        gallery = []
        for i, row in enumerate(chosen):
            target = folder/f'view_{i+1:02d}.jpg'
            with Image.open(_asset_path(row['crop'], bundle, output)) as crop:
                crop.convert('RGB').save(target, quality=94)
            gallery.append(dict(path=target.relative_to(output).as_posix(), shot_id=row['shot_id'],
                time_sec=row['full_time_sec'], original_time_sec=row['original_time_sec'],
                source_frame=row['source_frame'], observation_id=row['detection_id']))
        if gallery:
            shutil.copyfile(output/gallery[0]['path'], folder/'representative.jpg')
        else:
            Image.new('RGB', (320, 200), (40, 44, 53)).save(folder/'representative.jpg')
        entity.update(representative=f'entities/{eid}/representative.jpg', gallery=gallery,
                      representative_missing=not bool(gallery))
        tile = Image.new('RGB', (320, 275), (24, 28, 36))
        crop = ImageOps.contain(Image.open(folder/'representative.jpg').convert('RGB'), (304, 210))
        tile.paste(crop, ((320-crop.width)//2, 8+(210-crop.height)//2))
        draw = ImageDraw.Draw(tile)
        draw.text((10, 225), f'{eid} · {entity["kind"]}', fill=entity_color(eid), font=_font(17))
        draw.text((10, 250), entity['label'][:23], fill='white', font=_font(16))
        tiles.append(tile)
        write_json(folder/'entity.json', entity)
    gallery = Image.new('RGB', (1280, 275*max(1, math.ceil(len(tiles)/4))), (24, 28, 36))
    for i, tile in enumerate(tiles):
        gallery.paste(tile, ((i % 4)*320, (i//4)*275))
    gallery.save(output/'entity_gallery.jpg', quality=93)


def _render_videos(bundle, timeline, samples, observations, labels, output, original=True):
    import av
    import numpy as np
    from PIL import ImageDraw
    sys.path.insert(0, str(ROOT.parent/'03_adaptive_sampling'))
    from media import VideoWriter
    fps = Fraction(timeline['fps'])
    width, height = timeline['size']
    by_frame = defaultdict(list)
    for row in observations:
        by_frame[(row['shot_id'], row['source_frame'])].append(row)
    sampled = {(s['shot_id'], s['source_frame']): s for s in samples}
    holds = {s['full_frame']: s['hold_frames'] for s in samples}
    adaptive_path, original_path = output/'.adaptive_entities_full.mp4', output/'.original_entities_full.mp4'
    adaptive_writer = VideoWriter(adaptive_path, width, height, fps, holds, crf=20)
    original_writer = VideoWriter(original_path, width, height, fps, crf=20) if original else None
    font, header_font = _font(max(17, width//75)), _font(max(16, width//95))
    seen = 0
    try:
        for shot, time_row in zip(bundle['shots'], timeline['shots']):
            sid, decoded = shot['shot_id'], 0
            with av.open(str(shot['source'])) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                for index, frame in enumerate(container.decode(stream)):
                    if index >= int(shot['frame_count']):
                        raise ValueError(f'Source has extra frames: {sid}')
                    if (frame.width, frame.height) != (width, height):
                        raise ValueError(f'Source size changed: {sid}')
                    image = frame.to_image().convert('RGB')
                    sample = sampled.get((sid, index))
                    if sample:
                        target = output/sample['image']; target.parent.mkdir(parents=True, exist_ok=True)
                        image.save(target, quality=94)
                    # Create missing representative crops from actual source pixels.
                    for row in by_frame[(sid, index)]:
                        crop = _asset_path(row.get('crop'), bundle, output)
                        if crop is None:
                            relative = f'crops/{row["detection_id"]}.jpg'
                            target = output/relative; target.parent.mkdir(parents=True, exist_ok=True)
                            x1, y1, x2, y2 = row['bbox_pixels']
                            image.crop((max(0, int(x1)), max(0, int(y1)), min(width, int(math.ceil(x2))),
                                        min(height, int(math.ceil(y2))))).save(target, quality=92)
                            row['crop'] = relative
                    if original_writer is None and sample is None:
                        decoded += 1
                        continue
                    draw = ImageDraw.Draw(image)
                    for row in by_frame[(sid, index)]:
                        eid = row['entity_id']; color = entity_color(eid)
                        x1, y1, x2, y2 = row['bbox_pixels']
                        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, width//450))
                        label = f'{eid} | {labels[eid]}'
                        tw = draw.textbbox((0, 0), label, font=font)[2]+10
                        tx, ty = max(0, min(int(x1), width-tw)), max(35, int(y1)-font.size-8)
                        draw.rectangle((tx, ty, tx+tw, ty+font.size+7), fill=(15, 18, 24))
                        draw.text((tx+4, ty+2), label, fill=color, font=font)
                    original_sec = time_row['original_start_sec']+float(index/fps)
                    draw.rectangle((0, 0, width, 34), fill=(15, 18, 24))
                    draw.text((10, 5), f'{sid}  |  original {original_sec:.3f}s  |  source frame {index}', font=header_font, fill='white')
                    array = np.asarray(image)[:, :, ::-1].copy()
                    full_frame = time_row['full_start_frame']+index
                    if original_writer:
                        original_writer.write(full_frame, array)
                    if sample:
                        adaptive_writer.write(full_frame, array); seen += 1
                    decoded += 1
            if decoded != int(shot['frame_count']):
                raise ValueError(f'Source decode count differs: {sid}: {decoded} != {shot["frame_count"]}')
    finally:
        adaptive_writer.close()
        if original_writer:
            original_writer.close()
    reports = []
    for pending, expected in [(adaptive_path, [s['full_frame'] for s in samples])] + (
            [(original_path, list(range(timeline['total_frames'])))] if original else []):
        with av.open(str(pending)) as container:
            stream = container.streams.video[0]
            duration = stream.duration*stream.time_base
            stamps = [frame.pts*frame.time_base for frame in container.decode(stream)]
        if stamps != [Fraction(i, 1)/fps for i in expected] or duration != Fraction(timeline['total_frames'], 1)/fps:
            raise RuntimeError(f'Output timing mismatch: {pending.name}')
        final = output/pending.name[1:]; pending.replace(final)
        reports.append(dict(video=final.name, frames=len(stamps), exact_timestamps=True,
                            duration_sec=float(duration), duration_error_sec=0))
    return dict(passed=True, sampled_frames=seen, source_frames=timeline['total_frames'],
                duration_sec=timeline['duration_sec'], videos=reports, audio=False,
                adaptive_boxes='Boxes at each selected SOURCE frame are held with that frame; no interpolation.')


def export_story(bundle, local, global_result, output, adaptive_input=None, *, render_videos=True, render_original=True):
    """Write catalogs, grounded story, presence tables and exact-time overlays.

    Does not write status.json: the orchestrator owns execution status and the
    final quality gate. Returns the paths and structural validation findings.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    coverage = validate_story_contract(bundle, local, global_result)
    timeline, samples = _timeline_and_samples(bundle, adaptive_input)
    time_shots = {s['id']: s for s in timeline['shots']}
    local_shots = {s['shot_id']: s for s in local['shots']}
    global_entities = {str(e['entity_id']): dict(e) for e in global_result['entities']}
    mapping = {str(k): str(v) for k, v in global_result['mapping'].items()}
    observations = []
    for shot in bundle['shots']:
        sid = shot['shot_id']; ts = time_shots[sid]
        fps = _rate(shot['fps']); width, height = shot['width'], shot['height']
        track_owner = {str(tid): entity for entity in local_shots[sid].get('entities', []) for tid in entity.get('track_ids', [])}
        for track in shot['tracks']:
            tid = str(track['track_id']); entity = track_owner.get(tid)
            if entity is None:
                continue
            lid = str(entity['local_id']); eid = mapping[lid]; global_entity = global_entities[eid]
            seen_frames = set()
            for observation in track.get('observations', []):
                frame = int(observation.get('frame', observation.get('source_frame', -1)))
                if not 0 <= frame < int(shot['frame_count']) or frame in seen_frames:
                    raise ValueError(f'Invalid/duplicate frame in {sid}/{tid}: {frame}')
                seen_frames.add(frame)
                box = list(map(float, observation['bbox']))
                if len(box) != 4 or not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f'Invalid box in {sid}/{tid}/{frame}')
                box = [max(0., min(float(bound), x)) for x, bound in zip(box, [width, height, width, height])]
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f'Box outside image in {sid}/{tid}/{frame}')
                key = f'{sid}_f{frame:08d}'
                observations.append(dict(detection_id=f'{key}_{tid}', frame_key=key, shot_id=sid,
                    source_frame=frame, source_time_sec=float(frame/fps), time_sec=float(frame/fps),
                    original_frame=int(observation.get('original_frame', ts['original_start_frame']+round(frame/fps*Fraction(ts['original_fps'])))),
                    original_time_sec=float(observation.get('original_time_sec', ts['original_start_sec']+float(frame/fps))),
                    full_frame=ts['full_start_frame']+frame, full_time_sec=float((ts['full_start_frame']+frame)/fps),
                    hold_frames=1, bbox_pixels=box, bbox=[x/bound*1000 for x, bound in zip(box, [width, height, width, height])],
                    bbox_format='xyxy normalized 0..1000', source_size=[width, height], entity_id=eid,
                    local_id=lid, tracklet_id=tid, label=global_entity.get('label', entity.get('label', eid)),
                    kind=global_entity.get('kind', entity.get('kind', 'unknown')),
                    appearance=global_entity.get('appearance', entity.get('appearance', '')),
                    detector_class=observation.get('category', 'unknown'),
                    **_observation_provenance(observation),
                    crop=observation.get('crop'), image=observation.get('image'),
                    identity_status=global_entity.get('identity_status', 'unresolved' if global_entity.get('uncertain') else 'estimated'),
                    identity_method='dense within-shot tracking, local review and global story association'))
    observations, duplicates = suppress_duplicate_boxes(observations)
    observations.sort(key=lambda r: (r['full_frame'], r['entity_id'], r['tracklet_id']))
    by_entity, by_frame = defaultdict(list), defaultdict(list)
    for row in observations:
        by_entity[row['entity_id']].append(row); by_frame[row['frame_key']].append(row)
    entities = []
    adaptive_frame_map = {s['frame_key']: s for s in samples}
    for eid, entity in global_entities.items():
        rows = by_entity[eid]
        sampled_rows = {r['frame_key']: r for r in rows if r['frame_key'] in adaptive_frame_map}
        display_intervals = []
        for key, row in sampled_rows.items():
            sample = adaptive_frame_map[key]
            hold_sec = float(sample['hold_frames']/Fraction(timeline['fps']))
            display_intervals.append(dict(shot_id=row['shot_id'], full_start_sec=sample['full_time_sec'],
                full_end_sec=sample['full_time_sec']+hold_sec, original_start_sec=sample['original_time_sec'],
                original_end_sec=sample['original_time_sec']+hold_sec,
                source_frame=row['source_frame'], hold_frames=sample['hold_frames']))
        entity.update(entity_id=eid, label=entity.get('label', eid), kind=entity.get('kind', 'unknown'),
            appearance=entity.get('appearance', ''), local_ids=sorted(k for k, v in mapping.items() if v == eid),
            color_rgb=list(entity_color(eid)), observations=len(rows), shots=sorted({r['shot_id'] for r in rows}),
            first_seen_sec=min((r['original_time_sec'] for r in rows), default=None),
            last_seen_sec=max((r['original_time_sec'] for r in rows), default=None),
            display_intervals=display_intervals, adaptive_observations=len(sampled_rows),
            adaptive_shots=sorted({r['shot_id'] for r in sampled_rows.values()}),
            dense_only_shots=sorted({r['shot_id'] for r in rows}-{r['shot_id'] for r in sampled_rows.values()}),
            interval_note='Display intervals describe held adaptive samples, not continuous source visibility.',
            identity_status=entity.get('identity_status', 'unresolved' if entity.get('uncertain') else 'estimated'))
        entities.append(entity)
    if render_videos:
        video_validation = _render_videos(bundle, timeline, samples, observations,
            {e['entity_id']: e['label'] for e in entities}, output, render_original)
    else:
        video_validation = dict(passed=None, skipped=True, reason='render_videos=False')
    _copy_representatives(entities, observations, bundle, output)
    catalog = dict(entities=entities, entity_count=len(entities), mode='story',
        identification='Original-frame tracking → per-shot Entity review → global linking → summary reconciliation',
        note='Identity estimates retain source observations and uncertainty; unobserved does not mean absent.')
    write_json(output/'entities.json', catalog)
    write_json(output/'global_human_entities.json', dict(entities=[e for e in entities if e['kind'] == 'person']))
    write_json(output/'global_other_entities.json', dict(entities=[e for e in entities if e['kind'] != 'person']))
    write_json(output/'timeline.json', timeline)
    write_json(output/'samples.json', samples)
    write_json(output/'observations.json', dict(bbox_format='xyxy normalized 0..1000', observations=observations))
    write_json(output/'frame_annotations.json', [dict(s, observations=by_frame[s['frame_key']]) for s in samples])
    write_json(output/'duplicate_observations.json', duplicates)
    write_json(output/'video_validation.json', video_validation)
    corrected = {s['shot_id']: s for s in global_result.get('shots', [])}
    shot_outputs, event_errors = [], []
    for shot in bundle['shots']:
        sid = shot['shot_id']; reviewed = local_shots[sid]; final = corrected.get(sid, reviewed); ts = time_shots[sid]
        entries = []
        for entity in reviewed.get('entities', []):
            eid = mapping[str(entity['local_id'])]; exported = next(e for e in entities if e['entity_id'] == eid)
            entries.append(dict(entity, entity_id=eid, representative=exported['representative'],
                global_label=exported['label'], kind=exported['kind'],
                observation_ids=[r['detection_id'] for r in by_entity[eid] if r['shot_id'] == sid and r['local_id'] == entity['local_id']]))
        events = []
        for event in final.get('events', []):
            row = dict(event)
            time_basis = row.get('time_basis', final.get('time_basis', 'full' if sid in corrected else 'shot'))
            row['time_basis'] = time_basis
            for key in ('subject', 'object'):
                value = row.get(key)
                if isinstance(value, str) and value in mapping:
                    row[key] = mapping[value]
                elif value and isinstance(value, str) and value not in global_entities:
                    event_errors.append(dict(shot_id=sid, field=key, value=value, reason='Unknown Entity reference in event'))
            for key in ('start_sec', 'end_sec'):
                if key in row:
                    moment = float(row[key])
                    shot_moment = moment-ts['full_start_sec'] if time_basis == 'full' else moment
                    if not -1e-6 <= shot_moment <= ts['duration_sec']+1e-6:
                        event_errors.append(dict(shot_id=sid, field=key, value=moment, reason='Event time outside shot'))
                    row['shot_'+key] = shot_moment
                    row['original_'+key] = ts['original_start_sec']+shot_moment
                    row['full_'+key] = ts['full_start_sec']+shot_moment
            events.append(row)
        shot_outputs.append(dict(shot_id=sid, full_start_sec=ts['full_start_sec'], duration_sec=ts['duration_sec'],
            original_start_sec=ts['original_start_sec'], entities=entries,
            human_entities=[e for e in entries if e['kind'] == 'person'],
            other_entities=[e for e in entries if e['kind'] != 'person'],
            initial_summary=reviewed.get('summary', ''), summary=final.get('summary', ''),
            summary_reconciled=sid in corrected, events=events,
            rejected_tracks=reviewed.get('rejected_tracks', []), uncertainty=final.get('uncertainty', [])))
    write_json(output/'shot_entities.json', shot_outputs)
    shot_ids = [s['shot_id'] for s in bundle['shots']]
    presence = [dict(entity_id=e['entity_id'], label=e['label'], **{sid: int(sid in e['shots']) for sid in shot_ids}) for e in entities]
    with (output/'entity_shot_presence.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['entity_id', 'label', *shot_ids]); writer.writeheader(); writer.writerows(presence)
    write_json(output/'entity_shot_presence.json', dict(shots=shot_ids, rows=presence,
        semantics={'1': 'At least one retained source-frame observation in this shot.', '0': 'Not observed; not proof of physical absence.'}))
    story = dict(schema_version=1, timeline=timeline, entities=entities, local_to_global=mapping, shots=shot_outputs,
        time_basis='Each event declares time_basis (shot or full); shot/full/original offsets are also exported explicitly.',
        uncertainties=global_result.get('uncertainties', []), identity_evidence=global_result.get('evidence', []))
    write_json(output/'story.json', story)
    lines = ['# Entity와 샷 요약', '', '## 전역 Entity', '']
    lines.extend(f'- **{e["entity_id"]} · {e["label"]}** ({e["kind"]}): {e["appearance"]}' for e in entities)
    for shot in shot_outputs:
        lines += ['', f'## {shot["shot_id"]} · {shot["original_start_sec"]:.3f}s', '', str(shot['summary']), '']
        for event in shot['events']:
            lines.append(f'- {event.get("subject", "?")} → {event.get("action", "")} → {event.get("object") or ""} ({event.get("certainty", "unspecified")})')
    (output/'story.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    validation = dict(coverage=coverage, duplicate_boxes_suppressed=duplicates['suppressed_count'],
        simultaneous_identity_conflicts=duplicates['conflict_count'], event_reference_errors=event_errors,
        observation_types=dict(Counter(row['observation_type'] for row in observations)),
        observation_sources=dict(Counter(row.get('detection_source', 'unspecified') for row in observations)),
        empty_summaries=[s['shot_id'] for s in shot_outputs if not s['summary']],
        unreconciled_shots=[s['shot_id'] for s in shot_outputs if not s['summary_reconciled']],
        entities_without_observations=[e['entity_id'] for e in entities if not e['observations']],
        missing_representatives=[e['entity_id'] for e in entities if e['representative_missing']],
        video=video_validation)
    validation['structurally_complete'] = (coverage['complete_track_review'] and not event_errors and
        not validation['empty_summaries'] and not validation['entities_without_observations'])
    validation['identity_consistent'] = not duplicates['conflict_count']
    write_json(output/'story_validation.json', validation)
    return dict(mode='story', entity_count=len(entities), observations=len(observations), sampled_frames=len(samples),
        duration_sec=timeline['duration_sec'], video='adaptive_entities_full.mp4' if render_videos else None,
        original_video='original_entities_full.mp4' if render_videos and render_original else None,
        catalog='entities.json', gallery='entity_gallery.jpg', presence='entity_shot_presence.csv',
        shot_entities='shot_entities.json', story='story.json', validation=validation)
