"""Optional, bounded SAM 2.1 rescue after local physical-entity review.

This module does not decide identities. It refines the entities supplied by the
local reviewer, preserving every source track in an audit file. Independent
forward/backward mask propagation is checked against detector evidence before
creating a visible observation. Missing or rejected masks never become boxes.

Public API returns ``(refined_bundle, refined_local, summary)``. The model is
loaded lazily, so geometry/cache tests do not require a GPU or a checkpoint.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import time

import numpy as np

LOG = logging.getLogger('entities.mask_refinement')
VERSION = 4
PINNED_REVISIONS = {'facebook/sam2.1-hiera-small': 'ee5bba1d82bb8749febdf90f45e84b687142ba03'}


@dataclass(frozen=True)
class MaskRefinementConfig:
    prompts_per_entity: int = 4
    minimum_pixels: int = 16
    minimum_stability: float = .70
    agreement_iou: float = .30
    detector_iou: float = .10
    maximum_area_multiplier: float = 4.0
    support_padding_sec: float = .75
    one_direction_near_detection_sec: float = .15
    representatives: int = 8
    save_masks: bool = True
    save_masked_crops: bool = True
    reject_cross_identity_collisions: bool = True

    def __post_init__(self):
        if self.prompts_per_entity < 1 or self.minimum_pixels < 1 or self.representatives < 1:
            raise ValueError('Positive prompt, pixel and representative counts required')
        if any(not 0 <= v <= 1 for v in (self.minimum_stability, self.agreement_iou, self.detector_iou)):
            raise ValueError('Mask thresholds must be in [0,1]')
        if self.maximum_area_multiplier <= 0 or self.support_padding_sec < 0:
            raise ValueError('Invalid support or area limit')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def _iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / max(1e-9, _area(a) + _area(b) - intersection)


def mask_bbox(mask):
    """Return an exclusive xyxy pixel box, or None for genuine absence."""
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None


def _observation_score(row):
    return float(row.get('confidence', 0)) * math.sqrt(max(1., _area(row['bbox'])))


def entity_observations(shot, entity):
    wanted = set(entity['track_ids'])
    rows = []
    segments = entity.get('segments') or []
    for track in shot['tracks']:
        if track['track_id'] not in wanted:
            continue
        for original in track['observations']:
            applicable = [s for s in segments if s.get('track_id') == track['track_id']]
            if applicable and not any(s['first_frame'] <= original['frame'] <= s['last_frame'] for s in applicable):
                continue
            row = deepcopy(original)
            row['source_track_id'] = track['track_id']
            rows.append(row)
    return sorted(rows, key=lambda row: (int(row['frame']), -_observation_score(row)))


def select_prompts(rows, limit=4):
    """Temporal diversity among actual observed boxes, never guessed boxes."""
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[int(row['frame'])].append(row)
    candidates = [max(group, key=_observation_score) for _, group in sorted(by_frame.items())]
    if not candidates:
        return []
    best = max(candidates, key=_observation_score)
    selected = [best]
    span = max(1, candidates[-1]['frame'] - candidates[0]['frame'])
    peak = max(1., _observation_score(best))
    while len(selected) < min(limit, len(candidates)):
        remaining = [r for r in candidates if r['frame'] not in {x['frame'] for x in selected}]
        selected.append(max(remaining, key=lambda r: min(abs(r['frame']-s['frame'])/span for s in selected)
                            + .1*_observation_score(r)/peak))
    return sorted(selected, key=lambda row: row['frame'])


def assess_mask(logits, spec, frame, config):
    """Use stability/geometry gates; scores are explicitly uncalibrated."""
    logits = np.asarray(logits, dtype=np.float32).squeeze()
    if logits.ndim != 2 or not np.isfinite(logits).all() or (spec.get('shape') and logits.shape != spec['shape']):
        return None, 'invalid_mask'
    mask = logits > 0
    pixels = int(mask.sum())
    if pixels < config.minimum_pixels:
        return None, 'empty_or_tiny_mask'
    stability = float((logits > 1.).sum() / max(1, (logits > -1.).sum()))
    if stability < config.minimum_stability:
        return None, 'unstable_mask'
    if not spec['support'][0] <= frame <= spec['support'][1]:
        return None, 'outside_detector_supported_interval'
    if spec.get('edit_segments') is not None and not any(a<=frame<b for a,b in spec['edit_segments']):
        return None, 'across_source_edit_without_observation'
    box = mask_bbox(mask)
    if _area(box) > config.maximum_area_multiplier * spec['reference_area']:
        return None, 'area_growth_drift'
    observed = spec['by_frame'].get(frame, [])
    if observed and max(_iou(box, row['bbox']) for row in observed) < config.detector_iou:
        return None, 'detector_disagreement'
    if frame in spec['prompt_frames']:
        anchor = next(row for row in spec['prompts'] if row['frame'] == frame)
        x1, y1, x2, y2 = [int(round(v)) for v in anchor['bbox']]
        inside = int(mask[max(0,y1):max(0,y2), max(0,x1):max(0,x2)].sum()) / pixels
        if inside < .5:
            return None, 'prompt_spill'
    return dict(mask=mask, bbox=box, pixels=pixels, stability=stability), None


class TransformersSam2Backend:
    """Installed Transformers 4.57+ video API, with independent direction states."""
    def __init__(self, model_path, device, revision=None):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        self.device = str(device)
        self.dtype = torch.bfloat16 if self.device.startswith('cuda') else torch.float32
        identity = checkpoint_identity(model_path, revision)
        kwargs = {'revision': identity['revision']} if not Path(model_path).expanduser().is_dir() else {}
        self.processor = Sam2VideoProcessor.from_pretrained(str(model_path), **kwargs)
        self.model = Sam2VideoModel.from_pretrained(str(model_path), dtype=self.dtype, **kwargs).to(device).eval()
        self.revision = getattr(self.model.config, '_commit_hash', None) or identity['revision']

    def predict(self, frames, shot, specs, reverse=False):
        import torch
        # Pixel preprocessing is independent of prompts. Reuse those immutable
        # CPU tensors across object/direction sessions without sharing memory,
        # masks, prompts or any identity-bearing inference state.
        video_key=(id(frames),shot.get('source'),shot.get('shot_id'),len(frames),shot['width'],shot['height'])
        if getattr(self,'_prepared_video_key',None)!=video_key:
            prepared=self.processor.init_video_session(video=frames,inference_device=self.device,
                inference_state_device='cpu',video_storage_device='cpu',processing_device='cpu',dtype=self.dtype)
            self._prepared_video_frames=prepared.processed_frames
            self._prepared_video_key=video_key
            del prepared
        # Transformers 4.57 evaluates every registered object on each call.
        # Objects whose first prompt occurs later can lack conditioning memory,
        # so use independent sessions rather than inventing an earlier prompt.
        # This also avoids competition between a carrier and a carried subject.
        for local_id in sorted(specs):
            session = self.processor.init_video_session(video=None, inference_device=self.device,
                        inference_state_device='cpu', video_storage_device='cpu', processing_device='cpu', dtype=self.dtype)
            session.processed_frames=dict(self._prepared_video_frames)
            session.video_height,session.video_width=shot['height'],shot['width']
            prompts=specs[local_id]['prompts']
            try:
                # SAM's video memory is stored in bfloat16 even with a CPU
                # float32 backbone; autocast keeps memory attention compatible.
                cpu_cast=torch.autocast('cpu',dtype=torch.bfloat16) if self.device.startswith('cpu') else nullcontext()
                with torch.inference_mode(),cpu_cast:
                    for row in prompts:
                        self.processor.add_inputs_to_inference_session(session,frame_idx=row['frame'],
                            obj_ids=1,input_boxes=[[row['bbox']]])
                    for frame in sorted({r['frame'] for r in prompts},reverse=reverse):
                        self.model(inference_session=session,frame_idx=frame,reverse=reverse)
                    start=len(frames)-1 if reverse else 0
                    for result in self.model.propagate_in_video_iterator(session,start_frame_idx=start,reverse=reverse):
                        masks=self.processor.post_process_masks([result.pred_masks],
                            original_sizes=[[shot['height'],shot['width']]],binarize=False,
                            apply_non_overlapping_constraints=False)[0]
                        yield int(result.frame_idx),{local_id:masks[0].float().cpu().numpy().squeeze()}
            finally:
                del session
                gc.collect()


def _decode(shot):
    import av
    with av.open(shot['source']) as container:
        frames = [frame.to_image().convert('RGB') for frame in container.decode(video=0)]
    if len(frames) != shot['frame_count']:
        raise ValueError(f"Original frame count changed: {shot['shot_id']}")
    return frames


def _checkpoint_files(path):
    return [p for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in {'.json','.safetensors','.bin'}]


def _legacy_local_identity(path, revision=None):
    files=[dict(name=str(p.relative_to(path)),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
           for p in _checkpoint_files(path)]
    inferred=path.name if re.fullmatch(r'[0-9a-f]{40}',path.name) else None
    return dict(kind='local',name=str(path.resolve()),revision=revision or inferred,files=files)


def _verified_snapshot(path):
    """Verify HF content-addressed links; a directory name alone proves nothing."""
    path=path.resolve()
    if (path.parent.name!='snapshots' or not re.fullmatch(r'[0-9a-f]{40}',path.name)
            or not path.parent.parent.name.startswith('models--')):
        return None
    repository=path.parent.parent
    manifest=[]
    for file in _checkpoint_files(path):
        resolved=file.resolve()
        if not file.is_symlink() or resolved.parent!=repository/'blobs':return None
        address=resolved.name
        if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',address):return None
        sha256=hashlib.sha256();blob=hashlib.sha1(f'blob {file.stat().st_size}\0'.encode())
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                sha256.update(chunk);blob.update(chunk)
        if address!=(sha256.hexdigest() if len(address)==64 else blob.hexdigest()):return None
        manifest.append(dict(name=str(file.relative_to(path)),sha256=sha256.hexdigest()))
    names={entry['name'] for entry in manifest}
    if 'config.json' not in names or not any(name.endswith(('.safetensors','.bin')) for name in names):return None
    return dict(kind='huggingface',name=repository.name[len('models--'):].replace('--','/'),
                revision=path.name,verified_files=manifest)


def _cached_snapshot(model_path,revision):
    roots=[]
    if os.environ.get('HF_HUB_CACHE'):roots.append(Path(os.environ['HF_HUB_CACHE']))
    if os.environ.get('HF_HOME'):roots.append(Path(os.environ['HF_HOME'])/'hub')
    roots.extend([Path(__file__).resolve().parent/'.cache/huggingface/hub',Path.home()/'.cache/huggingface/hub'])
    for root in roots:
        path=root/('models--'+str(model_path).replace('/','--'))/'snapshots'/revision
        if path.is_dir():return path
    return None


def checkpoint_identity(model_path, revision=None):
    """Pin Hub downloads and equate local snapshots only after content verification."""
    path=Path(model_path).expanduser()
    if path.is_dir():
        verified=_verified_snapshot(path)
        if verified and (revision is None or verified['revision']==revision):return verified
        identity=_legacy_local_identity(path,revision)
        # Content matters even when a weight edit preserves file size and time.
        for entry,file in zip(identity['files'],_checkpoint_files(path)):
            with file.open('rb') as stream:entry['sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
        return identity
    pinned=revision or PINNED_REVISIONS.get(str(model_path))
    if not pinned or not re.fullmatch(r'[0-9a-f]{40}',pinned):
        raise ValueError('A Hub SAM model requires an explicit 40-character model_revision, or a local checkpoint directory.')
    snapshot=_cached_snapshot(model_path,pinned)
    if snapshot:
        verified=_verified_snapshot(snapshot)
        if verified:return verified
        # A corrupted or copied snapshot must never inherit a trusted Hub key.
        return checkpoint_identity(snapshot,pinned)
    return dict(kind='huggingface',name=str(model_path),revision=pinned)


def _compatible_checkpoint_keys(payload,model_path,revision):
    keys={_digest(payload)}
    identity=payload['model']
    if not identity.get('verified_files'):return keys
    path=Path(model_path).expanduser()
    if not path.is_dir():path=_cached_snapshot(model_path,identity['revision'])
    if path is None or _verified_snapshot(path)!=identity:return keys
    # Accept the old metadata key only for this verified snapshot and unchanged
    # input/configuration. Preserve its result fingerprint for downstream caches.
    old_models=[_legacy_local_identity(path,revision),
                dict(kind='huggingface',name=identity['name'],revision=identity['revision'])]
    return keys|{_digest(dict(payload,model=old)) for old in old_models}


def reject_mask_collisions(accepted, threshold=.90):
    """Identical foreground for two identities is drift, not an entity merge.

    Nested bounding boxes alone never trigger this rule; actual mask pixels
    must substantially coincide. Source detector observations remain intact.
    """
    by_frame=defaultdict(list)
    for lid,rows in accepted.items():
        for frame,(data,_) in rows.items():by_frame[frame].append((lid,data['mask']))
    rejected=set()
    for frame,rows in by_frame.items():
        for index,(left,a) in enumerate(rows):
            for right,b in rows[index+1:]:
                overlap=float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))
                if overlap>=threshold:
                    rejected.add((left,frame));rejected.add((right,frame))
    for lid,frame in rejected:accepted[lid].pop(frame,None)
    return len(rejected)


def _specs(shot, local_shot, config):
    specs = {}
    for entity in local_shot.get('entities', []):
        rows = entity_observations(shot, entity)
        if not rows:
            continue
        by_frame = defaultdict(list)
        for row in rows:
            by_frame[int(row['frame'])].append(row)
        prompts = select_prompts(rows, config.prompts_per_entity)
        pad = round(config.support_padding_sec * shot['fps'])
        edges=[0]+shot.get('internal_boundaries',{}).get('cut_frames',[])+[shot['frame_count']]
        edit_segments=[(a,b) for a,b in zip(edges,edges[1:]) if any(a<=f<b for f in by_frame)]
        specs[entity['local_id']] = dict(entity=entity, rows=rows, by_frame=by_frame, prompts=prompts,
            edit_segments=edit_segments,
            shape=(shot['height'],shot['width']),
            prompt_frames={int(p['frame']) for p in prompts},
            reference_area=float(np.median([_area(p['bbox']) for p in prompts])),
            support=(max(0, rows[0]['frame']-pad), min(shot['frame_count']-1, rows[-1]['frame']+pad)))
    return specs


def _mask_row(mask_data, shot, spec, frame, agreement, mask_path):
    original = spec['by_frame'].get(frame, [])
    supporting = max(original, key=lambda row: _iou(row['bbox'], mask_data['bbox'])) if original else None
    nearest = min(spec['rows'], key=lambda row: abs(row['frame']-frame))
    fps = shot['fps']
    result = dict(frame=frame, source_frame=frame, shot_id=shot['shot_id'], time_sec=frame/fps,
        full_time_sec=shot['full_start_sec']+frame/fps,
        original_time_sec=shot['original_start_frame']/shot['original_fps']+frame/fps,
        original_frame=shot['original_start_frame']+round(frame/fps*shot['original_fps']),
        bbox=mask_data['bbox'], confidence=round(mask_data['stability']*(agreement if agreement is not None else .75),6),
        confidence_semantics='uncalibrated mask stability times directional agreement; not detector confidence',
        category=spec['entity'].get('kind', 'unknown'), detection_source='sam2_mask', evidence='sam2_mask',
        tracker_confirmed=True, image=None, crop=None, mask=mask_path,
        source_track_ids=list(spec['entity']['track_ids']),
        detector_supported=bool(original), nearest_detector_frame=nearest['frame'],
        mask_quality=dict(stability=mask_data['stability'], forward_backward_iou=agreement,
                          pixels=mask_data['pixels'], calibrated=False))
    # A changed crop needs a fresh embedding. Never relabel an old detector
    # crop's embedding as the new SAM crop's visual feature.
    if supporting and supporting.get('feature') and list(supporting['bbox'])==list(mask_data['bbox']):
        result.update(feature=supporting['feature'],feature_provenance='identical_source_detection_crop')
    return result


def _build_track(shot, spec, accepted, output, frames, config):
    from PIL import Image
    from shot_tracking import select_representatives
    local_id = spec['entity']['local_id']
    tid = f"{shot['shot_id']}:M{_digest(local_id)[:10]}"
    rows = {}
    # Preserve actual detector evidence where SAM cannot corroborate a mask.
    # Every competing original box remains in source_tracks.json for review.
    for frame, observed in spec['by_frame'].items():
        fallback = deepcopy(max(observed, key=_observation_score))
        fallback.update(source_track_ids=list(spec['entity']['track_ids']),
                        mask_refinement_status='detector_fallback', original_box_candidates=len(observed))
        rows[frame] = fallback
    for frame, (mask_data, agreement) in sorted(accepted.items()):
        mask_path = None
        if config.save_masks:
            mask_path = f"mask_refinement/masks/{shot['shot_id']}/{tid.split(':')[-1]}_f{frame:08d}.png"
            destination = output/mask_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_data['mask'].astype(np.uint8)*255).save(destination)
        rows[frame] = _mask_row(mask_data, shot, spec, frame, agreement, mask_path)
    observations = [rows[frame] for frame in sorted(rows)]
    for row in observations:
        row['track_id'] = tid
    representatives = select_representatives(observations, config.representatives)
    for row in representatives:
        frame = row['frame']
        image_path = f"mask_refinement/frames/{shot['shot_id']}/f{frame:08d}.jpg"
        crop_path = f"mask_refinement/crops/{shot['shot_id']}/{tid.split(':')[-1]}_f{frame:08d}.jpg"
        (output/image_path).parent.mkdir(parents=True, exist_ok=True)
        (output/crop_path).parent.mkdir(parents=True, exist_ok=True)
        frames[frame].save(output/image_path, quality=94)
        x1,y1,x2,y2 = row['bbox']
        rect=(int(x1),int(y1),math.ceil(x2),math.ceil(y2))
        raw_crop=frames[frame].crop(rect)
        raw_crop.save(output/crop_path,quality=96)
        row['image'],row['crop'] = image_path,crop_path
        if config.save_masked_crops and row.get('evidence')=='sam2_mask' and frame in accepted:
            # Only accepted, observed SAM foreground may remove background.
            # Detector-only fallback boxes never acquire an invented mask.
            mask=accepted[frame][0]['mask'][rect[1]:rect[3],rect[0]:rect[2]]
            pixels=np.asarray(raw_crop).copy()
            if mask.shape!=pixels.shape[:2]:
                raise ValueError(f'Mask/crop shape mismatch: {shot["shot_id"]} frame {frame}')
            pixels[~mask]=127
            masked_path=f"mask_refinement/masked_crops/{shot['shot_id']}/{tid.split(':')[-1]}_f{frame:08d}.png"
            (output/masked_path).parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(pixels).save(output/masked_path)
            row['masked_crop']=masked_path
            row['masked_crop_background']=[127,127,127]
    first,last = observations[0]['frame'],observations[-1]['frame']
    gaps = [dict(start_frame=a['frame']+1,end_frame_exclusive=b['frame'],duration_sec=(b['frame']-a['frame']-1)/shot['fps'])
            for a,b in zip(observations,observations[1:]) if b['frame']>a['frame']+1]
    mask_count = sum(row.get('evidence')=='sam2_mask' for row in observations)
    return dict(track_id=tid, source_track_ids=list(spec['entity']['track_ids']),local_id=local_id,
        observations=observations,representatives=representatives,category=spec['entity'].get('kind','unknown'),
        first_frame=first,last_frame=last,observed_frames=len(observations),
        observed_duration_sec=len(observations)/shot['fps'],span_duration_sec=(last-first+1)/shot['fps'],
        observation_coverage=len(observations)/(last-first+1),
        mean_confidence=float(np.mean([row['confidence'] for row in observations])),occlusion_gaps=gaps,
        quality_flags=['mask_refinement_incomplete'] if mask_count<len(observations) else [],
        uncertain=bool(spec['entity'].get('uncertain')),classification_status='local_vlm_review',
        mask_frames=mask_count,detector_fallback_frames=len(observations)-mask_count)


def refine_local_tracks(bundle, local, output, device='cuda:0', model_path='facebook/sam2.1-hiera-small',
                        *, config=None, force=False, backend=None, model_revision=None):
    """Return new bundle/local/summary without mutating existing perception data."""
    config = config if isinstance(config,MaskRefinementConfig) else MaskRefinementConfig(**(config or {}))
    output = Path(output).expanduser().resolve()
    cache = output/'mask_refinement'
    identity=checkpoint_identity(model_path,model_revision)
    payload=dict(version=VERSION,bundle=bundle.get('fingerprint'),local=local,config=asdict(config),model=identity)
    key = _digest(payload)
    compatible_keys=_compatible_checkpoint_keys(payload,model_path,model_revision)
    if not force and all((cache/name).is_file() for name in ('bundle.json','local.json','summary.json')):
        summary = json.loads((cache/'summary.json').read_text())
        if summary.get('fingerprint') in compatible_keys:
            refined=json.loads((cache/'bundle.json').read_text());review=json.loads((cache/'local.json').read_text())
            assets=[row.get(field) for shot in refined['shots'] for track in shot['tracks']
                    for row in track.get('representatives',[]) for field in ('image','crop')]
            assets += [row['masked_crop'] for shot in refined['shots'] for track in shot['tracks']
                       for row in track.get('representatives',[]) if row.get('masked_crop')]
            if all(value and (Path(refined['asset_root'])/value).is_file() for value in assets):
                if summary['fingerprint']!=key:
                    LOG.info('SAM cache reused after checkpoint content-hash verification: %s',cache)
                return refined,review,summary
    started=time.perf_counter()
    refined,review=deepcopy(bundle),deepcopy(local)
    old_assets=Path(bundle.get('asset_root',output)).resolve()
    if old_assets!=output:
        # Old assets are still valid evidence; retain absolute references until
        # the exporter copies them into the selected result directory.
        for shot in refined['shots']:
            asset_rows=list(shot.get('scene_frames',[]))+list(shot.get('adaptive_scene_frames',[]))
            for track in shot['tracks']:
                asset_rows+=list(track.get('observations',[]))+list(track.get('representatives',[]))
            for row in asset_rows:
                for field in ('image','crop','masked_crop','mask'):
                    if row.get(field) and not Path(row[field]).is_absolute():row[field]=str(old_assets/row[field])
    local_by_shot={row['shot_id']:row for row in review['shots']}
    if set(local_by_shot)!={row['shot_id'] for row in bundle['shots']}:
        raise ValueError('Local review must cover all source shots before mask refinement')
    _write(cache/'source_tracks.json',dict(bundle_fingerprint=bundle.get('fingerprint'),
                                        asset_root=bundle.get('asset_root'),shots=bundle['shots']))
    records=[];owns_backend=backend is None
    try:
        for shot in refined['shots']:
            local_shot=local_by_shot[shot['shot_id']]
            shot_cache=cache/'shots'/f"{shot['shot_id']}.json"
            if not force and shot_cache.is_file():
                previous=json.loads(shot_cache.read_text())
                if previous.get('fingerprint') in compatible_keys:
                    cached_shot=previous['shot']
                    assets=[row.get(field) for track in cached_shot['tracks'] for row in track.get('representatives',[])
                            for field in ('image','crop')]
                    assets += [row['masked_crop'] for track in cached_shot['tracks']
                               for row in track.get('representatives',[]) if row.get('masked_crop')]
                    if all(value and (output/value).is_file() for value in assets):
                        shot.clear();shot.update(cached_shot);local_shot.clear();local_shot.update(previous['local'])
                        records.append(previous['summary']);LOG.info('[CACHE HIT] SAM2 shot %s',shot['shot_id']);continue
            specs=_specs(shot,local_shot,config)
            if not specs:
                records.append(dict(shot_id=shot['shot_id'],status='no_reviewed_entities'));continue
            frames=_decode(shot)
            if backend is None:
                backend=TransformersSam2Backend(model_path,device,revision=model_revision)
            LOG.info('[SAM2] %s: %d physical entities, %d frames, independent forward/backward passes',
                     shot['shot_id'],len(specs),len(frames))
            forward={};accepted=defaultdict(dict);rejections=Counter();directional_conflicts=0
            for frame,predictions in backend.predict(frames,shot,specs,reverse=False):
                for lid,logits in predictions.items():
                    data,reason=assess_mask(logits,specs[lid],frame,config)
                    if data:
                        forward[(lid,frame)]=(np.packbits(data.pop('mask')),data)
                    else:rejections['forward:'+reason]+=1
            for frame,predictions in backend.predict(frames,shot,specs,reverse=True):
                for lid,logits in predictions.items():
                    data,reason=assess_mask(logits,specs[lid],frame,config)
                    previous=forward.pop((lid,frame),None)
                    if previous:
                        packed,first=previous;first['mask']=np.unpackbits(packed,count=shot['height']*shot['width']).reshape(shot['height'],shot['width']).astype(bool)
                    else:first=None
                    if first and data:
                        agreement=float(np.logical_and(first['mask'],data['mask']).sum()/max(1,np.logical_or(first['mask'],data['mask']).sum()))
                        if agreement>=config.agreement_iou:
                            chosen=first if first['stability']>=data['stability'] else data
                            accepted[lid][frame]=(chosen,agreement)
                        else:directional_conflicts+=1;rejections['forward_backward_disagreement']+=1
                    elif first or data:
                        near=min(abs(row['frame']-frame) for row in specs[lid]['rows'])/shot['fps']
                        if near<=config.one_direction_near_detection_sec:
                            accepted[lid][frame]=(first or data,None)
                        else:rejections['one_direction_without_nearby_detection']+=1
                    if reason:rejections['backward:'+reason]+=1
            collisions=reject_mask_collisions(accepted) if config.reject_cross_identity_collisions else 0
            if collisions:rejections['cross_identity_mask_collision']+=collisions
            assigned={tid for spec in specs.values() for tid in spec['entity']['track_ids']}
            retained=[track for track in shot['tracks'] if track['track_id'] not in assigned]
            built=[]
            for lid,spec in specs.items():
                track=_build_track(shot,spec,accepted[lid],output,frames,config);built.append(track)
                entity=spec['entity'];entity['source_track_ids']=list(entity['track_ids']);entity['track_ids']=[track['track_id']]
                if entity.get('segments'):entity['source_segments']=entity.pop('segments')
            shot['tracks']=retained+built
            record=dict(shot_id=shot['shot_id'],status='refined',source_tracks=len(assigned),physical_tracks=len(built),
                mask_frames=sum(t['mask_frames'] for t in built),detector_fallback_frames=sum(t['detector_fallback_frames'] for t in built),
                propagated_only_frames=sum(not row.get('detector_supported',True) for t in built for row in t['observations']),
                directional_conflicts=directional_conflicts,cross_identity_mask_collisions=collisions,rejected_masks=dict(rejections),
                prompts={lid:[dict(frame=r['frame'],bbox=r['bbox'],source_track_id=r['source_track_id']) for r in s['prompts']] for lid,s in specs.items()})
            records.append(record);shot['mask_refinement']=record
            _write(shot_cache,dict(fingerprint=key,shot=shot,local=local_shot,summary=record))
            del frames,forward,accepted
    finally:
        if owns_backend:
            backend=None;gc.collect()
            if str(device).startswith('cuda'):
                import torch
                torch.cuda.empty_cache()
    refined.update(fingerprint=key,asset_root=str(output),pre_refinement_fingerprint=bundle.get('fingerprint'),
                   pre_refinement_stats=deepcopy(bundle.get('stats',{})),
                   pre_refinement_tracker=bundle.get('tracker'),
                   tracker='Local Entity consolidation + SAM2.1 bidirectional mask refinement over detector tracking',
                   visibility_policy='accepted SAM2 masks or actual source detections; rejected/empty masks remain gaps')
    refined.setdefault('stats',{}).update(
        tracks=sum(len(shot['tracks']) for shot in refined['shots']),
        singleton_tracks=sum(len(track['observations'])==1 for shot in refined['shots'] for track in shot['tracks']),
        observed_track_boxes=sum(len(track['observations']) for shot in refined['shots'] for track in shot['tracks']))
    summary=dict(version=VERSION,fingerprint=key,model=str(model_path),model_identity=identity,config=asdict(config),shots=records,
        elapsed_sec=time.perf_counter()-started,mask_frames=sum(r.get('mask_frames',0) for r in records),
        detector_fallback_frames=sum(r.get('detector_fallback_frames',0) for r in records),
        propagated_only_frames=sum(r.get('propagated_only_frames',0) for r in records),
        quality_semantics='Mask agreement is a geometric consistency check, not proof of physical identity.',
        warnings=['Detector fallbacks remain where mask evidence failed.'] if any(r.get('detector_fallback_frames') for r in records) else [])
    _write(cache/'bundle.json',refined);_write(cache/'local.json',review);_write(cache/'summary.json',summary)
    return refined,review,summary


def raw_track_masks(bundle,output,device='cuda:0',model_path='facebook/sam2.1-hiera-small',
                    *,config=None,force=False,backend=None,model_revision=None):
    """Return ``(evidence_bundle, summary)`` before semantic identity review.

    Each raw track is a separate hypothesis. Overlapping masks are preserved
    as evidence for the caller's graph, never treated as independent identities
    or merged here. Raw track IDs remain stable in the returned bundle. Masks
    are PNGs relative to ``evidence_bundle['asset_root']``.
    """
    root=Path(output).expanduser().resolve()/'raw_track_masks'
    values=asdict(config) if isinstance(config,MaskRefinementConfig) else dict(config or {})
    values['reject_cross_identity_collisions']=False
    # No model review is implied: these are deliberately unresolved hypotheses.
    local=dict(shots=[dict(shot_id=shot['shot_id'],entities=[dict(local_id=f"raw:{track['track_id']}",
        track_ids=[track['track_id']],kind='unknown',label='Raw track hypothesis',uncertain=True)
        for track in shot['tracks'] if track.get('observations')],rejected_tracks=[])
        for shot in bundle['shots']])
    refined,_,summary=refine_local_tracks(bundle,local,root,device,model_path,config=values,
                                        force=force,backend=backend,model_revision=model_revision)
    for shot in refined['shots']:
        for track in shot['tracks']:
            sources=track.get('source_track_ids') or [track['track_id']]
            if len(sources)!=1:raise ValueError('Raw mask hypotheses must retain one source track each')
            original_id=sources[0]
            track['mask_refinement_track_id']=track['track_id']
            track['track_id']=original_id
            track['classification_status']='unreviewed_raw_track_hypothesis'
            track['uncertain']=True
            for row in track['observations']+track.get('representatives',[]):row['track_id']=original_id
    refined['mask_evidence_mode']='raw_track_hypotheses; overlaps preserved; no identity decisions'
    summary=dict(summary,mode='raw_track_hypotheses',identity_decisions=False,
                 cross_track_mask_overlap_policy='preserved for downstream evidence comparison')
    _write(root/'bundle.json',refined);_write(root/'summary.json',summary)
    return refined,summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,type=Path);parser.add_argument('--local',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--model',default='facebook/sam2.1-hiera-small');parser.add_argument('--force',action='store_true')
    parser.add_argument('--model-revision',help='Pinned 40-character Hub revision for a non-default checkpoint')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    _,_,summary=refine_local_tracks(json.loads(args.bundle.read_text()),json.loads(args.local.read_text()),
                                  args.output,args.device,args.model,force=args.force,model_revision=args.model_revision)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
