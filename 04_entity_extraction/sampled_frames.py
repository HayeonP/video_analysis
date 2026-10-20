"""Selected-image-only media access and time-aware sparse association.

No original video is decoded in this module. Missing selected JPEGs fail closed.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace


def discover_sampled_shots(source):
    from PIL import Image
    root = Path(source).resolve()
    summary = json.loads((root / 'summary.json').read_text())
    shots, seen, full_start = [], set(), 0.
    for item in summary['videos']:
        sid = item['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid in seen:
            raise ValueError('Unsafe or duplicate shot ID')
        seen.add(sid)
        metadata = (root / item['samples']).resolve()
        if not metadata.is_relative_to(root):raise ValueError('Sample metadata escapes input')
        meta = json.loads(metadata.read_text())
        fps = Fraction(meta['source_fps']); count = int(meta['source_frames'])
        original_fps = Fraction(meta.get('original_fps', meta['source_fps']))
        if fps <= 0 or count < 1 or original_fps != fps:raise ValueError('Invalid source timing')
        samples = meta['samples']; indices = [int(s['source_frame']) for s in samples]
        if not indices or indices != sorted(set(indices)) or indices[0] != 0 or indices[-1] != count-1:
            raise ValueError('Samples must be ordered, unique, and retain first/last source frames')
        paths, hashes, size = {}, {}, None
        for n, s in enumerate(samples):
            frame = indices[n]
            hold = indices[n+1] if n+1 < len(indices) else count
            if s.get('hold_until_frame') != hold:raise ValueError('Sample hold intervals must cover the source exactly')
            if abs(float(s.get('source_time_sec', frame/fps))-float(frame/fps)) > 1e-5:
                raise ValueError('Sample timestamp differs from source frame')
            original_frame=int(meta.get('original_start_frame',0))+frame
            if int(s.get('original_frame',original_frame)) != original_frame:
                raise ValueError('Sample original frame disagrees with shot offset')
            if abs(float(s.get('original_time_sec',original_frame/fps))-float(original_frame/fps)) > 1e-5:
                raise ValueError('Sample original timestamp disagrees with shot offset')
            p = (metadata.parent / s['image']).resolve()
            if not p.is_relative_to(root) or not p.is_file():raise ValueError(f'Missing/unsafe selected image: {p}')
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if size is None:size = im.size
                if im.size != size:raise ValueError('Sample dimensions changed inside shot')
            paths[str(frame)] = str(p)
            hashes[str(frame)] = hashlib.sha256(p.read_bytes()).hexdigest()
        signature = hashlib.sha256(json.dumps(dict(meta=meta, images=hashes), sort_keys=True).encode()).hexdigest()
        shots.append(dict(shot_id=sid, source=meta.get('source',''), metadata=str(metadata),
            analysis_frames='sampled', selected_images=paths, selected_image_hashes=hashes,
            source_sha256=signature, fps=float(fps), frame_count=count,
            width=size[0], height=size[1], original_start_frame=int(meta.get('original_start_frame',0)),
            original_fps=float(original_fps), original_video=meta.get('original_video',meta.get('source','')),
            adaptive_frames=indices, adaptive_samples=samples, full_start_sec=full_start,
            duration_sec=float(count/fps), source_metadata={'sampled_input_sha256':signature}))
        full_start += float(count/fps)
    if not shots:raise ValueError('No adaptive shots found')
    return shots


def iter_shot_images(shot, wanted=None):
    """Yield (original shot frame number, RGB PIL image), never invent frames."""
    if shot.get('analysis_frames') == 'sampled':
        from PIL import Image
        allowed = {int(k) for k in shot['selected_images']}
        requested = allowed if wanted is None else set(wanted)
        if not requested <= allowed:raise ValueError(f'Unselected evidence requested: {sorted(requested-allowed)}')
        for i in sorted(requested):
            with Image.open(shot['selected_images'][str(i)]) as image:
                yield i, image.convert('RGB')
    else:
        import av
        with av.open(shot['source']) as container:
            stream = container.streams.video[0];stream.codec_context.thread_count = 2
            for i, frame in enumerate(container.decode(stream)):
                if wanted is None or i in wanted:yield i, frame.to_image().convert('RGB')


def validate_sampled_bundle(bundle, source):
    current = discover_sampled_shots(source)
    old = bundle['shots']
    if [(s['shot_id'],s['source_sha256']) for s in old] != [(s['shot_id'],s['source_sha256']) for s in current]:
        raise ValueError('Selected frames or sample metadata changed; use a fresh output')
    # Relocation is harmless; refresh paths after binding contents and timing.
    for a,b in zip(old,current):a['selected_images']=b['selected_images']
    assert_sampled_evidence(bundle)


def assert_sampled_evidence(bundle):
    if bundle.get('analysis_frames') != 'sampled':return
    for shot in bundle['shots']:
        allowed=set(shot['adaptive_frames'])
        rows=list(shot.get('scene_frames',[]))+list(shot.get('adaptive_scene_frames',[]))
        for track in shot.get('tracks',[]):rows += track['observations'] + track.get('representatives',[])
        if any(r['frame'] not in allowed for r in rows):raise ValueError('Off-sample evidence entered the pipeline')


class SparseTracker:
    """Conservative one-to-one association; expiry uses source time, not sample count.

    No dense Kalman step or synthetic detections. A broken link may be joined
    later by the local/global visual review; this is not an identity decision.
    """
    def __init__(self, fps, config):
        self.fps,self.config=fps,config
        self.active={};self.next_id=1

    def update(self, rows, frame):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        from shot_tracking import box_iou, tracking_confidence
        self.active={k:v for k,v in self.active.items() if (frame-v['frame'])/self.fps <= self.config.track_buffer_sec+1/self.fps+1e-6}
        ids=list(self.active);cost=np.full((len(ids),len(rows)),1e6)
        for i,k in enumerate(ids):
            old=self.active[k]
            for j,row in enumerate(rows):
                overlap=box_iou(old['row']['bbox'],row['bbox'])
                a,b=old['row'].get('feature'),row.get('feature')
                similarity=float(np.dot(a,b)/(max(1e-9,np.linalg.norm(a)*np.linalg.norm(b)))) if a and b else 0.
                if overlap < .3 and similarity < .75:continue
                elapsed=(frame-old['frame'])/self.fps
                score=.65*max(0.,similarity)+.35*overlap-.05*elapsed/max(self.config.track_buffer_sec,1e-6)
                if score>=.45 or (not (a and b) and overlap>=.5):cost[i,j]=1-score
        matches={}
        if cost.size:
            ii,jj=linear_sum_assignment(cost)
            matches={int(j):ids[int(i)] for i,j in zip(ii,jj) if cost[i,j]<1e5}
        visible=[]
        for j,row in enumerate(rows):
            tid=matches.get(j)
            if tid is None:
                if tracking_confidence(row,self.config)<self.config.new_track_confidence:continue
                tid=self.next_id;self.next_id+=1
            confirmed=tid in self.active
            self.active[tid]=dict(row=row,frame=frame)
            visible.append(SimpleNamespace(track_id=tid,idx=j,is_activated=confirmed))
        return visible
