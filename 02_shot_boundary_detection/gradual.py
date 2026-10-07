"""TransNetV2 inference with complete, non-overlapping source-frame coverage."""
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
from urllib.request import Request, urlopen

import cv2
import numpy as np

from video_utils import sha256, probe

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'models/transnetv2.pth'
WEIGHTS_SHA256 = 'eed5336d5d6a013c67f5863505a26e7e835053e64a9ce413d6b089ccba07bb53'
WEIGHTS_URL = ('https://huggingface.co/magnusdtd/TransNetV2/resolve/'
               '1c0e86a1a6333ff57c4458a44fe4b5b4a0faae2a/transnetv2-pytorch-weights.pth')


def ensure_weights():
    if not WEIGHTS.exists():
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        partial = WEIGHTS.with_suffix('.part')
        try:
            print('TransNetV2 가중치 다운로드 (약 30 MB)', flush=True)
            request = Request(WEIGHTS_URL, headers={'User-Agent': 'shot-boundary/2.0'})
            with urlopen(request, timeout=60) as source, partial.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            if sha256(partial) != WEIGHTS_SHA256:
                raise ValueError('TransNetV2 가중치 체크섬이 다릅니다.')
            partial.replace(WEIGHTS)
        finally:
            partial.unlink(missing_ok=True)
    if sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError('TransNetV2 가중치가 변경되었습니다.')
    return WEIGHTS


def runs(mask):
    """Convert a boolean mask to half-open spans."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def transition_regions(single, many, black, fps, *, threshold=0.5, min_shot_seconds=0.25,
                fade_bridge_seconds=0.75):
    """Identify scene interiors and transition regions, without exporting yet.

    Bridge gradual predictions to nearby black frames. Short stable fragments
    become part of a transition region; contiguous_spans retains all their frames.
    """
    single, many, black = np.asarray(single), np.asarray(many), np.asarray(black, dtype=bool)
    if len(single) != len(many) or len(single) != len(black):
        raise ValueError('프레임별 검출 결과 길이가 다릅니다.')
    transition = (single >= threshold) | (many >= threshold)
    excluded = transition | black
    dark_runs = runs(black)
    bridge = round(fade_bridge_seconds * fps)
    for start, end in runs(many >= threshold):
        if end - start < 3:
            continue
        for dark_start, dark_end in dark_runs:
            if 0 <= start - dark_end <= bridge:
                excluded[dark_end:start] = True
            if 0 <= dark_start - end <= bridge:
                excluded[end:dark_start] = True
    minimum = max(1, math.ceil(min_shot_seconds * fps))
    fragments = np.zeros(len(black), dtype=bool)
    for start, end in runs(~excluded):
        if end - start < minimum:
            fragments[start:end] = True
    excluded |= fragments
    omissions = []
    for start, end in runs(excluded):
        reasons = []
        if transition[start:end].any(): reasons.append('transition')
        if black[start:end].any(): reasons.append('black')
        if fragments[start:end].any(): reasons.append('short_fragment')
        omissions.append(dict(start_frame=start, end_frame_exclusive=end,
                              frame_count=end-start, reasons=reasons))
    return runs(~excluded), omissions


def contiguous_spans(single, many, black, fps, *, threshold=0.5, min_shot_seconds=0.25,
                     fade_bridge_seconds=0.75):
    """Split each gap between scene interiors once; never discard input frames.

    A one-frame hard-cut prediction belongs to the preceding scene. Wider fade
    regions are divided at their midpoint (rounded up to an integer frame).
    Initial and final transition regions remain in the first and last clips.
    """
    stable, regions = transition_regions(single, many, black, fps, threshold=threshold,
        min_shot_seconds=min_shot_seconds, fade_bridge_seconds=fade_bridge_seconds)
    total = len(single)
    if total == 0:
        return [], []
    cuts = [(left[1] + right[0] + 1) // 2 for left, right in zip(stable, stable[1:])]
    boundaries = [0, *cuts, total]
    spans = list(zip(boundaries, boundaries[1:]))
    for region in regions:
        region['cut_frame'] = next((cut for cut in cuts
            if region['start_frame'] <= cut <= region['end_frame_exclusive']), None)
    return spans, regions


def context_source(source):
    """Use verified full-source context for samples made by our downloader."""
    sidecar = source.with_suffix('.source.json')
    if not sidecar.is_file():
        return source
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    if meta.get('source_start_sec') != 0 or meta.get('sha256') != sha256(source):
        return source
    name = meta.get('source_file', '')
    if not name or Path(name).name != name:
        return source
    original = source.parent / name
    if original.is_file() and meta.get('source_sha256') == sha256(original):
        def rate(path):
            return next(x['avg_frame_rate'] for x in probe(path)['streams'] if x['codec_type'] == 'video')
        if rate(original) == rate(source):
            return original
    return source


def detect_gradual(source, stats_path, *, threshold=0.5, min_shot_seconds=0.25):
    import torch
    from scenedetect import FrameTimecode
    from vendor.transnetv2.model import TransNetV2
    torch.set_num_threads(2)
    weights = ensure_weights()
    inference_source = context_source(source)
    # Match the official inference preprocessing and 100-frame / 50-frame windows.
    raw = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-i', str(inference_source),
        '-map', '0:v:0', '-vf', 'scale=48:27', '-pix_fmt', 'rgb24',
        '-vsync', '0', '-f', 'rawvideo', '-',
    ])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 27, 48, 3)
    if not len(frames):
        raise ValueError('디코딩된 영상 프레임이 없습니다.')
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    black = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            # Maximum luminance preserves small bright text on black title cards.
            # Inspect full-resolution frames so small title text is not resized away.
            black.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).max() <= 16)
    finally:
        capture.release()
    if fps <= 0 or len(black) > len(frames) or (inference_source == source and len(black) != len(frames)):
        raise ValueError('디코더 사이 프레임 수가 다릅니다. CFR 입력 영상을 사용하세요.')
    model = TransNetV2()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.eval()
    padded = np.pad(frames, ((25, 25 + (50 - len(frames) % 50) % 50), (0, 0), (0, 0), (0, 0)), mode='edge')
    single, many = [], []
    with torch.inference_mode():
        for index in range(0, len(padded) - 50, 50):
            prediction, all_prediction = model(torch.from_numpy(padded[index:index+100][None].copy()))
            single.extend(torch.sigmoid(prediction)[0, 25:75, 0].numpy())
            many.extend(torch.sigmoid(all_prediction['many_hot'])[0, 25:75, 0].numpy())
    single, many = np.asarray(single[:len(black)]), np.asarray(many[:len(black)])
    spans, regions = contiguous_spans(single, many, black, fps, threshold=threshold,
                                  min_shot_seconds=min_shot_seconds)
    with Path(stats_path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'time_sec', 'single_probability', 'transition_probability', 'black'])
        for index in range(len(black)):
            writer.writerow([index, index/fps, float(single[index]), float(many[index]), int(black[index])])
    for item in regions:
        item['start_sec'], item['end_sec'] = item['start_frame']/fps, item['end_frame_exclusive']/fps
    scenes = [(FrameTimecode(start, fps=fps), FrameTimecode(end, fps=fps)) for start, end in spans]
    details = dict(total_frames=len(black), duration_sec=len(black)/fps, fps=fps,
                   inference_source=str(inference_source), inference_context_frames=len(frames),
                   excluded_ranges=[], excluded_frame_count=0, transition_ranges=regions,
                   boundary_policy='one_midpoint_per_transition; keep_all_frames',
                   model_sha256=WEIGHTS_SHA256, black_max_luma=16, fade_bridge_seconds=0.75)
    return scenes, details
