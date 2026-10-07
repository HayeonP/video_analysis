"""Cheap motion/appearance analysis and cumulative-change frame selection."""
from dataclasses import dataclass, asdict
import math
import cv2
import numpy as np


@dataclass
class Config:
    analysis_width: int = 256
    change_budget: float = 0.08
    min_fps: float = 1.0
    max_fps: float = 10.0
    motion_weight: float = 2.0
    visual_weight: float = 1.0
    camera_weight: float = 0.15
    smoothing_seconds: float = 0.12
    visual_guard: float = 0.18
    transient_guard: float = 0.25

    def validate(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key}는 0보다 큰 유한한 숫자여야 합니다.')
        if self.analysis_width < 64 or self.min_fps > self.max_fps:
            raise ValueError('analysis_width는 64 이상, min_fps는 max_fps 이하여야 합니다.')


def small_frame(frame, width):
    height = max(32, round(frame.shape[0] * min(width, frame.shape[1]) / frame.shape[1]))
    return cv2.resize(frame, (min(width, frame.shape[1]), height), interpolation=cv2.INTER_AREA)


def spatial_stat(values, mask, percentile=None):
    """Global change plus local tiles, so a small moving subject is not averaged away."""
    data = values[mask]
    if data.size == 0:
        return 0.0
    statistic = lambda x: float(np.percentile(x, percentile)) if percentile else float(x.mean())
    whole = statistic(data)
    tiles = []
    for ys in np.array_split(np.arange(values.shape[0]), 4):
        for xs in np.array_split(np.arange(values.shape[1]), 4):
            part = values[np.ix_(ys, xs)][mask[np.ix_(ys, xs)]]
            if len(part) >= 12:
                tiles.append(statistic(part))
    if not tiles:
        return whole
    local = max(tiles) if percentile else float(np.mean(sorted(tiles)[-3:]))
    return max(whole, local)


def appearance_change(previous, current, mask=None):
    error = np.abs(previous.astype(np.float32) - current.astype(np.float32)).mean(axis=2) / 255.0
    if mask is None:
        mask = np.ones(error.shape, dtype=bool)
    return spatial_stat(error, mask)


def motion_features(previous, current):
    """Farneback flow minus a spatially supported RANSAC camera transform."""
    a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    h, w = a.shape
    flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    gradient = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    gy, gx = np.mgrid[6:h-6:6, 6:w-6:6]
    textured = gradient[gy, gx] > 8
    points = np.stack((gx[textured], gy[textured]), axis=1).astype(np.float32)
    affine = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    accepted, inlier_ratio = False, 0.0
    if len(points) >= 20:
        moved = points + flow[points[:, 1].astype(int), points[:, 0].astype(int)]
        estimate, inliers = cv2.estimateAffinePartial2D(points, moved, method=cv2.RANSAC,
            ransacReprojThreshold=1.2, maxIters=500, confidence=.99, refineIters=5)
        if estimate is not None and inliers is not None:
            good = points[inliers.ravel().astype(bool)]
            inlier_ratio = float(inliers.mean())
            scale = float(np.hypot(estimate[0, 0], estimate[1, 0]))
            if (len(good) >= 12 and inlier_ratio >= .55 and
                np.ptp(good[:, 0]) >= .45*w and np.ptp(good[:, 1]) >= .35*h and
                .90 <= scale <= 1.10 and np.linalg.norm(estimate[:, 2]) < .20*max(h,w)):
                affine, accepted = estimate.astype(np.float32), True
    camera = np.stack((affine[0,0]*xx + affine[0,1]*yy + affine[0,2] - xx,
                       affine[1,0]*xx + affine[1,1]*yy + affine[1,2] - yy), axis=2)
    mask = gradient > 4
    mask[:4] = mask[-4:] = False
    mask[:, :4] = mask[:, -4:] = False
    diagonal = math.hypot(w, h)
    residual = np.linalg.norm(flow - camera, axis=2)
    motion = spatial_stat(residual, mask, 85) / diagonal
    raw_motion = spatial_stat(np.linalg.norm(flow, axis=2), mask, 85) / diagonal
    camera_motion = float(np.percentile(np.linalg.norm(camera, axis=2), 90)) / diagonal
    aligned = cv2.warpAffine(previous, affine, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones((h,w), np.uint8), affine, (w,h), flags=cv2.INTER_NEAREST) > 0
    valid[:3] = valid[-3:] = False
    valid[:, :3] = valid[:, -3:] = False
    return dict(motion=max(0.0, motion - .0002), raw_motion=raw_motion,
                camera_motion=camera_motion, visual=max(0.0, appearance_change(aligned, current, valid) - .002),
                raw_visual=appearance_change(previous, current), camera_compensated=accepted,
                camera_inlier_ratio=inlier_ratio)


class Selector:
    def __init__(self, fps, config):
        config.validate()
        self.fps, self.config = float(fps), config
        self.previous = self.anchor = None
        self.last_selected = 0
        self.accumulated = self.rate = 0.0
        self.indices, self.rows = [], []

    def push(self, frame, index):
        cfg = self.config
        if index != len(self.rows):
            raise ValueError('프레임을 0부터 순서대로 전달해야 합니다.')
        small = small_frame(frame, cfg.analysis_width)
        features = dict(motion=0., raw_motion=0., camera_motion=0., visual=0., raw_visual=0.,
                        camera_compensated=False, camera_inlier_ratio=0.)
        reason, anchor_change = '', 0.0
        if self.previous is None:
            reason = 'first'
            step = smoothed = 0.0
        else:
            features = motion_features(self.previous, small)
            step = (cfg.motion_weight * features['motion'] + cfg.visual_weight * features['visual'] +
                    cfg.camera_weight * features['camera_motion'])
            dt = 1 / self.fps
            alpha = 1 - math.exp(-dt / cfg.smoothing_seconds)
            self.rate += alpha * (step / dt - self.rate)
            smoothed = .5 * step + .5 * self.rate * dt
            self.accumulated += smoothed
            anchor_change = appearance_change(self.anchor, small)
            elapsed = (index - self.last_selected) / self.fps
            # A one-frame event can occur between nominal sampling instants.
            if features['raw_visual'] >= cfg.transient_guard:
                reason = 'transient'
            elif elapsed + 1e-12 >= 1 / cfg.min_fps:
                reason = 'max_interval'
            elif elapsed + 1e-12 >= 1 / cfg.max_fps:
                if anchor_change >= cfg.visual_guard:
                    reason = 'visual_guard'
                elif self.accumulated >= cfg.change_budget:
                    reason = 'change_budget'
        accumulated = self.accumulated
        if reason:
            self.indices.append(index)
            self.last_selected = index
            self.anchor = small.copy()
            self.accumulated = 0.0
        row = dict(frame=index, time_sec=index/self.fps, **features, step_score=step,
                   smoothed_score=smoothed, accumulated_score=accumulated, anchor_change=anchor_change,
                   selected=bool(reason), reason=reason)
        self.rows.append(row)
        self.previous = small
        return row

    def finish(self):
        if not self.rows:
            raise ValueError('영상에 디코딩할 프레임이 없습니다.')
        last = len(self.rows) - 1
        if self.indices[-1] != last:
            self.indices.append(last)
            self.rows[last]['selected'] = True
            self.rows[last]['reason'] = 'last'
        return self.indices, self.rows
