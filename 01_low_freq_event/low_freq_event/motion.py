"""Conservative camera-aware continuity; appearance is still required for every link."""
from pathlib import Path
import numpy as np


def camera_motion(previous_path: Path, current_path: Path, previous_boxes, current_boxes):
    import cv2
    def features(path, boxes):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None, None, None
        image = cv2.resize(image, (960, round(image.shape[0] * 960 / image.shape[1])))
        height, width = image.shape
        mask = np.full_like(image, 255)
        # Broadcast overlays and foreground people must not define camera motion.
        mask[:round(height*.12)] = 0
        mask[round(height*.85):] = 0
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(mask, (max(0, int(x1*width/1000)-4), max(0, int(y1*height/1000)-4)),
                          (min(width, int(x2*width/1000)+4), min(height, int(y2*height/1000)+4)), 0, -1)
        keypoints, descriptors = cv2.ORB_create(nfeatures=4000).detectAndCompute(image, mask)
        return keypoints, descriptors, (width, height)
    k1, d1, size1 = features(previous_path, previous_boxes)
    k2, d2, size2 = features(current_path, current_boxes)
    rejected = dict(valid=False, reason='insufficient_background_matches')
    if d1 is None or d2 is None or len(d1) < 16 or len(d2) < 16:
        return rejected
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    matches = [pair[0] for pair in pairs if len(pair)==2 and pair[0].distance < .72*pair[1].distance]
    if len(matches) < 16:
        return rejected
    a = np.float32([k1[m.queryIdx].pt for m in matches])
    b = np.float32([k2[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3.)
    if matrix is None or inliers is None:
        return rejected
    good = inliers.ravel().astype(bool)
    ratio = float(good.mean())
    scale = float(np.linalg.norm(matrix[:, 0]))
    span = np.ptp(a[good], axis=0) / size1
    if good.sum() < 16 or ratio < .45 or not .4 <= scale <= 2.5 or min(span) < .15:
        return dict(valid=False, reason='unreliable_transform', inliers=int(good.sum()), inlier_ratio=ratio)
    source_scale = np.diag([size1[0]/1000, size1[1]/1000, 1.])
    target_scale = np.diag([1000/size2[0], 1000/size2[1], 1.])
    affine = np.eye(3); affine[:2] = matrix
    normalized = target_scale @ affine @ source_scale
    return dict(valid=True, matrix=normalized[:2].tolist(), inliers=int(good.sum()),
                inlier_ratio=ratio, scale=scale)


def continuity(query, samples, motion, settings, blocked=False):
    """No coordinate-only links, no bridging cuts or long/unobserved gaps."""
    if blocked or not motion or not motion.get('valid'):
        return None
    previous = max(samples, key=lambda s: s.get('time_sec', -1), default={})
    dt = query['time_sec'] - previous.get('time_sec', -100)
    if not 0 < dt <= 1.5 or abs(previous.get('time_sec', -100) - motion['previous_time_sec']) > 1e-6:
        return None
    key = 'body_feature' if query['kind']=='person' else 'object_feature'
    if not query.get(key) or not previous.get(key):
        return None
    appearance = float(np.dot(query[key], previous[key]))
    threshold = .65 if query['kind']=='person' else .80
    if appearance < threshold or (query['kind'] != 'person' and query.get('detector_class') != previous.get('detector_class')):
        return None
    x1,y1,x2,y2 = previous['bbox']
    corners = np.array([[x1,y1,1],[x2,y1,1],[x1,y2,1],[x2,y2,1]]) @ np.array(motion['matrix']).T
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    qlo, qhi = np.array(query['bbox'][:2]), np.array(query['bbox'][2:])
    area = float(np.prod(hi-lo)); qarea = float(np.prod(qhi-qlo))
    if min(area, qarea) <= 0:
        return None
    size_ratio = qarea/area
    displacement = float(np.linalg.norm((qlo+qhi-lo-hi)/2)) / np.sqrt(area)
    intersection = float(np.prod(np.maximum(0, np.minimum(hi,qhi)-np.maximum(lo,qlo))))
    overlap = intersection / (area+qarea-intersection)
    if not .45 <= size_ratio <= 2.2 or displacement > .5 or overlap < .25:
        return None
    geometry = .5*overlap + .5*max(0., 1-displacement/.5)
    affinity = .10 + .25*(appearance-threshold)/(1-threshold) + .30*geometry
    return affinity, dict(identity_method='appearance_camera_continuity', identity_similarity=appearance,
        identity_iou=overlap, identity_camera_inlier_ratio=motion['inlier_ratio'],
        identity_gap_sec=dt)
