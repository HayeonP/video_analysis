"""CPU-only provisional track segmentation from source edits and cached evidence.

No identity labels enter this module. Sibling intervals remain free to reunify
in semantic review. Source observations are partitioned without interpolation.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 4


@dataclass(frozen=True)
class TemporalRepairConfig:
    abrupt_similarity: float = .45
    discontinuous_similarity: float = .50
    discontinuous_box_iou: float = .20
    discontinuous_gap_sec: float = .50
    sustained_similarity: float = .70
    adjacent_similarity: float = .65
    window_box_iou: float = .60
    window_stability: float = .60
    minimum_mean_confidence: float = .35
    minimum_area_fraction: float = .003
    context_observations: int = 3
    drift_minimum_observations: int = 24
    drift_endpoint_similarity: float = .50
    drift_endpoint_box_iou: float = .20
    drift_minimum_variance_gain: float = .25


def _unit(value):
    array = np.asarray(value, dtype=np.float32)
    return array / max(float(np.linalg.norm(array)), 1e-9)


def appearance_boundaries(observations, features, config=None, *, fps=None):
    """Flag a change only when adjacent appearance and stable windows agree."""
    from shot_tracking import box_iou
    config = config or TemporalRepairConfig()
    rows = sorted(observations, key=lambda row: row['frame'])
    result = []
    for index in range(1, len(rows)):
        previous, current = rows[index-1], rows[index]
        a, b = int(previous['frame']), int(current['frame'])
        if a not in features or b not in features:continue
        similarity = float(_unit(features[a]) @ _unit(features[b]))
        if fps and (b-a)/float(fps) > config.discontinuous_gap_sec and similarity < config.adjacent_similarity:
            result.append(dict(after_frame=a,before_frame=b,reason='unobserved_gap_with_changed_appearance',
                               adjacent_similarity=similarity,gap_sec=(b-a)/float(fps),
                               status='provisional_boundary_for_visual_review'))
            continue
        # Fast edits/occlusions can switch targets more quickly than a stable
        # multi-observation window forms. These are only review candidates: the
        # semantic stage may still reunify a fast-moving person's intervals.
        if similarity < config.discontinuous_similarity and box_iou(previous['bbox'],current['bbox']) < config.discontinuous_box_iou:
            result.append(dict(after_frame=a,before_frame=b,reason='joint_appearance_and_position_discontinuity',
                               adjacent_similarity=similarity,status='provisional_boundary_for_visual_review'))
            continue
        before = [r for r in rows[max(0,index-config.context_observations):index] if r['frame'] in features]
        after = [r for r in rows[index:index+config.context_observations] if r['frame'] in features]
        if len(before)<2 or len(after)<2:
            # A sparse endpoint may still expose an extreme appearance AND
            # position jump; preserve it for review rather than hide it inside
            # a different subject's atlas.
            if similarity < .35 and box_iou(previous['bbox'],current['bbox']) < .20:
                result.append(dict(after_frame=a,before_frame=b,reason='extreme_appearance_and_box_jump',
                                   adjacent_similarity=similarity,status='provisional_boundary_for_visual_review'))
            continue
        stability=min(float(np.mean([_unit(features[x['frame']]) @ _unit(features[y['frame']])
                    for x,y in zip(window,window[1:])])) for window in (before,after))
        if stability < config.window_stability:continue
        sustained=float(_unit(np.mean([_unit(features[r['frame']]) for r in before],axis=0)) @
                        _unit(np.mean([_unit(features[r['frame']]) for r in after],axis=0)))
        overlap=float(np.mean([box_iou(x['bbox'],y['bbox']) for x in before for y in after]))
        reason=None
        if similarity < config.abrupt_similarity and sustained < .60:
            reason='stable_windows_with_abrupt_appearance_change'
        elif similarity < config.adjacent_similarity and sustained < config.sustained_similarity and overlap < config.window_box_iou:
            reason='sustained_appearance_and_box_change'
        if reason:
            result.append(dict(after_frame=a,before_frame=b,reason=reason,adjacent_similarity=similarity,
                window_similarity=sustained,window_box_iou=overlap,window_stability=stability,
                status='provisional_boundary_for_visual_review'))
    return result


def _features_for_track(track, shot, output, frame_cache):
    from shot_tracking import box_iou
    result = {int(row['frame']): row['feature'] for row in track.get('representatives', []) if row.get('feature')}
    cache = shot.get('candidate_evidence', {}).get('detector_cache')
    if not cache:return result
    for row in track['observations']:
        path = output / cache / f"f{row['frame']:08d}.json"
        if path not in frame_cache:
            frame_cache[path] = json.loads(path.read_text())['detections'] if path.is_file() else []
        best = max(frame_cache[path], key=lambda item: box_iou(item['bbox'], row['bbox']), default=None)
        if best and best.get('feature') and box_iou(best['bbox'], row['bbox']) > .99:
            result[int(row['frame'])] = best['feature']
    return result


def drift_boundary(observations,features,existing=(),config=None):
    """Find gradual identity drift missed by adjacent-frame change tests.

    Long tracks whose endpoints differ in both appearance and location get a
    two-regime change-point proposal only if it substantially reduces descriptor
    variance. This is a temporal hypothesis, never an identity declaration.
    """
    from shot_tracking import box_iou
    config=config or TemporalRepairConfig();window=2*config.context_observations
    rows=sorted((r for r in observations if r['frame'] in features and r.get('confidence',0)>=.24),key=lambda r:r['frame'])
    if len(rows)<max(config.drift_minimum_observations,2*window):return None
    values=np.asarray([_unit(features[r['frame']]) for r in rows]);count=len(values)
    similarity=float(_unit(values[:window].mean(0)) @ _unit(values[-window:].mean(0)))
    overlap=box_iou(np.median([r['bbox'] for r in rows[:window]],axis=0),
                    np.median([r['bbox'] for r in rows[-window:]],axis=0))
    if similarity>=config.drift_endpoint_similarity or overlap>=config.drift_endpoint_box_iou:return None
    cumulative=np.cumsum(values,axis=0);baseline=float(np.sum((values-values.mean(0))**2))
    candidates=[]
    for index in range(window,count-window+1):
        gain=float(np.sum(cumulative[index-1]**2)/index+
                   np.sum((cumulative[-1]-cumulative[index-1])**2)/(count-index)-np.sum(cumulative[-1]**2)/count)
        candidates.append((gain,index))
    gain,index=max(candidates)
    if gain/max(baseline,1e-9)<config.drift_minimum_variance_gain:return None
    # Existing abrupt-cut candidates already isolate this change; avoid adding
    # near-duplicate tiny fragments and unnecessary semantic re-review.
    if any(abs(sum(r['frame']<flag['before_frame'] for r in rows)-index)<window for flag in existing):return None
    return dict(after_frame=rows[index-1]['frame'],before_frame=rows[index]['frame'],
                reason='long_track_appearance_and_location_drift',endpoint_similarity=similarity,
                endpoint_box_iou=overlap,variance_gain=gain/max(baseline,1e-9),
                transition_end_frame=rows[index+window]['frame'] if index+2*window<=count else None,
                status='provisional_boundary_for_visual_review')


def split_temporal_tracks(bundle, output, *, config=None, dry_run=False):
    """Apply source barriers even without SAM, then reuse the existing segmenter.

    dry_run skips segment materialization but does cache source-boundary scans.
    The returned report lists both barrier and appearance changes. Input is never
    mutated, and passing a repaired bundle again is idempotent.
    """
    from internal_scene_boundaries import apply_internal_barriers
    from mask_track_graph import segment_temporal_candidates
    config=config or TemporalRepairConfig();output=Path(output)
    if bundle.get('raw_temporal_repair',{}).get('version')==VERSION:
        return (bundle,bundle['raw_temporal_repair']) if dry_run else bundle
    refined=apply_internal_barriers(bundle,bundle,output) if not bundle.get('internal_boundaries') else deepcopy(bundle)
    changes=[]
    originals={s['shot_id']:[t['track_id'] for t in s['tracks']] for s in bundle['shots']}
    for shot in refined['shots']:
        frame_cache={}
        for track in shot['tracks']:
            rows=track['observations']
            if len(rows)<3:continue
            if np.mean([r.get('confidence',0) for r in rows]) < config.minimum_mean_confidence:continue
            area=np.median([(r['bbox'][2]-r['bbox'][0])*(r['bbox'][3]-r['bbox'][1]) for r in rows])
            if area / max(1,shot.get('width',1)*shot.get('height',1)) < config.minimum_area_fraction:continue
            features=_features_for_track(track,shot,output,frame_cache)
            flags=appearance_boundaries(rows,features,config,fps=shot.get('fps'))
            drift=drift_boundary(rows,features,flags,config)
            if drift:
                flags.append(drift)
                # A gradual switch has an ambiguous transition, unlike a hard
                # cut. Keep its first context window independently reviewable
                # so a mixed boundary crop cannot anchor the whole later track
                # to the earlier subject. Every observation remains available.
                if drift.get('transition_end_frame') is not None:
                    flags.append(dict(before_frame=drift['transition_end_frame'],
                        reason='end_of_gradual_drift_transition_window',
                        status='provisional_boundary_for_visual_review'))
            if flags:
                track['identity_boundary_candidates']=flags
                changes.append(dict(shot_id=shot['shot_id'],parent_track_id=track['track_id'],
                                    boundaries=flags,source_observations=len(rows)))
    barrier_changes=[s['shot_id'] for s in refined['shots'] if [t['track_id'] for t in s['tracks']]!=originals[s['shot_id']]]
    report=dict(version=VERSION,config=asdict(config),changed_shots=sorted(set(barrier_changes)|{r['shot_id'] for r in changes}),
                source_barrier_shots=barrier_changes,changes=changes,
                policy='Temporal hypotheses only; no manual labels, forced identity differences or removed detector observations.')
    if dry_run:return bundle,report
    # The common segmenter also remaps mask constraints. For raw detections there
    # are no masks: do not manufacture an empty mask graph that hides detector
    # co-visibility constraints downstream.
    had_mask_graph=bool(refined.get('mask_track_graph'))
    refined=segment_temporal_candidates(refined,output)
    if not had_mask_graph:
        refined.pop('mask_track_graph',None)
        for shot in refined['shots']:shot.pop('mask_graph',None)
    refined['raw_temporal_repair']=report
    refined['fingerprint']=hashlib.sha256(json.dumps([refined.get('fingerprint'),report],sort_keys=True).encode()).hexdigest()
    folder=output/'raw_temporal_repair';folder.mkdir(parents=True,exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (folder/'bundle.json').write_text(json.dumps(refined,ensure_ascii=False))
    return refined
