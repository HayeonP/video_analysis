"""Pre-semantic track graph from independently prompted SAM foreground masks.

Mask equality can support duplicate-box fusion. Mask separation is recorded as
geometric evidence, never proof that a face, arm, held object or animal belongs
to another physical identity. A hard veto applies only to automatic foreground
mask union. Physical identity constraints require semantic evidence elsewhere.

No movie labels, object counts, shot numbers, or Qwen calls appear here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import time

import numpy as np

LOG = logging.getLogger('entities.mask_track_graph')
VERSION = 3
APPEARANCE_VERSION = 1
DINO_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'


@dataclass(frozen=True)
class MaskGraphConfig:
    duplicate_iou: float = .80
    duplicate_fraction: float = .80
    duplicate_min_frames: int = 3
    singleton_duplicate_iou: float = .94
    separated_iou: float = .05
    separated_min_frames: int = 3
    hard_min_gap_ratio: float = .60
    representatives: int = 8


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def _area(box):
    return max(0., box[2]-box[0]) * max(0., box[3]-box[1])


def _box_iou(a, b):
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1.,_area(a)+_area(b)-inter)


def _box_gap_ratio(a, b):
    dx=max(0.,a[0]-b[2],b[0]-a[2]);dy=max(0.,a[1]-b[3],b[1]-a[3])
    # Normalize by the larger foreground's linear extent. Adjacent body parts
    # normally have zero gap; small heads/hands do not get a tiny denominator.
    return math.hypot(dx,dy)/max(1.,math.sqrt(max(_area(a),_area(b))))


def _asset(root, path):
    value=Path(path)
    return value if value.is_absolute() else root/value


def _read_mask(path):
    from PIL import Image
    with Image.open(path) as image:
        mask=np.asarray(image.convert('L'))>0
    y,x=np.nonzero(mask)
    if not len(x):return None
    box=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
    return dict(array=mask,bbox=box,pixels=int(len(x)))


def mask_pair_metrics(a, b):
    """Actual foreground pixels, never bounding-box overlap as a mask proxy."""
    if a['array'].shape!=b['array'].shape:
        raise ValueError('Foreground masks do not share source image coordinates')
    x1=max(a['bbox'][0],b['bbox'][0]);y1=max(a['bbox'][1],b['bbox'][1])
    x2=min(a['bbox'][2],b['bbox'][2]);y2=min(a['bbox'][3],b['bbox'][3])
    intersection=int(np.logical_and(a['array'][y1:y2,x1:x2],b['array'][y1:y2,x1:x2]).sum()) if x2>x1 and y2>y1 else 0
    return dict(mask_iou=intersection/max(1,a['pixels']+b['pixels']-intersection),
                intersection_over_smaller=intersection/max(1,min(a['pixels'],b['pixels'])),
                bbox_iou=_box_iou(a['bbox'],b['bbox']),
                area_ratio=min(a['pixels'],b['pixels'])/max(a['pixels'],b['pixels']),
                box_gap_ratio=_box_gap_ratio(a['bbox'],b['bbox']),
                left_pixels=a['pixels'],right_pixels=b['pixels'])


def collect_pair_evidence(shot, asset_root):
    """Load one frame's masks at a time so long videos cannot exhaust RAM."""
    root=Path(asset_root)
    frames=defaultdict(list)
    valid_counts=Counter()
    for track in shot['tracks']:
        for row in track['observations']:
            if row.get('mask') and row.get('evidence')=='sam2_mask':
                frames[int(row['frame'])].append((track['track_id'],row))
    pairs=defaultdict(list)
    for frame,rows in sorted(frames.items()):
        masks=[]
        for tid,row in sorted(rows,key=lambda item:item[0]):
            mask=_read_mask(_asset(root,row['mask']))
            if mask is not None:
                valid_counts[tid]+=1;masks.append((tid,mask,row))
        for index,(left,a,oa) in enumerate(masks):
            for right,b,ob in masks[index+1:]:
                metrics=mask_pair_metrics(a,b)
                pair=tuple(sorted([left,right]))
                metrics.update(frame=frame,left_mask=oa['mask'],right_mask=ob['mask'])
                pairs[pair].append(metrics)
    return pairs,valid_counts


def classify_pair(frames, left_count, right_count, config):
    """Separate same-foreground proof, review evidence, and mask-union vetoes."""
    if not frames:return dict(relation='no_shared_mask_evidence',hard=False)
    duplicate=[r for r in frames if r['mask_iou']>=config.duplicate_iou]
    separated=[r for r in frames if r['mask_iou']<=config.separated_iou
               and r['intersection_over_smaller']<=.10]
    far=[r for r in separated if r['box_gap_ratio']>=config.hard_min_gap_ratio
         and r['area_ratio']>=.20]
    share=len(duplicate)/len(frames)
    single=(min(left_count,right_count)==1 and len(frames)==1
            and frames[0]['mask_iou']>=config.singleton_duplicate_iou)
    enough=len(duplicate)>=config.duplicate_min_frames
    # A substantial contradictory foreground observation blocks an automatic
    # union, and may instead indicate an identity switch inside a raw track.
    if (single or enough) and share>=config.duplicate_fraction and not separated:
        relation='same_foreground';hard=True
    elif len(separated)>=config.separated_min_frames and len(far)>=config.separated_min_frames and not duplicate:
        relation='spatially_separated_foregrounds';hard=True
    elif separated:
        relation='distinct_mask_regions_require_semantic_check';hard=False
    elif duplicate:
        relation='partial_duplicate_or_identity_switch';hard=False
    else:
        relation='inconclusive_mask_overlap';hard=False
    return dict(relation=relation,hard=hard,hard_scope='foreground_regions',
                physical_identity_constraint=False,shared_frames=len(frames),duplicate_frames=len(duplicate),
                separated_frames=len(separated),far_separated_frames=len(far),
                duplicate_fraction=round(share,6),
                median_mask_iou=round(float(np.median([r['mask_iou'] for r in frames])),6),
                evidence_frames=[r['frame'] for r in (duplicate if relation=='same_foreground' else separated)[:8]],
                physical_identity_caveat='Even far-separated mask regions can be parts of one physical subject. Geometry alone never prohibits a shared person/animal/object identity.')


def whole_region_support(left, right, pair_rows, minimum_frames=3):
    """A third whole mask containing both regions prevents a body-part veto."""
    ids={tid for pair in pair_rows for tid in pair}
    for whole in sorted(ids-{left,right}):
        supports=[]
        for part in (left,right):
            pair=tuple(sorted([part,whole]));rows=pair_rows.get(pair,[])
            frames=set()
            for row in rows:
                whole_pixels=row['left_pixels'] if pair[0]==whole else row['right_pixels']
                part_pixels=row['left_pixels'] if pair[0]==part else row['right_pixels']
                if row['intersection_over_smaller']>=.80 and whole_pixels>=part_pixels:
                    frames.add(row['frame'])
            supports.append(frames)
        common=supports[0]&supports[1]
        if len(common)>=minimum_frames:
            return dict(whole_track=whole,frames=sorted(common)[:8],
                        reason='Both separated regions are contained in a shared independently prompted foreground mask')
    return None


def constrained_components(track_ids, positive_pairs, negative_pairs):
    """No transitive duplicate union may cross an existing foreground veto."""
    owner={tid:tid for tid in track_ids};groups={tid:{tid} for tid in track_ids}
    forbidden={frozenset(pair) for pair in negative_pairs}
    rejected=[]
    for left,right,score in sorted(positive_pairs,key=lambda row:-row[2]):
        a,b=owner[left],owner[right]
        if a==b:continue
        if any(frozenset([x,y]) in forbidden for x in groups[a] for y in groups[b]):
            rejected.append(dict(left=left,right=right,reason='transitive_separation_conflict'));continue
        first,last=sorted([a,b]);groups[first]|=groups.pop(last)
        for tid in groups[first]:owner[tid]=first
    return [sorted(group) for _,group in sorted(groups.items())],rejected


def identity_boundary_candidates(track, pair_rows, config):
    """Expose single-track changes of foreground affiliation without guessing IDs.

    A track matching different overlapping raw tracks at different times is a
    review boundary. Automatic splitting requires independent identities; mask
    changes alone are insufficient when pose or viewpoint changes rapidly.
    """
    associations=defaultdict(list)
    tid=track['track_id']
    for (left,right),frames in pair_rows.items():
        if tid not in (left,right):continue
        other=right if left==tid else left
        for row in frames:
            if row['mask_iou']>=config.duplicate_iou:
                associations[row['frame']].append((other,row['mask_iou']))
    ordered=[]
    for frame,options in sorted(associations.items()):
        other,score=max(options,key=lambda item:item[1]);ordered.append((frame,other,score))
    result=[]
    for previous,current in zip(ordered,ordered[1:]):
        if previous[1]==current[1]:continue
        between=pair_rows.get(tuple(sorted([previous[1],current[1]])),[])
        strong_separation=sum(row['mask_iou']<=config.separated_iou
                              and row['intersection_over_smaller']<=.1 for row in between)
        if strong_separation>=config.separated_min_frames:
            result.append(dict(after_frame=previous[0],before_frame=current[0],
                earlier_foreground_track=previous[1],later_foreground_track=current[1],
                reason='foreground affiliation changes between independently separated mask regions',
                status='requires_semantic_split_review',single_crop_identity_boundary=True))
    return result


def _merge_tracks(members, tracks, shot, config):
    from shot_tracking import select_representatives
    canonical=min(members)
    rows=defaultdict(list)
    aliases=set()
    for tid in members:
        aliases.update(tracks[tid].get('source_track_ids',[tid]))
        for original in tracks[tid]['observations']:
            row=deepcopy(original);row['_source_track']=tid;rows[int(row['frame'])].append(row)
    observations=[]
    for frame,options in sorted(rows.items()):
        def quality(row):
            mask=row.get('mask_quality',{})
            return (row.get('evidence')=='sam2_mask',mask.get('forward_backward_iou') or 0.,
                    mask.get('stability',0.),row.get('confidence',0.))
        chosen=deepcopy(max(options,key=quality))
        chosen['source_observation_refs']=[dict(track_id=r['_source_track'],frame=frame,
                                              bbox=r['bbox'],mask=r.get('mask')) for r in options]
        chosen.pop('_source_track',None)
        chosen['track_id']=canonical
        chosen['source_track_ids']=sorted(aliases)
        observations.append(chosen)
    result=deepcopy(tracks[canonical])
    result.update(track_id=canonical,source_track_ids=sorted(aliases),mask_graph_members=members,
                  observations=observations,representatives=select_representatives(observations,config.representatives),
                  first_frame=observations[0]['frame'],last_frame=observations[-1]['frame'],
                  observed_frames=len(observations),observed_duration_sec=len(observations)/shot['fps'],
                  span_duration_sec=(observations[-1]['frame']-observations[0]['frame']+1)/shot['fps'],
                  observation_coverage=len(observations)/(observations[-1]['frame']-observations[0]['frame']+1),
                  classification_status='mask_geometry_candidate_pending_semantics',
                  mask_graph_status='same_foreground_fused' if len(members)>1 else 'independent_candidate')
    # Only existing crop views may be selected. New SAM boxes lacking crops are
    # not relabeled as detector crop embeddings or exposed as a broken image.
    available=[r for r in observations if r.get('crop') and r.get('image')]
    result['representatives']=select_representatives(available,config.representatives) if available else deepcopy(
        tracks[canonical].get('representatives',[]))
    return result


def analyze_mask_track_graph(evidence_bundle, output, *, config=None):
    """Pure mask-file analysis; usable independently of a GPU/SAM installation."""
    config=config if isinstance(config,MaskGraphConfig) else MaskGraphConfig(**(config or {}))
    output=Path(output).resolve();asset_root=Path(evidence_bundle.get('asset_root',output))
    refined=deepcopy(evidence_bundle);records=[]
    for shot in refined['shots']:
        pairs,counts=collect_pair_evidence(shot,asset_root)
        decisions=[];positives=[];negatives=[];soft=[]
        for (left,right),frames in sorted(pairs.items()):
            decision=classify_pair(frames,counts[left],counts[right],config)
            if decision['relation']=='spatially_separated_foregrounds':
                support=whole_region_support(left,right,pairs,config.separated_min_frames)
                if support:
                    decision.update(relation='possible_parts_of_shared_foreground',hard=False,whole_region_support=support)
            decision.update(left=left,right=right)
            decisions.append(dict(decision,frames=frames))
            if decision['relation']=='same_foreground':positives.append((left,right,decision['median_mask_iou']))
            elif decision['hard']:negatives.append([left,right])
            elif decision['separated_frames']:soft.append([left,right])
        tracks={t['track_id']:t for t in shot['tracks']}
        groups,blocked=constrained_components(list(tracks),positives,negatives)
        mapping={tid:min(group) for group in groups for tid in group}
        boundaries={tid:identity_boundary_candidates(track,pairs,config) for tid,track in tracks.items()}
        boundaries={tid:rows for tid,rows in boundaries.items() if rows}
        shot['tracks']=[_merge_tracks(group,tracks,shot,config) for group in groups]
        for track in shot['tracks']:
            issues=[dict(source_track_id=tid,**r) for tid in track['mask_graph_members'] for r in boundaries.get(tid,[])]
            if issues:
                track['identity_boundary_candidates']=issues
                track['uncertain']=True
        hard=sorted({tuple(sorted([mapping[a],mapping[b]])) for a,b in negatives if mapping[a]!=mapping[b]})
        conditional=sorted({tuple(sorted([mapping[a],mapping[b]])) for a,b in soft if mapping[a]!=mapping[b]})
        edge_frames=defaultdict(set)
        for decision in decisions:
            a,b=mapping[decision['left']],mapping[decision['right']]
            if a==b:continue
            pair=tuple(sorted([a,b]))
            for row in decision['frames']:
                if row['mask_iou']<=config.separated_iou and row['intersection_over_smaller']<=.10:
                    kind='hard' if decision['hard'] and decision['relation']!='same_foreground' else 'conditional'
                    if kind=='hard' and row['box_gap_ratio']<config.hard_min_gap_ratio:continue
                    edge_frames[(kind,*pair)].add(row['frame'])
        audit_rel=f'mask_track_graph/shots/{shot["shot_id"]}.json'
        record=dict(shot_id=shot['shot_id'],input_tracks=len(tracks),output_tracks=len(groups),
                    mask_asset_root=str(asset_root),
                    cannot_link_scope='foreground_regions',
                    cannot_link_usage='Hard veto for automatic mask-region union only; conditional evidence for semantic physical-entity identity.',
                    physical_cannot_link=[],
                    source_to_candidate=mapping,duplicate_groups=[g for g in groups if len(g)>1],
                    cannot_link=[list(pair) for pair in hard],conditional_separation=[list(pair) for pair in conditional],
                    mask_union_veto=[list(pair) for pair in hard],
                    semantic_review_pairs=[list(pair) for pair in sorted(set(hard)|set(conditional))],
                    separation_frame_evidence=[dict(kind=kind,scope='foreground_regions',left=a,right=b,frames=sorted(frames))
                                               for (kind,a,b),frames in sorted(edge_frames.items())],
                    identity_boundary_candidates=boundaries,blocked_unions=blocked,
                    tracks_with_masks=len(counts),mask_observations=sum(counts.values()),audit=audit_rel)
        shot['mask_graph']=record
        _write(output/audit_rel,dict(record,pair_evidence=decisions))
        records.append(record)
        LOG.info('[MASK GRAPH] %s: %d → %d candidates; %d strict/%d conditional separation pairs',
                 shot['shot_id'],len(tracks),len(groups),len(hard),len(conditional))
    # All source assets remain valid when graph output is another directory.
    for shot in refined['shots']:
        rows=list(shot.get('scene_frames',[]))+list(shot.get('adaptive_scene_frames',[]))
        rows += [o for t in shot['tracks'] for o in t['observations']+t.get('representatives',[])]
        for row in rows:
            for field in ('image','crop','mask','masked_crop'):
                if row.get(field):row[field]=str(_asset(asset_root,row[field]).resolve())
            for source in row.get('source_observation_refs',[]):
                if source.get('mask'):source['mask']=str(_asset(asset_root,source['mask']).resolve())
    key=_digest(dict(version=VERSION,source=evidence_bundle.get('fingerprint'),config=asdict(config),records=records))
    summary=dict(version=VERSION,fingerprint=key,source_fingerprint=evidence_bundle.get('fingerprint'),
                 config=asdict(config),shots=records,input_tracks=sum(r['input_tracks'] for r in records),
                 output_tracks=sum(r['output_tracks'] for r in records),
                 hard_cannot_link=sum(len(r['cannot_link']) for r in records),
                 cannot_link_scope='foreground_regions',
                 physical_identity_constraints_from_geometry=0,
                 conditional_separation=sum(len(r['conditional_separation']) for r in records),
                 limitation='All geometric separation is conditional for physical identity, including far-separated foregrounds. Body parts and whole subjects require semantic role review; mask union vetoes must never force different human/animal IDs.')
    refined.update(fingerprint=key,pre_mask_graph_fingerprint=evidence_bundle.get('fingerprint'),
                   asset_root=str(output),mask_track_graph=summary)
    refined.setdefault('stats',{}).update(tracks=summary['output_tracks'],
        singleton_tracks=sum(len(t['observations'])==1 for s in refined['shots'] for t in s['tracks']),
        observed_track_boxes=sum(len(t['observations']) for s in refined['shots'] for t in s['tracks']))
    refined=segment_temporal_candidates(refined,output)
    _write(output/'mask_track_graph/bundle.json',refined)
    _write(output/'mask_track_graph/summary.json',refined['mask_track_graph'])
    return refined


def _materialize_segment_representatives(bundle, shot, track, output, limit=8):
    """Materialize real source pixels if an interval misses prior keyframes."""
    from shot_tracking import select_representatives
    if track.get('representatives'):return
    import av
    from PIL import Image
    selected=select_representatives(track['observations'],limit)
    requested={row['frame']:row for row in selected}
    token=_digest(track['track_id'])[:16]
    root=Path(bundle.get('asset_root',output))
    with av.open(shot['source']) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index not in requested:continue
            row=requested[index];image=frame.to_image().convert('RGB')
            box=[max(0,min(image.width,float(row['bbox'][0]))),max(0,min(image.height,float(row['bbox'][1]))),
                 max(0,min(image.width,float(row['bbox'][2]))),max(0,min(image.height,float(row['bbox'][3])))]
            rect=(math.floor(box[0]),math.floor(box[1]),math.ceil(box[2]),math.ceil(box[3]))
            if rect[2]<=rect[0] or rect[3]<=rect[1]:raise ValueError('Segment representative has an empty observed box')
            image_path=output/'mask_track_graph/segment_frames'/shot['shot_id']/f'f{index:08d}.jpg'
            crop_path=output/'mask_track_graph/segment_crops'/shot['shot_id']/f'{token}_f{index:08d}.jpg'
            image_path.parent.mkdir(parents=True,exist_ok=True);crop_path.parent.mkdir(parents=True,exist_ok=True)
            image.save(image_path,quality=94);crop=image.crop(rect);crop.save(crop_path,quality=96)
            row['image']=str(image_path);row['crop']=str(crop_path)
            row.pop('masked_crop',None);row.pop('masked_crop_background',None)
            if row.get('mask') and row.get('evidence')=='sam2_mask':
                with Image.open(_asset(root,row['mask'])) as mask_image:
                    foreground=np.asarray(mask_image.convert('L'))>0
                if foreground.shape!=(image.height,image.width):raise ValueError('Mask/source shape mismatch in temporal segment')
                pixels=np.asarray(crop).copy();inside=foreground[rect[1]:rect[3],rect[0]:rect[2]]
                pixels[~inside]=127
                masked_path=crop_path.with_suffix('.png');Image.fromarray(pixels).save(masked_path)
                row['masked_crop']=str(masked_path);row['masked_crop_background']=[127,127,127]
            # Freshly encoded bytes require fresh descriptor attribution later.
            if row.get('feature'):
                row.setdefault('source_crop_feature',row.pop('feature'))
                row.setdefault('source_crop_feature_provenance',row.pop('feature_provenance','previous crop before segment materialization'))
    missing=[row['frame'] for row in selected if not row.get('crop') or not Path(row['crop']).is_file()]
    if missing:raise ValueError(f'Original frames missing for temporal representative materialization: {missing}')
    track['representatives']=selected


def remap_temporal_constraints(graph, children, frame_sets):
    """A veto follows its actual supporting frame, never a disjoint interval."""
    support=defaultdict(set)
    for record in graph.get('separation_frame_evidence',[]):
        support[(record['kind'],*sorted([record['left'],record['right']]))].update(record['frames'])
    hard=set();conditional=set();evidence=[];warnings=[]
    for kind,field in [('hard','cannot_link'),('conditional','conditional_separation')]:
        for left,right in graph.get(field,[]):
            key=(kind,*sorted([left,right]));known=support.get(key,set())
            for a in children.get(left,[left]):
                for b in children.get(right,[right]):
                    if a==b:continue
                    shared=frame_sets.get(a,set()) & frame_sets.get(b,set())
                    if not shared:continue
                    relevant=shared&known
                    split=(children.get(left,[left])!=[left] or children.get(right,[right])!=[right])
                    pair=tuple(sorted([a,b]))
                    if relevant or not split:
                        (hard if kind=='hard' else conditional).add(pair)
                        evidence.append(dict(kind=kind,scope='foreground_regions',left=pair[0],right=pair[1],frames=sorted(relevant),
                                             source_pair=[left,right]))
                    elif not known:
                        # Old graph schemas without frame support cannot justify
                        # transferring a hard identity veto to each child interval.
                        conditional.add(pair)
                        warnings.append(dict(left=pair[0],right=pair[1],reason='split_constraint_lacks_frame_support',
                                             source_pair=[left,right]))
    conditional-=hard
    return dict(cannot_link=[list(p) for p in sorted(hard)],
                cannot_link_scope='foreground_regions',physical_cannot_link=[],
                mask_union_veto=[list(p) for p in sorted(hard)],
                semantic_review_pairs=[list(p) for p in sorted(hard|conditional)],
                conditional_separation=[list(p) for p in sorted(conditional)],
                separation_frame_evidence=evidence,constraint_remap_warnings=warnings)


def segment_temporal_candidates(bundle, output, *, representatives=8):
    """Turn flagged tracker identity changes into provisional interval candidates.

    This does not declare different identities. VLM may assign every sibling
    interval to the same entity. Existing observations are partitioned exactly;
    no interpolated frames or new visibility claims are created.
    """
    output=Path(output).expanduser().resolve();refined=deepcopy(bundle)
    all_records=[]
    for shot in refined['shots']:
        original=list(shot['tracks']);children={};replacement=[];records=[]
        for track in original:
            observations=sorted(track['observations'],key=lambda row:row['frame'])
            flags=track.get('identity_boundary_candidates',[])
            cuts=sorted({int(flag['before_frame']) for flag in flags
                         if observations[0]['frame']<int(flag['before_frame'])<=observations[-1]['frame']}) if observations else []
            groups=[]
            for row in observations:
                index=sum(row['frame']>=cut for cut in cuts)
                while len(groups)<=index:groups.append([])
                groups[index].append(row)
            groups=[group for group in groups if group]
            if len(groups)<=1:
                children[track['track_id']]=[track['track_id']];replacement.append(track);continue
            parent=track['track_id'];child_ids=[f'{parent}:C{i:03d}' for i in range(1,len(groups)+1)]
            children[parent]=child_ids
            for cid,rows in zip(child_ids,groups):
                child=deepcopy(track);frames={row['frame'] for row in rows}
                child['observations']=rows
                by_frame={row['frame']:row for row in rows}
                child['representatives']=[deepcopy(row) for row in track.get('representatives',[])
                    if row['frame'] in frames and list(row['bbox'])==list(by_frame[row['frame']]['bbox'])
                    and row.get('crop') and row.get('image')]
                child.update(track_id=cid,first_frame=rows[0]['frame'],last_frame=rows[-1]['frame'],
                    observed_frames=len(rows),observed_duration_sec=len(rows)/shot['fps'],
                    span_duration_sec=(rows[-1]['frame']-rows[0]['frame']+1)/shot['fps'],
                    observation_coverage=len(rows)/(rows[-1]['frame']-rows[0]['frame']+1),
                    mean_confidence=float(np.mean([row.get('confidence',0) for row in rows])),
                    source_identity_boundary_candidates=deepcopy(flags),identity_boundary_candidates=[],
                    temporal_candidate=dict(parent_track_id=parent,sibling_track_ids=child_ids,
                        first_frame=rows[0]['frame'],last_frame=rows[-1]['frame'],
                        same_identity_allowed=True,status='provisional_interval_for_visual_identity_review'),
                    classification_status='temporal_candidate_pending_semantics',uncertain=True)
                child['occlusion_gaps']=[dict(start_frame=a['frame']+1,end_frame_exclusive=b['frame'],
                    duration_sec=(b['frame']-a['frame']-1)/shot['fps'])
                    for a,b in zip(rows,rows[1:]) if b['frame']>a['frame']+1]
                for row in child['observations']+child['representatives']:
                    row.setdefault('source_temporal_track_id',parent);row['track_id']=cid
                _materialize_segment_representatives(refined,shot,child,output,representatives)
                replacement.append(child)
            # Equality of the frame multiset catches both omissions and duplicates.
            before=Counter(row['frame'] for row in observations)
            after=Counter(row['frame'] for child in replacement if child['track_id'] in child_ids for row in child['observations'])
            if before!=after:raise AssertionError('Temporal candidates changed source observation coverage')
            records.append(dict(parent_track_id=parent,children=child_ids,cuts=cuts,
                source_track_ids=track.get('source_track_ids',[parent]),boundaries=deepcopy(flags),
                observation_count=sum(before.values()),coverage_verified=True,
                policy='Sibling candidates can share one entity; segmentation does not assert distinct physical identities.'))
        shot['tracks']=replacement
        graph=shot.setdefault('mask_graph',{})
        if records:
            frame_sets={track['track_id']:{row['frame'] for row in track['observations']} for track in replacement}
            graph.update(remap_temporal_constraints(graph,children,frame_sets))
            old_mapping=deepcopy(graph.get('source_to_candidate',{}))
            expanded={source:children.get(target,[target]) for source,target in old_mapping.items()}
            graph['source_to_candidate_before_segmentation']=old_mapping
            graph['source_to_candidates']=expanded
            graph['source_to_candidate']={source:targets[0] for source,targets in expanded.items() if len(targets)==1}
            graph['temporal_candidate_segments']=records;graph['output_tracks']=len(replacement)
            _write(output/'mask_track_graph/segments'/f'{shot["shot_id"]}.json',
                   dict(shot_id=shot['shot_id'],segments=records,constraints={key:graph[key] for key in
                        ('cannot_link','conditional_separation','separation_frame_evidence','constraint_remap_warnings')}))
        all_records.extend(dict(shot_id=shot['shot_id'],**record) for record in records)
    if not all_records:return refined
    previous=bundle.get('fingerprint')
    summary=refined.setdefault('mask_track_graph',{})
    summary.update(version=VERSION,shots=[shot.get('mask_graph',{}) for shot in refined['shots']],
                   fused_tracks_before_segmentation=sum(len(shot['tracks']) for shot in bundle['shots']),
                   output_tracks=sum(len(shot['tracks']) for shot in refined['shots']),
                   temporal_candidate_segments=all_records,
                   hard_cannot_link=sum(len(shot.get('mask_graph',{}).get('cannot_link',[])) for shot in refined['shots']),
                   conditional_separation=sum(len(shot.get('mask_graph',{}).get('conditional_separation',[])) for shot in refined['shots']))
    key=_digest(dict(version=VERSION,source=previous,segments=all_records,constraints=summary['shots']))
    summary['fingerprint']=key
    refined.update(fingerprint=key,pre_temporal_segmentation_fingerprint=previous)
    refined.setdefault('stats',{}).update(tracks=summary['output_tracks'],
        singleton_tracks=sum(len(t['observations'])==1 for s in refined['shots'] for t in s['tracks']),
        observed_track_boxes=sum(len(t['observations']) for s in refined['shots'] for t in s['tracks']))
    return refined


def build_mask_track_graph(bundle, output, device='cuda:0', model='facebook/sam2.1-hiera-small',
                           *, config=None, mask_config=None, force=False, backend=None, model_revision=None):
    """Run independent raw-track SAM evidence, then constrain duplicate fusion."""
    from mask_refinement import raw_track_masks
    started=time.perf_counter()
    evidence,mask_summary=raw_track_masks(bundle,output,device=device,model_path=model,
        config=mask_config,force=force,backend=backend,model_revision=model_revision)
    from internal_scene_boundaries import apply_internal_barriers
    evidence=apply_internal_barriers(evidence,bundle,output)
    refined=analyze_mask_track_graph(evidence,output,config=config)
    refined['mask_track_graph']['elapsed_sec']=time.perf_counter()-started
    refined['mask_track_graph']['mask_evidence_summary']=mask_summary
    _write(Path(output)/'mask_track_graph/bundle.json',refined)
    _write(Path(output)/'mask_track_graph/summary.json',refined['mask_track_graph'])
    return refined


def _file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _encode_images(encoder, images):
    """Use the existing pinned DINO model/processor; injection keeps tests CPU-only."""
    if hasattr(encoder,'encode_images'):
        vectors=np.asarray(encoder.encode_images(images),dtype=np.float32)
    else:
        import torch
        batch=encoder.processor(images=images,return_tensors='pt').to(encoder.device)
        with torch.inference_mode():
            vectors=encoder.model(**batch).last_hidden_state[:,0].float().cpu().numpy()
    if vectors.ndim!=2 or len(vectors)!=len(images) or not np.isfinite(vectors).all():
        raise ValueError('Appearance encoder returned invalid vectors')
    lengths=np.linalg.norm(vectors,axis=1,keepdims=True)
    if (lengths<=1e-9).any():raise ValueError('Appearance encoder returned zero vectors')
    return np.round(vectors/lengths,7).tolist()


def encode_masked_representatives(bundle, output, device='cuda:0', *, model='facebook/dinov2-small',
                                  revision=DINO_REVISION, batch_size=16, force=False, encoder=None,
                                  save_path=None):
    """Recompute descriptors for actual representative crop bytes, preferring masks.

    A mask changes the input crop, so a detector-crop descriptor is never reused
    under a foreground-mask label. Cache keys include image bytes, model name,
    exact model commit and extraction version. Original vectors are retained as
    ``source_crop_feature`` when replaced, with their original provenance.
    """
    from PIL import Image
    from shot_tracking import DenseTrackingConfig, DinoEncoder
    if not re.fullmatch(r'[0-9a-f]{40}',revision):
        raise ValueError('Appearance extraction requires an explicit pinned model commit')
    if not 1<=batch_size<=64:raise ValueError('batch_size must be in 1..64')
    output=Path(output).expanduser().resolve();root=Path(bundle.get('asset_root',output))
    destination=Path(save_path).expanduser() if save_path is not None else output/'mask_track_graph/bundle.json'
    if not destination.is_absolute():destination=destination.resolve()
    refined=deepcopy(bundle)
    requests={};rows=[]
    for shot in refined['shots']:
        for track in shot['tracks']:
            for row in track.get('representatives',[]):
                field='masked_crop' if row.get('masked_crop') else 'crop'
                if not row.get(field):
                    raise ValueError(f'Representative has no image asset: {track["track_id"]} frame {row.get("frame")}')
                path=_asset(root,row[field]).resolve()
                if not path.is_file():raise FileNotFoundError(f'Representative crop is unavailable: {path}')
                sha=_file_sha256(path)
                spec=dict(version=APPEARANCE_VERSION,model=model,revision=revision,image_sha256=sha)
                key=_digest(spec)
                requests.setdefault(key,dict(spec=spec,path=path))
                rows.append((row,key,field,str(path),track['track_id']))
    cache=output/'mask_track_graph/appearance_cache'
    vectors={};pending=[];cache_hits=0
    for key,request in requests.items():
        path=cache/f'{key}.json'
        if path.is_file() and not force:
            value=json.loads(path.read_text())
            feature=np.asarray(value.get('feature',[]),dtype=np.float32)
            if value.get('spec')==request['spec'] and feature.ndim==1 and len(feature) and np.isfinite(feature).all():
                vectors[key]=feature.tolist();cache_hits+=1;continue
        pending.append((key,request))
    owns_encoder=encoder is None;started=time.perf_counter()
    try:
        if pending and encoder is None:
            encoder=DinoEncoder(DenseTrackingConfig(appearance_model=model,appearance_revision=revision),device)
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size];images=[]
            for _,request in batch:
                with Image.open(request['path']) as image:images.append(image.convert('RGB'))
            encoded=_encode_images(encoder,images)
            for (key,request),feature in zip(batch,encoded):
                vectors[key]=feature
                _write(cache/f'{key}.json',dict(spec=request['spec'],feature=feature))
            LOG.info('[MASK APPEARANCE] %d/%d unique crops encoded',min(start+batch_size,len(pending)),len(pending))
    finally:
        if owns_encoder and encoder is not None:
            encoder=None
            import gc
            gc.collect()
            if str(device).startswith('cuda'):
                import torch
                torch.cuda.empty_cache()
    for row,key,field,path,tid in rows:
        old_provenance=row.get('feature_provenance')
        expected=dict(model=model,revision=revision,image_sha256=requests[key]['spec']['image_sha256'],
                      image=path,input_kind='sam_foreground_masked_crop' if field=='masked_crop' else 'representative_crop',
                      extraction_version=APPEARANCE_VERSION,source_track_id=tid)
        if row.get('feature') and old_provenance!=expected:
            row.setdefault('source_crop_feature',row['feature'])
            row.setdefault('source_crop_feature_provenance',old_provenance or 'previous source crop; original pipeline provenance')
        row['feature']=vectors[key]
        row['feature_provenance']=expected
    before=bundle.get('pre_appearance_fingerprint',bundle.get('fingerprint'))
    summary=dict(version=APPEARANCE_VERSION,model=model,revision=revision,batch_size=batch_size,
                 representative_rows=len(rows),unique_images=len(requests),cache_hits=cache_hits,
                 encoded_images=len(pending),masked_rows=sum(field=='masked_crop' for _,_,field,_,_ in rows),
                 elapsed_sec=time.perf_counter()-started,
                 provenance='Descriptors correspond to the exact referenced crop bytes; original crop descriptors remain separately attributed.')
    refined.update(pre_appearance_fingerprint=before,appearance_embedding=summary,
                   fingerprint=_digest(dict(source=before,version=APPEARANCE_VERSION,model=model,revision=revision,
                                           assignments=[(tid,row['frame'],key,field) for row,key,field,_,tid in rows])))
    _write(destination.parent/'appearance.json',summary)
    _write(destination,refined)
    return refined


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--model',default='facebook/sam2.1-hiera-small')
    parser.add_argument('--force',action='store_true');parser.add_argument('--evidence-only',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    bundle=json.loads(args.bundle.read_text())
    result=analyze_mask_track_graph(bundle,args.output) if args.evidence_only else build_mask_track_graph(
        bundle,args.output,args.device,args.model,force=args.force)
    print(json.dumps({k:v for k,v in result['mask_track_graph'].items() if k not in {'shots','mask_evidence_summary'}},ensure_ascii=False))
