#!/usr/bin/env python3
"""Strong detection -> observation fusion -> visual links -> bounded semantic review."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import fcntl
from fractions import Fraction
import gc
import logging
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event'),str(ROOT.parent/'03_adaptive_sampling')]
from run_timing import RunTiming, utc_now, format_duration
START=time.perf_counter();STARTED=utc_now()
import cv2
import torch
from PIL import Image
from low_freq_event.config import Settings
from low_freq_event.detection import ObjectEncoder
from low_freq_event.storage import atomic_json, read_json
from extract_entities import prepare_inputs
from entity_models import observations, iou
from entity_output import export_results
from strong_detector import StrongDetectorConfig, detect_proposals, VERSION as DETECTOR_VERSION
from efficient_tracks import build_tracklets,rank_candidates
from bounded_matching import export_mapping
from anchored_matching import match_anchored
from duplicate_tracks import resolve_assigned_duplicates
from compare_results import compare_results
from result_report import write_run_reports
from pipeline_config import load_config, select_input, AppearanceConfig
from observation_fusion import fuse_frame_observations,reconcile_duplicate_tracklets

LOG=logging.getLogger('entities')
VERSION=4


def prepare_features(frames,output,settings,proposals,model,policy):
    signature=dict(version=2,detector=read_json(output/'yolo_proposals.meta.json')['fingerprint'],
                   class_kinds=policy.class_kinds,
                   settings=settings.model_dump(),proposals_sha256=hashlib.sha256((output/'yolo_proposals.json').read_bytes()).hexdigest())
    meta=output/'features.meta.json'
    if not meta.exists() or read_json(meta)!=signature:
        for name in ('features','crops','grounding'):
            if (output/name).exists():shutil.rmtree(output/name)
        atomic_json(meta,signature)
    missing=[f for f in frames if not (output/'features'/f'{f["frame_key"]}.json').exists()]
    LOG.info('Features: %d cached / %d new',len(frames)-len(missing),len(missing))
    if not missing:return
    encoder=ObjectEncoder(settings)
    for index,f in enumerate(missing):
        rows=[]
        for p in proposals[f['frame_key']]:
            kind=policy.kind_for(p['category'])
            rows.append(dict(kind=kind,category=p['category'],label='인물' if kind=='person' else p['category'],
                appearance=f'{model} class {p["category"]}',bbox=p['bbox'],detection_confidence=p['confidence'],uncertain=False))
        atomic_json(output/'grounding'/f'{f["frame_key"]}.json',dict(frame_key=f['frame_key'],detections=rows,source=model))
        with Image.open(output/f['image']) as im:
            raw=observations(encoder,im.convert('RGB'),f,rows,proposals[f['frame_key']],output,use_qwen=False)
        people=[o for o in raw if o['kind']=='person']
        crops=[]
        for o in people:
            with Image.open(output/o['crop']) as crop:crops.append(crop.convert('RGB'))
        for o,vector in zip(people,encoder.object_features(crops)):o['visual_feature']=vector
        for o in raw:
            match=max(proposals[f['frame_key']],key=lambda p:iou(p['bbox'],o['bbox']),default={})
            o['detector_class']=match.get('category',o.get('detector_class'))
            o['grounding_source']=match.get('source',f'{model} detector')
            o['detector_provenance']={k:match[k] for k in ('source','tile_xyxy','alternative_categories') if k in match}
        atomic_json(output/'features'/f'{f["frame_key"]}.json',raw)
        if index%25==0:LOG.info('Features %d/%d',index+1,len(missing))
    del encoder;gc.collect();torch.cuda.empty_cache()


def off_matching(groups,settings,config=None):
    config=config or AppearanceConfig()
    memory={};assignments={}
    for key,entry in groups.items():
        candidates=rank_candidates(entry,memory,limit=2)
        # Only very strong distinctive appearance links; this is the no-VLM baseline.
        target=None
        if candidates and candidates[0]['score']>=config.off_match_threshold and (len(candidates)==1 or candidates[0]['score']-candidates[1]['score']>=config.off_match_margin):
            target=candidates[0]['entity_id']
        if target is None:
            target=f'G{len(memory)+1:04d}';memory[target]=dict(deepcopy(entry),members=[])
            memory[target]['observations']=[]
        memory[target]['members'].append(key);memory[target]['observations'].extend(entry['observations'])
        assignments[key]=target
    return memory,assignments,[]


def defer_conflicting_tracks(groups,memory,mapping,audit):
    """Keep ambiguous observed boxes as provisional identities; never erase them."""
    conflicts=audit.get('remaining_conflicts',[])
    tracks=set()
    for conflict in conflicts:
        ids=conflict['track_ids']
        if len(ids)!=len(set(ids)):
            raise RuntimeError('한 트랙에 같은 프레임의 관측이 중복되었습니다.')
        tracks.update(ids)
    resolutions=[]
    for track in sorted(tracks):
        previous=mapping[track];key=f'U_conflict_{track}'
        memory[key]=dict(label='미확정 대상',appearance='동시 관측 충돌로 기준 대상 연결을 보류했습니다.',
                        kind=memory[previous]['kind'],observations=groups[track]['observations'],
                        identity_status='unresolved',members=[track])
        mapping[track]=key
        resolutions.append(dict(track_id=track,previous=previous,assigned=key,reason='unresolved co-visibility conflict'))
    audit['conflict_resolutions']=resolutions


def fused_tracklets(raw,frames,settings,policy,output):
    # Preserve repeated duplicate evidence before any frame-level suppression.
    provisional=build_tracklets(raw,frames,settings,config=policy.tracking)
    provisional={f'P{key[1:]}':entry for key,entry in provisional.items()}
    fusion_settings=dict(face_conflict_quality=settings.face_conflict_quality,observation_fusion=policy.fusion)
    reconciled,aliases,track_audit=reconcile_duplicate_tracklets(provisional,fusion_settings)
    track_audit['aliases']=aliases
    reduced=[obs for entry in reconciled.values() for obs in entry['observations']]
    fused,frame_audit=fuse_frame_observations(reduced,fusion_settings)
    atomic_json(output/'track_fusion.json',track_audit)
    atomic_json(output/'observation_fusion.json',frame_audit)
    # Rebuild clean local tracks from surviving actual detections. No VLM ID is
    # required for either fusion pass; raw feature files remain unchanged.
    return build_tracklets(fused,frames,settings,config=policy.tracking)


def refresh_export_quality(output,catalog,duplicates):
    """Report the final catalog after duplicate suppression/conflict deferral.

    Matching-stage failures remain visible. Only the derived unresolved-count
    warning is replaced, since export can remove empty groups or defer links.
    """
    path=output/'anchored/quality.json'
    quality=read_json(path)
    old_unresolved=quality.get('unresolved_identity_groups',0)
    old_count_warning=f'동일성이 미확정인 관측 묶음이 {old_unresolved}개 남았습니다. 서로 다른 개체임을 뜻하지 않습니다.'
    previous_export_warnings=set(quality.get('export_warnings',[]))
    warnings=[warning for warning in quality.get('warnings',[])
              if warning!=old_count_warning and warning not in previous_export_warnings]
    entities=catalog['entities']
    unresolved_ids=[entry['entity_id'] for entry in entities if entry.get('identity_status')=='unresolved']
    deferred=duplicates.get('conflict_resolutions',[])
    export_warnings=[]
    if unresolved_ids:
        export_warnings.append(f'동일성이 미확정인 관측 묶음이 {len(unresolved_ids)}개 남았습니다. 서로 다른 개체임을 뜻하지 않습니다.')
    if deferred:
        export_warnings.append(f'동시 관측 충돌로 {len(deferred)}개 트랙의 동일성 연결을 보류했습니다.')
    warnings=list(dict.fromkeys(warnings+export_warnings))
    quality.setdefault('pre_export_identity_counts',dict(
        unresolved_identity_groups=old_unresolved,
        resolved_identity_groups=quality.get('resolved_identity_groups',0)))
    quality.update(status='complete_with_warnings' if warnings else 'complete',warnings=warnings,
        export_warnings=export_warnings,unresolved_identity_groups=len(unresolved_ids),
        resolved_identity_groups=len(entities)-len(unresolved_ids),final_entities=len(entities),
        unresolved_entity_ids=unresolved_ids,
        post_export_deferred_tracks=[item['track_id'] for item in deferred],
        counts_source='entities.json after duplicate suppression and co-visibility conflict deferral')
    atomic_json(path,quality)
    summary_path=output/'anchored/summary.json'
    if summary_path.exists():
        summary=read_json(summary_path)
        summary.setdefault('pre_export_final_entities',summary.get('final_entities'))
        summary.update(final_entities=len(entities),quality=quality)
        atomic_json(summary_path,summary)
    return quality


def run_mode(args,mode,output,seed=None):
    # A --mode run targeting a both-mode subfolder shares the same lock.
    # Two invocations must never consume the same persisted budget concurrently.
    output.mkdir(parents=True,exist_ok=True)
    if output.resolve()==args.output.resolve():
        return _run_mode_locked(args,mode,output,seed)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_mode_locked(args,mode,output,seed)


def _run_mode_locked(args,mode,output,seed=None):
    output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    policy=args.policy;detector=policy.detector
    settings=Settings(perception_device='auto',person_confidence=detector.person_confidence,
        object_confidence=detector.object_confidence,
        tracking_low_confidence=min(.20,.5*min(detector.person_confidence,detector.object_confidence)),json_retries=0)
    identity_strategy='visual_catalog_v3' if mode=='with_qwen' else 'appearance_retrieval'
    config=dict(pipeline='efficient',version=VERSION,identity_strategy=identity_strategy,input=str(args.input.resolve()),mode=mode,
                detector=detector.model,detector_config=asdict(detector),detector_version=DETECTOR_VERSION,
                policy=policy.to_dict(),settings=settings.model_dump(),qwen_limit=args.qwen_max_calls,batch_shots=args.batch_shots)
    config=json.loads(json.dumps(config,ensure_ascii=False,allow_nan=False))
    path=output/'run_config.json'
    if path.exists() and read_json(path)!=config:raise ValueError('설정이 바뀌었습니다. 새 --output 경로를 사용하세요.')
    atomic_json(path,config)
    def step(name,label,op,*p,**kw):
        with args.timing.stage(f'{mode}.{name}',f'{mode} / {label}'):return op(*p,**kw)
    frames,timeline=step('inputs','입력 준비',prepare_inputs,args.input.resolve(),output)
    if (output/'timeline.json').exists() and read_json(output/'timeline.json')!=timeline:raise ValueError('입력 변경: 새 output을 사용하세요.')
    atomic_json(output/'samples.json',frames);atomic_json(output/'timeline.json',timeline)
    atomic_json(output/'status.json',dict(status='running',samples=len(frames),pipeline='efficient'))
    try:
        if seed:
            if read_json(seed/'timeline.json')!=timeline:raise ValueError('YOLO 공유 입력 불일치')
            for name in ('yolo_proposals.json','yolo_proposals.meta.json','detector_summary.json'):
                if (seed/name).exists():shutil.copy2(seed/name,output/name)
            proposals=read_json(output/'yolo_proposals.json')
            LOG.info('OFF reuses this run\'s detector proposals only; no Qwen-derived data shared.')
        else:proposals=step('detector',f'{detector.model} + 작은 대상 타일 검출',detect_proposals,frames,output,settings,detector)
        step('features','얼굴·몸·범용 시각 특징',prepare_features,frames,output,settings,proposals,detector.model,policy)
        raw=[o for f in frames for o in read_json(output/'features'/f'{f["frame_key"]}.json')]
        groups=step('tracks','동일 샷 중복 관측 융합·일대일 추적',fused_tracklets,raw,frames,settings,policy,output)
        weak=[k for k,e in groups.items() if len(e['observations'])<2 and max(o.get('detection_confidence',0) for o in e['observations'])<policy.filtering.singleton_confidence]
        for key in weak:groups[key]['low_observation_support']=True
        atomic_json(output/'tracklets.json',groups)
        atomic_json(output/'weak_tracklets.json',dict(ids=weak,excluded=policy.filtering.drop_weak_singletons,
            confidence_threshold=policy.filtering.singleton_confidence,
            reason='A single weak observation may be genuine in a short or sparsely sampled shot. Retained unless explicit filtering is enabled.'))
        if policy.filtering.drop_weak_singletons:groups={k:e for k,e in groups.items() if k not in set(weak)}
        LOG.info('Tracklets: %d; weak singletons: %d (%s)',len(groups),len(weak),'excluded' if policy.filtering.drop_weak_singletons else 'retained')
        atomic_json(output/'identity_before_review.json',dict(assignments={o['detection_id']:dict(entity_id=k) for k,e in groups.items() for o in e['observations']}))
        if mode=='with_qwen':
            memory,mapping,joins=step('anchored','외형 연결 → 의미 검토 → 전체 관계 검토',match_anchored,groups,frames,timeline,output,settings,args.qwen_max_calls,args.batch_shots,config=policy.matching,visual_config=policy.visual_association)
        else:memory,mapping,joins=step('association','외형 기반 연결',off_matching,groups,settings,policy.appearance)
        # Corrections are identity-layer annotations; original features and boxes stay immutable.
        export_groups=deepcopy(groups)
        for key,entry in export_groups.items():
            if key in mapping:
                for obs in entry['observations']:obs['entity_kind']=memory[mapping[key]]['kind']
        export_groups,mapping,duplicates=resolve_assigned_duplicates(export_groups,mapping,settings)
        defer_conflicting_tracks(export_groups,memory,mapping,duplicates)
        atomic_json(output/'duplicate_observations.json',duplicates)
        method=('observation fusion, visual association, bounded semantic review and catalog relationship review' if mode=='with_qwen'
                else 'appearance tracking and conservative visual retrieval (no Qwen)')
        registry,visual,observed=export_mapping(export_groups,memory,mapping,identity_method=method)
        atomic_json(output/'identity_review.json',dict(qwen_enabled=mode=='with_qwen',joins=joins,aliases={}))
        result=step('export','카탈로그·행렬·영상 저장',export_results,output,frames,timeline,observed,registry,visual,mode=mode)
        catalog=read_json(output/'entities.json')
        catalog['identification']=f'{detector.model} + observation fusion + SFace/OSNet/DINO tracklets'+(' + visual association / bounded Qwen semantic review / catalog relationships' if mode=='with_qwen' else ' / no Qwen')
        atomic_json(output/'entities.json',catalog)
        result.update(status='complete',pipeline='efficient',identity_strategy=identity_strategy,elapsed_sec=time.perf_counter()-start)
        result['elapsed_human']=format_duration(result['elapsed_sec'])
        if mode=='with_qwen':
            result['qwen_generations']=read_json(output/'qwen_budget.json')['generations']
            result['qwen_limit']=args.qwen_max_calls
            quality=refresh_export_quality(output,catalog,duplicates)
            result.update(status=quality['status'],warnings=quality['warnings'],identity_quality=quality)
            for warning in quality['warnings']:LOG.warning('%s: %s',mode,warning)
        atomic_json(output/'status.json',result);return result
    except BaseException as exc:
        atomic_json(output/'status.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='03 adaptive 결과 폴더; 생략 시 inputs 안의 유일한 입력 사용')
    parser.add_argument('--output',type=Path,help='생략 시 outputs/<입력 폴더 이름>_entities')
    parser.add_argument('--config',type=Path,default=ROOT/'config.default.json',help='검출·추적·매칭 설정 JSON')
    parser.add_argument('--mode',choices=['both','with_qwen','without_qwen'],default='both')
    parser.add_argument('--detector',help='설정 파일의 detector.model을 덮어쓸 모델 이름/경로')
    parser.add_argument('--qwen-max-calls',type=int,choices=range(1,9),default=8,help='실패/재시도 포함 실제 Qwen 생성 절대 상한 (기본 8)')
    parser.add_argument('--batch-shots',type=int,default=4)
    args=parser.parse_args()
    if args.batch_shots<1:parser.error('batch-shots must be positive')
    try:
        args.policy=load_config(args.config.expanduser())
        if args.detector:args.policy=replace(args.policy,detector=replace(args.policy.detector,model=args.detector))
        args.input=select_input(args.input,ROOT/'inputs')
        args.output=(args.output.expanduser() if args.output else ROOT/'outputs'/f'{args.input.name}_entities').resolve()
    except (OSError,ValueError,TypeError) as exc:parser.error(str(exc))
    args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    with (args.output/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.timing=RunTiming(args.output,started=START,started_at=STARTED,mode=args.mode)
        completed_status='complete'
        try:
            if args.mode=='both':
                with args.timing.stage('with_qwen','Qwen ON 전체',kind='mode'):on=run_mode(args,'with_qwen',args.output/'with_qwen')
                with args.timing.stage('without_qwen','Qwen OFF 전체',kind='mode'):off=run_mode(args,'without_qwen',args.output/'without_qwen',args.output/'with_qwen')
                with args.timing.stage('comparison','ON/OFF 비교 영상'):compare_results(args.output,on,off)
                if on.get('warnings'):completed_status='complete_with_warnings'
            else:
                with args.timing.stage(args.mode,args.mode,kind='mode'):single=run_mode(args,args.mode,args.output)
                if single.get('warnings'):completed_status='complete_with_warnings'
                write_run_reports(args.output,args.mode)
        except BaseException as exc:
            args.timing.finish('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
        else:
            args.timing.finish(completed_status)
            atomic_json(ROOT/'outputs/latest_run.json',dict(output=str(args.output),input=str(args.input),completed_at=utc_now()))


if __name__=='__main__':main()
