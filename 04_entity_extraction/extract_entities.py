#!/usr/bin/env python3
"""Entity extraction on stage-03 samples with stage-01 perception and identity modules."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gc
import hashlib
from itertools import combinations_with_replacement
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from run_timing import RunTiming, format_duration, utc_now

PROCESS_STARTED=time.perf_counter()
PROCESS_STARTED_AT=utc_now()

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'01_low_freq_event'),str(ROOT.parent/'03_adaptive_sampling')]

import av
import cv2
import numpy as np
from PIL import Image
import torch

from low_freq_event.config import Settings
from low_freq_event.backend import QwenBackend
from low_freq_event.detection import ObjectEncoder, associate_frame, reconcile_tracklets, select_references
from low_freq_event.identity import canonical
from low_freq_event.motion import camera_motion
from low_freq_event.storage import atomic_json, read_json, file_fingerprint
from entity_models import DETECT_PROMPT, MergeReview, detect_batch, observations
from entity_output import export_results
from identity_review import ReviewBackend, review_identities
from temporal_identity import review_temporal_identities, review_retrieved_candidates, split_shot_seeds, TEMPORAL_VERSION
from sequential_identity import run_sequential, VERSION as SEQUENTIAL_VERSION

LOG=logging.getLogger('entities')


def prepare_inputs(source,output):
    summary=read_json(source/'summary.json')
    frames=[]; shots=[]; total=0; fps=None; size=None; fingerprints=[]
    for row in summary['videos']:
        metadata=source/row['samples']; meta=read_json(metadata)
        path=metadata.parent/meta['output_video']
        fingerprints.append(dict(video=file_fingerprint(path),metadata=file_fingerprint(metadata)))
        rate=Fraction(meta['source_fps'])
        if fps is None: fps=rate
        if fps!=rate: raise ValueError('전체 시각화에는 모든 샷의 원본 FPS가 같아야 합니다.')
        samples=meta['samples']; count=int(meta['source_frames'])
        if not samples or samples[0]['source_frame']!=0:
            raise ValueError('샘플은 첫 프레임부터 시작해야 합니다.')
        shot_frames=[]
        for i,sample in enumerate(samples):
            start=sample['source_frame']; end=sample['hold_until_frame']
            expected=samples[i+1]['source_frame'] if i+1<len(samples) else count
            if end!=expected or end<=start: raise ValueError('샘플 유지 구간에 간격/중복이 있습니다.')
            key=f'{row["id"]}_f{start:08d}'
            f=dict(sample,frame_key=key,shot_id=row['id'],full_frame=total+start,
                full_time_sec=float((total+start)/fps),hold_frames=end-start,
                image=f'frames/{key}.jpg',original_video=meta['original_video'])
            shot_frames.append(f)
        with av.open(str(path)) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=2
            if stream.duration*stream.time_base!=count/fps:
                raise ValueError(f'샷 길이가 samples.json과 다릅니다: {path}')
            decoded=0
            for i,frame in enumerate(container.decode(stream)):
                if i>=len(samples) or frame.pts*frame.time_base!=samples[i]['source_frame']/fps:
                    raise ValueError(f'VFR 시각이 samples.json과 다릅니다: {path}')
                if size is None: size=(frame.width,frame.height)
                if size!=(frame.width,frame.height): raise ValueError('입력 샷 해상도가 서로 다릅니다.')
                destination=output/shot_frames[i]['image']
                if not destination.exists():
                    destination.parent.mkdir(exist_ok=True)
                    frame.to_image().convert('RGB').save(destination,quality=96)
                decoded+=1
            if decoded!=len(samples): raise ValueError('입력 영상의 샘플 수가 다릅니다.')
        frames.extend(shot_frames)
        shots.append(dict(id=row['id'],full_start_frame=total,source_frames=count,
            full_start_sec=float(total/fps),duration_sec=float(count/fps),
            original_start_frame=meta['original_start_frame'],original_video=meta['original_video']))
        total+=count
    if not frames: raise ValueError('입력 샘플이 없습니다.')
    return frames,dict(fps=str(fps),size=list(size),total_frames=total,duration_sec=float(total/fps),
                      shots=shots,input_fingerprints=fingerprints)


def yolo_proposals(frames,output,settings):
    from ultralytics import YOLO
    from low_freq_event.perception import model_files
    path=output/'yolo_proposals.json'
    if path.exists():
        LOG.info('[CACHE HIT] %s / YOLO 후보 재사용 — 모델 추론 생략', output.name)
        return read_json(path)
    LOG.info('[CACHE MISS] %s / YOLO 신규 추론: %d frames', output.name, len(frames))
    model=YOLO(str(model_files()/'yolo11s.pt'))
    proposals={}
    for i,f in enumerate(frames):
        image=Image.open(output/f['image']).convert('RGB')
        pred=model.predict(image,conf=min(settings.person_confidence,settings.object_confidence),
            imgsz=settings.detector_size,iou=.45,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),verbose=False)[0]
        rows=[]
        for box,conf,cls in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),pred.boxes.cls.cpu().numpy()):
            if conf<(settings.person_confidence if int(cls)==0 else settings.object_confidence): continue
            rows.append(dict(category=model.names[int(cls)],bbox=(box/np.array([image.width,image.height]*2)*1000).tolist(),confidence=float(conf)))
        proposals[f['frame_key']]=rows
        if i%25==0: LOG.info('YOLO proposals %d/%d',i+1,len(frames))
    atomic_json(path,proposals)
    del model;gc.collect();torch.cuda.empty_cache()
    return proposals


def ground_frames(frames,output,backend,proposals,batch_size):
    for offset in range(0,len(frames),batch_size):
        batch=[f for f in frames[offset:offset+batch_size] if not (output/'grounding'/f'{f["frame_key"]}.json').exists()]
        if not batch: continue
        images=[Image.open(output/f['image']).convert('RGB') for f in batch]
        LOG.info('Qwen foreground grounding %d-%d/%d',offset+1,min(offset+batch_size,len(frames)),len(frames))
        for result in detect_batch(backend,batch,images,proposals):
            atomic_json(output/'grounding'/f'{result["frame_key"]}.json',result)


def ground_without_qwen(frames,output,proposals):
    LOG.info('YOLO 후보를 OFF 검출 형식으로 변환합니다 (추론 없음).')
    animal_classes={'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe'}
    for f in frames:
        rows=[]
        for p in proposals[f['frame_key']]:
            kind='person' if p['category']=='person' else ('animal' if p['category'] in animal_classes else 'object')
            rows.append(dict(kind=kind,category=p['category'],label=p['category'],
                appearance=f'YOLO11s class: {p["category"]}',bbox=p['bbox'],
                detection_confidence=p['confidence'],uncertain=False))
        atomic_json(output/'grounding'/f'{f["frame_key"]}.json',dict(frame_key=f['frame_key'],detections=rows,source='YOLO11s'))


def extract_features(frames,output,settings,proposals,*,use_qwen=True):
    missing=[f for f in frames if not (output/'features'/f'{f["frame_key"]}.json').exists()]
    LOG.info('외형 특징: %d개 캐시 재사용 / %d개 신규 추출', len(frames)-len(missing), len(missing))
    if not missing: return
    encoder=ObjectEncoder(settings)
    for i,f in enumerate(missing):
        image=Image.open(output/f['image']).convert('RGB')
        detections=read_json(output/'grounding'/f'{f["frame_key"]}.json')['detections']
        raw=observations(encoder,image,f,detections,proposals[f['frame_key']],output,use_qwen=use_qwen)
        atomic_json(output/'features'/f'{f["frame_key"]}.json',raw)
        if i%20==0: LOG.info('Appearance features %d/%d',i+1,len(missing))
    del encoder;gc.collect();torch.cuda.empty_cache()


def associate(frames,output,settings):
    registry={};visual={}; previous=None; raw_all=[]
    for f in frames:
        raw=read_json(output/'features'/f'{f["frame_key"]}.json');motion=None
        if previous and previous['shot_id']==f['shot_id']:
            motion=camera_motion(output/previous['image'],output/f['image'],
                previous['boxes'],[o['bbox'] for o in raw if o['kind']=='person'])
            motion['previous_time_sec']=previous['original_time_sec']
        associate_frame(raw,registry,visual,settings,motion=motion)
        raw_all.extend(raw)
        previous=dict(f,boxes=[o['bbox'] for o in raw if o['kind']=='person'])
    if raw_all: reconcile_tracklets(registry,visual,settings)
    return registry,visual,raw_all



def seed_cache(seed,output,config,timeline,*,qwen):
    if seed is None: return
    seed=Path(seed).resolve()
    if seed==output: raise ValueError('재사용 원본과 출력 폴더는 달라야 합니다.')
    old=read_json(seed/'run_config.json')
    for key in ('input','settings'):
        if old.get(key)!=config[key]: raise ValueError(f'재사용 캐시의 {key}가 현재 입력/설정과 다릅니다.')
    if read_json(seed/'timeline.json')!=timeline:
        raise ValueError('재사용 캐시의 입력 영상 해시/타임라인이 다릅니다.')
    names=['yolo_proposals.json']
    if qwen:
        if old.get('mode','with_qwen')!='with_qwen': raise ValueError('Qwen 결과 캐시가 아닙니다.')
        for key in ('prompt_sha256','batch_size'):
            if old.get(key)!=config[key]: raise ValueError(f'Qwen 캐시의 {key}가 다릅니다.')
        names+=['grounding','features','crops','request_cache','traces','review_traces','workers']
    for name in names:
        source=seed/name;destination=output/name
        if not source.exists() or destination.exists(): continue
        if source.is_dir(): shutil.copytree(source,destination)
        else: shutil.copy2(source,destination)
    atomic_json(output/'cache_provenance.json',dict(source=str(seed),copied_categories=names,
        qwen_derived_cache_allowed=qwen,source_run_config=old))


def run_one(args,mode,output,seed=None):
    mode_started=time.perf_counter()
    def step(name,label,operation,*parameters,**options):
        with args.timing.stage(f'{mode}.{name}',f'{mode} / {label}'):
            return operation(*parameters,**options)
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    settings=Settings(perception_device='auto',max_pixels=196608)
    # Bound CPU thread pools while neural models run on the selected GPU.
    torch.set_num_threads(4);cv2.setNumThreads(2)
    source=args.input.resolve()
    use_qwen=mode=='with_qwen'
    config=dict(input=str(source),settings=settings.model_dump(),batch_size=args.batch_size,
        prompt_sha256=hashlib.sha256(DETECT_PROMPT.encode()).hexdigest() if use_qwen else None,
        pipeline_version=3,mode=mode,qwen_detection=use_qwen,qwen_identity_review=use_qwen,
        temporal_identity_version=TEMPORAL_VERSION if use_qwen and args.qwen_identity=='temporal' else None)
    if use_qwen and args.qwen_identity=='sequential':
        config.update(sequential_identity_version=SEQUENTIAL_VERSION)
    manifest=output/'run_config.json'
    if manifest.exists() and read_json(manifest)!=config:
        raise ValueError('기존 결과 설정과 다릅니다. 새 --output 폴더를 지정하세요.')
    atomic_json(manifest,config)
    frames,timeline=step('inputs','입력 디코딩·타임라인 준비',prepare_inputs,source,output)
    old_timeline=output/'timeline.json'
    if old_timeline.exists() and read_json(old_timeline)!=timeline:
        raise ValueError('입력 영상/샘플 메타데이터가 변경되었습니다. 새 결과 폴더를 사용하세요.')
    atomic_json(old_timeline,timeline);atomic_json(output/'samples.json',frames)
    step('cache','캐시 준비',seed_cache,seed,output,config,timeline,qwen=use_qwen)
    started=time.perf_counter()
    atomic_json(output/'status.json',dict(status='running',samples=len(frames)))
    try:
        proposals=step('yolo','YOLO 검출 후보',yolo_proposals,frames,output,settings)
        if use_qwen:
            backend=QwenBackend(settings,output/'traces')
            step('grounding','Qwen 검출 보완',ground_frames,frames,output,backend,proposals,args.batch_size)
            if backend.model is not None:
                del backend.model;backend.model=None;gc.collect();torch.cuda.empty_cache()
        else:
            step('grounding','YOLO 후보 형식 변환 (추론 없음)',ground_without_qwen,frames,output,proposals)
        step('features','외형 특징 추출',extract_features,frames,output,settings,proposals,use_qwen=use_qwen)
        if use_qwen and args.qwen_identity=='sequential':
            raw_all=[o for f in frames for o in read_json(output/'features'/f'{f["frame_key"]}.json')]
            review_backend=ReviewBackend(settings,output/'review_traces')
            registry,visual,raw_all,joins=step('sequential','정방향 누적·독립 역방향·최종 검토',
                run_sequential,raw_all,frames,timeline,output,review_backend,settings,proposals)
            if review_backend.model is not None:
                del review_backend.model;gc.collect();torch.cuda.empty_cache()
        else:
            registry,visual,raw_all=step('association','외형 기반 ID 연결',associate,frames,output,settings)
        if use_qwen and args.qwen_identity=='temporal':
            registry,visual=step('shot_seeds','샷별 초기 ID 분리',split_shot_seeds,registry,visual,raw_all,output)
        if not (use_qwen and args.qwen_identity=='sequential'):
            atomic_json(output/'identity_before_review.json',dict(registry=registry,assignments=visual.get('assignments',{})))
            joins=[]
        if use_qwen and args.qwen_identity!='sequential':
            review_backend=ReviewBackend(settings,output/'review_traces')
            joins=step('gallery','crop 갤러리 검토',review_identities,registry,visual,raw_all,output,review_backend,
                cross_shot=args.qwen_identity=='gallery')
            if args.qwen_identity=='temporal':
                joins+=step('temporal','시간순 장면 매칭',review_temporal_identities,registry,visual,raw_all,frames,timeline,output,review_backend)
                joins+=step('retrieval','추가 후보 검색·재판정',review_retrieved_candidates,registry,visual,raw_all,frames,timeline,output,review_backend)
            if review_backend.model is not None:
                del review_backend.model;gc.collect();torch.cuda.empty_cache()
        atomic_json(output/'identity_review.json',dict(qwen_enabled=use_qwen,joins=joins,aliases={k:canonical(registry,k) for k in registry}))
        atomic_json(output/'identity_state.json',dict(registry=registry,assignments=visual.get('assignments',{})))
        result=step('export','카탈로그·전체 영상 저장',export_results,output,frames,timeline,raw_all,registry,visual,mode=mode)
        result.update(status='complete',elapsed_sec=time.perf_counter()-mode_started,
            processing_elapsed_sec=time.perf_counter()-started,
            elapsed_scope='full mode including input and cache preparation')
        result['elapsed_human']=format_duration(result['elapsed_sec'])
        atomic_json(output/'status.json',result)
        LOG.info('완료: %s',json.dumps(result,ensure_ascii=False))
        return result
    except BaseException as exc:
        atomic_json(output/'status.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc)))
        raise


def run(args):
    if args.mode!='both':
        with args.timing.stage(args.mode,args.mode+' 전체',kind='mode'):
            result=run_one(args,args.mode,args.output,args.reuse_qwen if args.mode=='with_qwen' else args.reuse_yolo)
        from result_report import write_run_reports
        with args.timing.stage('result_validation','등장 행렬·ID·출력 일관성 검사'):
            write_run_reports(args.output,args.mode)
        return result
    with args.timing.stage('with_qwen','Qwen ON 전체',kind='mode'):
        with_result=run_one(args,'with_qwen',args.output/'with_qwen',args.reuse_qwen)
    with args.timing.stage('without_qwen','Qwen OFF 전체',kind='mode'):
        without_result=run_one(args,'without_qwen',args.output/'without_qwen',args.output/'with_qwen')
    from compare_results import compare_results
    with args.timing.stage('comparison','OFF/ON 비교 영상 생성',kind='comparison'):
        compare_results(args.output,with_result,without_result)


def main():
    parser=argparse.ArgumentParser(description='Adaptive samples → entity catalog and annotated full VFR video')
    parser.add_argument('--input',type=Path,default=ROOT/'inputs/sintel_adaptive')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/sintel_sequential')
    parser.add_argument('--mode',choices=['both','with_qwen','without_qwen'],default='both')
    parser.add_argument('--reuse-qwen',type=Path,help='같은 입력/설정의 기존 Qwen 검출·특징·응답 캐시 재사용')
    parser.add_argument('--reuse-yolo',type=Path,help='without_qwen 단독 실행 시 YOLO 후보만 재사용')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--qwen-identity',choices=['gallery','temporal','sequential'],default='sequential',
        help='sequential: 전체 과거 entity 기억 → 독립 역방향 → 최종 재검토 (기본)')
    args=parser.parse_args()
    if not 1<=args.batch_size<=8: parser.error('--batch-size must be 1..8')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    # Prevent two concurrent invocations from writing the same output cache.
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.timing=RunTiming(args.output,started=PROCESS_STARTED,started_at=PROCESS_STARTED_AT,
            mode=args.mode,reuse_qwen=args.reuse_qwen,reuse_yolo=args.reuse_yolo)
        try:
            run(args)
        except BaseException as exc:
            args.timing.finish('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
            raise
        else:
            args.timing.finish('complete')


if __name__=='__main__': main()
