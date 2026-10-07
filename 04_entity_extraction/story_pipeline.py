#!/usr/bin/env python3
"""Original-shot tracking -> local cast -> global cast -> revised shot narrative."""
from __future__ import annotations
import argparse
import fcntl
import gc
import json
import logging
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
from low_freq_event.config import Settings
from low_freq_event.storage import atomic_json,read_json
from run_timing import utc_now,format_duration
from shot_tracking import build_dense_tracks
from story_semantics import StoryBackend,review_shots,resolve_global,revise_narrative,apply_global_rejections
from story_export import export_story

LOG=logging.getLogger('story')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--detector',default='yolo11x.pt')
    parser.add_argument('--qwen-max-calls',type=int,default=24)
    parser.add_argument('--model-id',default='Qwen/Qwen3-VL-32B-Instruct')
    parser.add_argument('--model-revision',default='0cfaf48183f594c314753d30a4c4974bc75f3ccb')
    parser.add_argument('--global-model-id',default='Qwen/Qwen3-VL-32B-Instruct',
                        help='Stronger model used only for the two global identity reviews')
    parser.add_argument('--global-model-revision',default='0cfaf48183f594c314753d30a4c4974bc75f3ccb')
    parser.add_argument('--skip-mask-refinement',action='store_true')
    parser.add_argument('--stage',choices=['all','tracking','geometry','local','mask','global','narrative','export'],default='all',
                        help='Later stages reuse completed earlier stages; all resumes cached model requests.')
    parser.add_argument('--evaluate-against',type=Path,help='Optional old result for independent manual-label evaluation')
    parser.add_argument('--evaluation-labels',type=Path,help='Independent partial labels; never used by inference')
    args=parser.parse_args()
    if args.qwen_max_calls<1:parser.error('qwen-max-calls must be positive')
    if args.input is None:
        candidates=[p for p in (ROOT/'inputs').iterdir() if p.is_dir() and (p/'summary.json').is_file()]
        if len(candidates)!=1:parser.error('Specify --input when inputs has zero or multiple videos.')
        args.input=candidates[0]
    args.input=args.input.expanduser().resolve()
    if bool(args.evaluate_against)!=bool(args.evaluation_labels):
        parser.error('Use --evaluate-against and --evaluation-labels together')
    if args.output is None and (ROOT/'outputs/latest_run.json').is_file():
        latest=read_json(ROOT/'outputs/latest_run.json'); previous=Path(latest['output'])
        if latest.get('input')==str(args.input) and (previous/'status.json').is_file():
            if read_json(previous/'status.json').get('pipeline')=='shot_story':args.output=previous
    output=(args.output or ROOT/'outputs'/f'{args.input.name}_story').expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(output/'run.log',encoding='utf-8')])
    import cv2,torch
    torch.set_num_threads(4);cv2.setNumThreads(2)
    started=time.perf_counter();started_at=utc_now();steps=[]
    settings=Settings(model_id=args.model_id,model_revision=args.model_revision,json_retries=1,
                      max_new_tokens=6500,reference_pixels=196608)
    history=read_json(output/'execution_history.json') if (output/'execution_history.json').exists() else {'invocations':[]}
    invocation=dict(started_at=started_at,stage=args.stage,model_id=args.model_id,model_revision=args.model_revision,
                    qwen_limit=args.qwen_max_calls,stages=steps,status='running')
    history['invocations'].append(invocation)
    atomic_json(output/'execution_history.json',history)
    atomic_json(output/'run_config.json',dict(input=str(args.input),output=str(output),detector=args.detector,
                qwen_max_calls=args.qwen_max_calls,settings=settings.model_dump(),mask_refinement=not args.skip_mask_refinement,
                global_model_id=args.global_model_id,global_model_revision=args.global_model_revision))
    def timed(name,fn,*a,**kw):
        before=time.perf_counter();LOG.info('STAGE %s',name)
        result=fn(*a,**kw)
        steps.append(dict(stage=name,elapsed_sec=time.perf_counter()-before));
        atomic_json(output/'stage_timing.json',dict(started_at=started_at,stages=steps))
        atomic_json(output/'execution_history.json',history)
        return result
    def finish_stage(name):
        atomic_json(output/'status.json',dict(status='stage_complete',pipeline='shot_story',stage=name,
                    started_at=started_at,completed_at=utc_now(),elapsed_sec=time.perf_counter()-started))
    with (output/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(output/'status.json',dict(status='running',pipeline='shot_story',started_at=started_at))
        backend=None
        try:
            if args.stage in ('all','tracking'):
                bundle=timed('dense_tracking',build_dense_tracks,args.input,output,model=args.detector,device='cuda:0')
            else:bundle=read_json(output/'dense_tracking/bundle.json')
            atomic_json(output/'tracking_summary.json',dict(stats=bundle['stats'],
                shots=[dict(shot_id=s['shot_id'],frames=s['frame_count'],tracks=len(s['tracks']),stats=s.get('stats')) for s in bundle['shots']],
                config=bundle.get('config'),source='original_dense_frames'))
            if args.stage=='tracking':finish_stage('tracking');return
            if not args.skip_mask_refinement:
                from mask_track_graph import build_mask_track_graph,encode_masked_representatives
                if args.stage in ('all','geometry'):
                    bundle=timed('foreground_tracking',build_mask_track_graph,bundle,output,device='cuda:0')
                    bundle=timed('masked_appearance',encode_masked_representatives,bundle,output,device='cuda:0')
                elif (output/'mask_track_graph/bundle.json').exists():
                    bundle=read_json(output/'mask_track_graph/bundle.json')
                elif args.stage=='local':
                    raise RuntimeError('Run --stage geometry first, or use --stage all for the complete pipeline.')
            if args.stage=='geometry':finish_stage('geometry');return
            backend=StoryBackend(settings,output,args.qwen_max_calls)
            if args.stage in ('all','local'):
                local=timed('shot_review',review_shots,bundle,output,backend)
                if local['failures'] or len(local['shots'])!=len(bundle['shots']):
                    raise RuntimeError('Some shots lack valid semantic review; see local_entities.json. No partial run is promoted.')
                atomic_json(output/'reviewed_tracking.json',bundle)
            else:
                local=read_json(output/'local_entities.json')
                if (output/'reviewed_tracking.json').exists():bundle=read_json(output/'reviewed_tracking.json')
            if args.stage=='local':finish_stage('local');return
            if not args.skip_mask_refinement:
                if args.stage in ('all','mask'):
                    if backend.model is not None:
                        del backend.model;backend.model=None;gc.collect();torch.cuda.empty_cache()
                    from mask_refinement import refine_local_tracks
                    bundle,local,mask_summary=timed('mask_refinement',refine_local_tracks,bundle,local,output,device='cuda:0')
                    bundle=timed('refined_appearance',encode_masked_representatives,bundle,output,device='cuda:0',
                                 save_path=output/'mask_refinement/bundle.json')
                elif (output/'mask_refinement/bundle.json').exists():
                    bundle=read_json(output/'mask_refinement/bundle.json')
                    local=read_json(output/'mask_refinement/local.json')
            if args.stage=='mask':finish_stage('mask');return
            if args.stage in ('all','global'):
                if backend.model is not None:
                    del backend.model;backend.model=None;gc.collect();torch.cuda.empty_cache()
                global_settings=settings.model_copy(update=dict(model_id=args.global_model_id,model_revision=args.global_model_revision))
                backend=StoryBackend(global_settings,output,args.qwen_max_calls)
                global_result=timed('global_resolution',resolve_global,bundle,local,output,backend)
            else:global_result=read_json(output/'global_entities.json')
            local=apply_global_rejections(local,global_result,output)
            if args.stage=='global':finish_stage('global');return
            if args.stage in ('all','narrative'):
                if backend.settings.model_id!=settings.model_id:
                    if backend.model is not None:
                        del backend.model;backend.model=None;gc.collect();torch.cuda.empty_cache()
                    backend=StoryBackend(settings,output,args.qwen_max_calls)
                global_result=timed('narrative_revision',revise_narrative,bundle,local,global_result,output,backend)
            if args.stage=='narrative':finish_stage('narrative');return
            if backend.model is not None:
                del backend.model;backend.model=None;gc.collect();torch.cuda.empty_cache()
            exported=timed('export',export_story,bundle,local,global_result,output,args.input)
            issues=[]
            validation=exported['validation']
            if validation['simultaneous_identity_conflicts']:issues.append(f"동일 ID의 서로 다른 동시 박스 {validation['simultaneous_identity_conflicts']}건")
            if validation['event_reference_errors']:issues.append('사건의 Entity 참조 또는 시각 오류')
            if validation['coverage']['unreviewed_tracks']:issues.append('미검토 트랙이 남아 있습니다.')
            if validation.get('empty_entities') or validation.get('missing_representatives'):issues.append('관측 또는 대표 이미지가 없는 Entity가 있습니다.')
            if global_result.get('errors'):issues.append('일부 전역 검토 실패: global_entities.json 확인')
            if global_result.get('narrative_failures'):issues.append('일부 샷의 의미 요약 보정 실패')
            missing=[dict(shot_id=s['shot_id'],objects=s['missing_objects']) for s in local['shots'] if s.get('missing_objects')]
            if missing:issues.append(f'{len(missing)}개 샷에 VLM이 지적한 검출 누락 후보가 남아 있습니다.')
            uncertain=[e['entity_id'] for e in global_result['entities'] if e.get('uncertain')]
            if uncertain:issues.append(f'{len(uncertain)}개 전역 Entity의 동일성/분류가 불확실합니다.')
            quality=dict(status='needs_review' if issues else 'awaiting_evaluation',
                success=False,issues=issues,missing_objects=missing,uncertain_entities=uncertain,
                export=exported,manual_evaluation=None,
                note='실행 완료와 실제 식별 성공을 구분합니다. 독립 평가와 영상 검토가 필요합니다.')
            if args.evaluate_against:
                from story_evaluation import evaluate_partial_identities
                baseline=args.evaluate_against/'observations.json' if args.evaluate_against.is_dir() else args.evaluate_against
                quality['manual_evaluation']=timed('partial_label_evaluation',evaluate_partial_identities,
                    baseline,output/'observations.json',args.evaluation_labels,output=output/'manual_identity_evaluation.json')
            atomic_json(output/'quality.json',quality)
            elapsed=time.perf_counter()-started
            status=dict(status='complete_with_warnings' if issues else 'complete',pipeline='shot_story',
                started_at=started_at,completed_at=utc_now(),elapsed_sec=elapsed,elapsed_human=format_duration(elapsed),
                warnings=issues,qwen_generations=backend.ledger['generations'],qwen_limit=args.qwen_max_calls,
                sampled_frames=exported.get('sampled_frames'),entity_count=len(global_result['entities']))
            status['model_id']=args.model_id;status['model_revision']=args.model_revision
            status['global_model_id']=args.global_model_id;status['global_model_revision']=args.global_model_revision
            atomic_json(output/'status.json',status)
            atomic_json(output/'run_timing.json',dict(**status,stages=steps))
            from story_report import write_report
            write_report(output)
            atomic_json(ROOT/'outputs/latest_run.json',dict(output=str(output),input=str(args.input),completed_at=utc_now()))
            LOG.info('Complete: %s, %s, %d entities, Qwen %d/%d',output,status['elapsed_human'],len(global_result['entities']),backend.ledger['generations'],args.qwen_max_calls)
        except BaseException as exc:
            atomic_json(output/'status.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                pipeline='shot_story',error=str(exc),started_at=started_at,elapsed_sec=time.perf_counter()-started))
            LOG.exception('Pipeline did not complete')
            raise
        finally:
            invocation.update(status=read_json(output/'status.json').get('status','unknown'),
                              ended_at=utc_now(),elapsed_sec=time.perf_counter()-started)
            atomic_json(output/'execution_history.json',history)


if __name__=='__main__':main()
