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
from story_semantics import StoryBackend,review_shots,resolve_global,revise_narrative,apply_global_rejections,global_input_fingerprint
from story_semantics import annotate_identity_scopes
from story_export import export_story

LOG=logging.getLogger('story')
THINKING_MODEL='Qwen/Qwen3-VL-8B-Thinking'
THINKING_REVISION='92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b'


def default_global_model():
    """Use the pinned converted checkpoint when installed, otherwise the hub."""
    checkpoint=ROOT/'models/Qwen3-VL-8B-Thinking-NF4'
    provenance=checkpoint/'source_provenance.json'
    if provenance.is_file() and (checkpoint/'config.json').is_file():
        source=read_json(provenance)
        if (source.get('source_model'),source.get('source_revision'))==(THINKING_MODEL,THINKING_REVISION):
            return str(checkpoint)
    return THINKING_MODEL


def extend_generation_budget(output, limit, reason):
    """Explicit operator action, preserving every prior generation and limit."""
    path=output/'qwen_budget.json'
    if not path.is_file():raise ValueError('Budget extension requires an existing run ledger')
    ledger=read_json(path)
    if limit<=ledger['limit']:raise ValueError('The extended limit must exceed the current limit')
    if not reason or not reason.strip():raise ValueError('Budget extension requires a reason')
    ledger.setdefault('budget_changes',[]).append(dict(previous_limit=ledger['limit'],new_limit=limit,
        generations_so_far=ledger['generations'],reason=reason.strip(),changed_at=utc_now()))
    ledger['limit']=limit
    atomic_json(path,ledger)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--detector')
    parser.add_argument('--analysis-frames', choices=['sampled','dense'], default='sampled',
                        help='Default: only adaptive selected images; dense is explicit legacy mode')
    parser.add_argument('--qwen-max-calls',type=int)
    parser.add_argument('--extend-qwen-budget-to',type=int,
                        help='Explicitly raise an existing ledger limit, preserving all prior usage')
    parser.add_argument('--budget-extension-reason',help='Required audit reason for an explicit budget extension')
    parser.add_argument('--model-id')
    parser.add_argument('--model-revision')
    parser.add_argument('--global-model-id',help='Defaults to pinned Thinking; compact catalog requests use bounded reasoning')
    parser.add_argument('--global-model-revision')
    parser.add_argument('--global-strategy',choices=['linked','batched','compact','pairwise'],help='Preserve focused cross-shot links and repair physical collisions (linked, default); compact is experimental')
    parser.add_argument('--global-review-max-calls',type=int,
                        help='Maximum new focused identity generations, including retries; cached replies are free')
    parser.add_argument('--global-initialization',choices=['local','whole'],
                        help='Start from local singleton hypotheses (default) or two whole-catalog VLM partitions')
    parser.add_argument('--context',type=Path,help='Optional source-backed video context JSON (weak prior only)')
    parser.add_argument('--original-video',type=Path,help='Original video used to recover relocated/missing shot media')
    masks=parser.add_mutually_exclusive_group()
    masks.add_argument('--skip-mask-refinement',action='store_true',default=None)
    masks.add_argument('--mask-refinement',dest='skip_mask_refinement',action='store_false')
    parser.add_argument('--stage',choices=['all','tracking','geometry','local','mask','global','narrative','export'],default='all',
                        help='Later stages reuse completed earlier stages; all resumes cached model requests.')
    parser.add_argument('--evaluate-against',type=Path,help='Optional old result for independent manual-label evaluation')
    parser.add_argument('--evaluation-labels',type=Path,help='Independent partial labels; never used by inference')
    args=parser.parse_args()
    from video_context import load_video_context
    context=load_video_context(args.context)
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
            if (read_json(previous/'status.json').get('pipeline')=='shot_story'
                    and (previous/'run_config.json').is_file()
                    and read_json(previous/'run_config.json').get('analysis_frames','dense')==args.analysis_frames):args.output=previous
    output=(args.output or ROOT/'outputs'/f'{args.input.name}_story_{args.analysis_frames}').expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    previous_config=read_json(output/'run_config.json') if (output/'run_config.json').is_file() else {}
    if previous_config.get('input') and previous_config['input']!=str(args.input):
        parser.error('This output belongs to a different input; choose a new --output')
    if previous_config and previous_config.get('analysis_frames','dense') != args.analysis_frames:
        parser.error('Analysis frame mode differs from existing output; use a fresh --output folder')
    if args.analysis_frames=='sampled' and args.skip_mask_refinement is False:
        parser.error('SAM video refinement reads dense frames; unavailable in sampled mode. Omit --mask-refinement.')
    if args.analysis_frames=='sampled':args.skip_mask_refinement=True
    args.global_strategy=args.global_strategy or previous_config.get('global_strategy','pairwise' if previous_config else 'linked')
    previous_settings=previous_config.get('settings',{})
    args.detector=args.detector or previous_config.get('detector') or 'yolo11x.pt'
    explicit_model=args.model_id
    args.model_id=args.model_id or previous_settings.get('model_id','Qwen/Qwen3-VL-8B-Instruct')
    default_revisions={'Qwen/Qwen3-VL-8B-Instruct':'0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                       'Qwen/Qwen3-VL-4B-Instruct':'ebb281ec70b05090aa6165b016eac8ec08e71b17',
                       THINKING_MODEL:THINKING_REVISION,
                       str(ROOT/'models/Qwen3-VL-8B-Thinking-NF4'):THINKING_REVISION}
    if args.model_revision is None:
        args.model_revision=(previous_settings.get('model_revision') if args.model_id==previous_settings.get('model_id')
                             else default_revisions.get(args.model_id))
    if not args.model_revision:parser.error('Specify --model-revision for the selected model')
    args.global_model_id=args.global_model_id or (None if explicit_model else previous_config.get('global_model_id')) or (
        args.model_id if explicit_model or previous_config else default_global_model())
    args.global_model_revision=args.global_model_revision or (
        previous_config.get('global_model_revision') if args.global_model_id==previous_config.get('global_model_id')
        else args.model_revision if args.global_model_id==args.model_id else default_revisions.get(args.global_model_id))
    if not args.global_model_revision:parser.error('Specify --global-model-revision for the selected global model')
    if args.qwen_max_calls is None:
        ledger=read_json(output/'qwen_budget.json') if (output/'qwen_budget.json').is_file() else {}
        args.qwen_max_calls=ledger.get('limit',previous_config.get('qwen_max_calls',320 if args.global_strategy=='linked' else 80))
    if args.extend_qwen_budget_to is not None:
        if not args.budget_extension_reason or not args.budget_extension_reason.strip():
            parser.error('--extend-qwen-budget-to requires --budget-extension-reason')
        if args.extend_qwen_budget_to<=args.qwen_max_calls:
            parser.error('The extended budget must exceed the current budget')
        args.qwen_max_calls=args.extend_qwen_budget_to
    elif args.budget_extension_reason:
        parser.error('--budget-extension-reason requires --extend-qwen-budget-to')
    if args.global_review_max_calls is None:args.global_review_max_calls=previous_config.get('global_review_max_calls',10 if args.global_strategy=='batched' else (5 if args.global_strategy=='compact' else (96 if args.global_strategy=='linked' else 32)))
    if args.global_initialization is None:args.global_initialization=previous_config.get('global_initialization','local')
    if previous_config:
        if (args.detector!=previous_config.get('detector',args.detector)
                and args.stage not in ('all','tracking')):
            parser.error('Detector changed: rerun --stage all (or tracking first) before reusing downstream artifacts')
        local_model_changed=any(previous_settings.get(key,args_value)!=args_value for key,args_value in
                                (('model_id',args.model_id),('model_revision',args.model_revision)))
        if local_model_changed and args.stage not in ('all','local'):
            parser.error('Local review model changed: rerun --stage all or local; use --global-model-id for a global-only change')
        global_model_changed=any(previous_config.get(key,args_value)!=args_value for key,args_value in
                                 (('global_model_id',args.global_model_id),('global_model_revision',args.global_model_revision)))
        if global_model_changed and args.stage not in ('all','global'):
            parser.error('Global review model changed: rerun --stage global before narrative/export')
        if (args.global_initialization!=previous_config.get('global_initialization',args.global_initialization)
                and args.stage in ('narrative','export')):
            parser.error('Global initialization changed: rerun --stage global before narrative/export')
    if args.qwen_max_calls<1:parser.error('qwen-max-calls must be positive')
    if args.global_strategy=='compact' and args.global_initialization!='local':parser.error('Compact strategy requires --global-initialization local')
    if previous_config and args.stage in ('narrative','export') and args.global_strategy!=previous_config.get('global_strategy','pairwise'):parser.error('Strategy changed; rerun --stage global')
    if args.global_review_max_calls<0:parser.error('global-review-max-calls must be nonnegative')
    if args.original_video is None and previous_config.get('original_video'):
        args.original_video=Path(previous_config['original_video'])
    if args.skip_mask_refinement is None:
        args.skip_mask_refinement=not previous_config.get('mask_refinement',True)
    if (previous_config and args.stage!='all'
            and (not args.skip_mask_refinement)!=previous_config.get('mask_refinement',True)):
        parser.error('Mask refinement mode changed: rerun --stage all to regenerate dependent results')
    previous_context=load_video_context(output/'video_context.json') if (output/'video_context.json').is_file() else {}
    if args.context is None:
        context=previous_context
    elif args.stage in ('narrative','export') and context!=previous_context:
        parser.error('Context changed: rerun --stage global before narrative/export')
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
    run_config=dict(input=str(args.input),output=str(output),detector=args.detector,
                analysis_frames=args.analysis_frames,qwen_max_calls=args.qwen_max_calls,settings=settings.model_dump(),mask_refinement=not args.skip_mask_refinement,
                global_model_id=args.global_model_id,global_model_revision=args.global_model_revision,
                global_initialization=args.global_initialization,global_strategy=args.global_strategy,
                global_review_max_calls=args.global_review_max_calls,context=context,
                original_video=str(args.original_video.resolve()) if args.original_video else None)
    from tqdm.auto import tqdm
    total_stages=(6 if args.skip_mask_refinement else 10) if args.stage=='all' else {
        'tracking':1,'geometry':1 if args.skip_mask_refinement else 3,'local':2,
        'mask':0 if args.skip_mask_refinement else 2,'global':1,'narrative':1,'export':1}[args.stage]
    if args.global_strategy=='compact' and args.stage in ('all','global'):total_stages+=1
    if args.global_strategy=='linked' and args.stage in ('all','global'):total_stages+=2
    progress=tqdm(total=total_stages,desc='04 단계',unit='stage')
    def timed(name,fn,*a,**kw):
        before=time.perf_counter();LOG.info('STAGE %s',name);progress.set_postfix_str(name)
        step=dict(stage=name,status='running',started_at=utc_now())
        steps.append(step)
        atomic_json(output/'progress.json',dict(stage=name,completed=int(progress.n),total=total_stages,status='running'))
        try:
            result=fn(*a,**kw)
            step['status']='complete'
            progress.update(1)
            return result
        except BaseException:
            step['status']='failed_or_interrupted'
            raise
        finally:
            step['elapsed_sec']=time.perf_counter()-before
            atomic_json(output/'stage_timing.json',dict(started_at=started_at,stages=steps))
            atomic_json(output/'execution_history.json',history)
            atomic_json(output/'progress.json',dict(stage=name,completed=int(progress.n),total=total_stages,status=step['status']))
    def finish_stage(name):
        atomic_json(output/'status.json',dict(status='stage_complete',pipeline='shot_story',stage=name,
                    started_at=started_at,completed_at=utc_now(),elapsed_sec=time.perf_counter()-started))
    with (output/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tracking_reference=None
        def load_verified_bundle(path):
            from input_media import validate_tracking_sources
            loaded=read_json(path)
            validate_tracking_sources(loaded,args.input,original_video=args.original_video,
                                      cache_dir=output/'input_media')
            if tracking_reference is not None:
                inherited=('version','detector_cache_key','config','tracker_class_policy','appearance_model')
                source_shots={row['shot_id']:row.get('fingerprint') for row in tracking_reference.get('shots',[])}
                child_shots={row['shot_id']:row.get('fingerprint') for row in loaded.get('shots',[])}
                if (source_shots!=child_shots
                        or any(loaded.get(key)!=tracking_reference.get(key) for key in inherited)):
                    raise ValueError('Derived tracking uses different detector/association settings; rerun --stage all.')
            return loaded
        if args.stage not in ('all','tracking'):
            resumed_bundle=load_verified_bundle(output/'dense_tracking/bundle.json')
            tracking_reference=resumed_bundle
        if args.extend_qwen_budget_to is not None:
            extend_generation_budget(output,args.extend_qwen_budget_to,args.budget_extension_reason)
        history['invocations'].append(invocation)
        atomic_json(output/'execution_history.json',history)
        atomic_json(output/'video_context.json',context)
        atomic_json(output/'run_config.json',run_config)
        atomic_json(output/'status.json',dict(status='running',pipeline='shot_story',started_at=started_at))
        backend=None
        try:
            if args.stage in ('all','tracking'):
                bundle=timed('sampled_tracking' if args.analysis_frames=='sampled' else 'dense_tracking',build_dense_tracks,args.input,output,model=args.detector,device='cuda:0',
                             original_video=args.original_video,analysis_frames=args.analysis_frames)
            else:bundle=resumed_bundle
            atomic_json(output/'tracking_summary.json',dict(stats=bundle['stats'],
                shots=[dict(shot_id=s['shot_id'],frames=s['frame_count'],tracks=len(s['tracks']),stats=s.get('stats')) for s in bundle['shots']],
                config=bundle.get('config'),source=args.analysis_frames,analysis_frames=args.analysis_frames))
            if args.stage=='tracking':finish_stage('tracking');return
            if not args.skip_mask_refinement:
                from mask_track_graph import build_mask_track_graph,encode_masked_representatives
                if args.stage in ('all','geometry'):
                    bundle=timed('foreground_tracking',build_mask_track_graph,bundle,output,device='cuda:0')
                    bundle=timed('masked_appearance',encode_masked_representatives,bundle,output,device='cuda:0')
                elif (output/'mask_track_graph/bundle.json').exists():
                    bundle=load_verified_bundle(output/'mask_track_graph/bundle.json')
                elif args.stage=='local':
                    raise RuntimeError('Run --stage geometry first, or use --stage all for the complete pipeline.')
            if args.stage in ('all','geometry','local'):
                from temporal_track_repair import split_temporal_tracks
                bundle=timed('temporal_track_repair',split_temporal_tracks,bundle,output)
            if args.stage=='geometry':finish_stage('geometry');return
            backend=StoryBackend(settings,output,args.qwen_max_calls)
            if args.stage in ('all','local'):
                local=timed('shot_review',review_shots,bundle,output,backend)
                if local['failures'] or len(local['shots'])!=len(bundle['shots']):
                    raise RuntimeError('Some shots lack valid semantic review; see local_entities.json. No partial run is promoted.')
                atomic_json(output/'reviewed_tracking.json',bundle)
            else:
                local=read_json(output/'local_entities.json')
                if (output/'reviewed_tracking.json').exists():bundle=load_verified_bundle(output/'reviewed_tracking.json')
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
                    bundle=load_verified_bundle(output/'mask_refinement/bundle.json')
                    local=read_json(output/'mask_refinement/local.json')
            if args.stage=='mask':finish_stage('mask');return
            if args.stage in ('all','global'):
                if args.global_strategy=='compact':
                    from compact_identity import prepare_local,prepare_temporal
                    bundle,local,temporal_audit=prepare_temporal(bundle,local,output)
                    if temporal_audit or not (output/'compact_temporal_audit.json').exists():
                        atomic_json(output/'compact_temporal_audit.json',temporal_audit)
                    local,split_audit=prepare_local(bundle,local)
                    if backend.settings.model_id!=settings.model_id:
                        backend=StoryBackend(settings,output,args.qwen_max_calls)
                    from temporal_profile_review import review_temporal_profiles
                    local=timed('temporal_identity_review',review_temporal_profiles,bundle,local,output,backend)
                    atomic_json(output/'reviewed_tracking.json',bundle)
                    if not args.skip_mask_refinement:
                        atomic_json(output/'mask_refinement/bundle.json',bundle)
                        atomic_json(output/'mask_refinement/local.json',local)
                    if split_audit or not (output/'unique_local_audit.json').exists():
                        atomic_json(output/'unique_local_audit.json',split_audit)
                    atomic_json(output/'local_entities.json',local)
                if args.global_strategy=='batched':
                    from batched_identity import run_batched
                    bundle,local,global_result=timed('batched_identity',run_batched,bundle,local,output,backend,max_calls=args.global_review_max_calls,
                        review_settings=dict(model_id=args.global_model_id,model_revision=args.global_model_revision),context=context)
                    atomic_json(output/'reviewed_tracking.json',bundle)
                    atomic_json(output/'local_entities.json',local)
                else:
                    if backend.model is not None:
                        del backend.model;backend.model=None;gc.collect();torch.cuda.empty_cache()
                    global_settings=settings.model_copy(update=dict(model_id=args.global_model_id,model_revision=args.global_model_revision))
                    backend=StoryBackend(global_settings,output,args.qwen_max_calls)
                    backend.physical_settings=settings
                    if args.global_strategy=='linked':
                        backend.identity_thinking_budget=1024
                        backend.adaptive_identity_depth=True
                    global_result=timed('global_resolution',resolve_global,bundle,local,output,backend,
                                        review_max_calls=args.global_review_max_calls,context=context,
                                        initialization=args.global_initialization,strategy=args.global_strategy)
                    if args.global_strategy=='linked':
                        if backend.model is not None:
                            del backend.model;backend.model=None;gc.collect();torch.cuda.empty_cache()
                        backend=StoryBackend(settings,output,args.qwen_max_calls)
                        backend.repair_settings=global_settings
                        from catalog_repair import build_profile_cache,repair
                        cache=timed('targeted_appearance',build_profile_cache,bundle,local,output,backend)
                        bundle,local,global_result=timed('physical_identity_repair',repair,bundle,local,global_result,output,backend,cache)
                        atomic_json(output/'reviewed_tracking.json',bundle)
                        atomic_json(output/'local_entities.json',local)
                        if not args.skip_mask_refinement:
                            atomic_json(output/'mask_refinement/bundle.json',bundle)
                            atomic_json(output/'mask_refinement/local.json',local)
            else:
                global_result=read_json(output/'global_entities.json')
                expected_global_model=dict(model_id=args.global_model_id,model_revision=args.global_model_revision)
                if (global_result.get('review_model')
                        and global_result['review_model']!=expected_global_model):
                    raise ValueError('Global catalog was generated by a different model; rerun --stage global before narrative/export')
                if (global_result.get('physical_review_model') and global_result['physical_review_model']!=
                        dict(model_id=settings.model_id,model_revision=settings.model_revision)):
                    raise ValueError('Physical review model changed; rerun --stage global before narrative/export')
                if global_result.get('external_context',{})!=context:
                    raise ValueError('Global catalog used different context; rerun --stage global before narrative/export')
                if global_result.get('global_input_fingerprint')!=global_input_fingerprint(bundle,local):
                    raise ValueError('Global catalog input provenance is missing or changed; rerun --stage global before narrative/export')
            if args.global_strategy=='compact':
                from compact_identity import finalize_compact
                prior_mapping=dict(global_result['mapping'])
                global_result=finalize_compact(global_result,bundle,local,output)
                if args.stage=='export' and prior_mapping!=global_result['mapping']:
                    raise ValueError('Final identity mapping changed; rerun --stage narrative before export.')
            global_result=annotate_identity_scopes(global_result,local)
            atomic_json(output/'global_entities.json',global_result)
            local=apply_global_rejections(local,global_result,output)
            if args.stage=='global':finish_stage('global');return
            if args.stage in ('all','narrative'):
                if (backend.settings.model_id,backend.settings.model_revision)!=(settings.model_id,settings.model_revision):
                    if backend.model is not None:
                        del backend.model;backend.model=None;gc.collect();torch.cuda.empty_cache()
                    backend=StoryBackend(settings,output,args.qwen_max_calls)
                if args.global_strategy=='batched' and (output/'shot_memory/memory.json').is_file():
                    from shot_memory import korean_catalog_narrative
                    global_result=timed('narrative_revision',korean_catalog_narrative,bundle,local,global_result,read_json(output/'shot_memory/memory.json'))
                else:
                    global_result=timed('narrative_revision',revise_narrative,bundle,local,global_result,output,backend)
            if args.stage=='narrative':finish_stage('narrative');return
            if backend.model is not None:
                del backend.model;backend.model=None;gc.collect();torch.cuda.empty_cache()
            from sampled_frames import assert_sampled_evidence
            assert_sampled_evidence(bundle)
            exported=timed('export',export_story,bundle,local,global_result,output,args.input)
            issues=[]
            validation=exported['validation']
            if validation['simultaneous_identity_conflicts']:issues.append(f"동일 ID의 서로 다른 동시 박스 {validation['simultaneous_identity_conflicts']}건")
            if validation['event_reference_errors']:issues.append('사건의 Entity 참조 또는 시각 오류')
            if validation['coverage']['unreviewed_tracks']:issues.append('미검토 트랙이 남아 있습니다.')
            empty_entities=validation.get('entities_without_observations') or validation.get('empty_entities') or []
            if empty_entities:issues.append('관측이 없는 Entity가 있습니다.')
            if validation.get('empty_summaries'):issues.append('요약이 비어 있는 샷이 있습니다.')
            if validation.get('missing_representatives'):issues.append('대표 이미지가 없는 Entity가 있습니다.')
            structural_failure=(validation.get('structurally_complete') is False or bool(
                validation['coverage']['unreviewed_tracks'] or validation['event_reference_errors']
                or empty_entities or validation.get('empty_summaries')))
            if global_result.get('errors'):issues.append('일부 전역 검토 실패: global_entities.json 확인')
            audit=global_result.get('identity_reconciliation',{})
            if audit.get('status')=='needs_review':
                issues.append('전역 동일성 검토에 미해결 또는 미검토 후보가 있습니다: '+('compact_identity/audit.json' if global_result.get('strategy')=='compact' else 'identity_reconciliation/audit.json'))
            local_audits=[s['local_identity_audit'] for s in bundle['shots'] if s.get('local_identity_audit')]
            if any(a.get('status')=='needs_review' for a in local_audits):
                issues.append('샷 내부 동일성 집중 검토에 미해결 후보가 있습니다: local_identity_audit/')
            classification_audits=[s['local_classification_audit'] for s in bundle['shots'] if s.get('local_classification_audit')]
            if any(a.get('status')=='needs_review' for a in classification_audits):
                issues.append('검출·의미 분류 충돌 검토에 미해결 후보가 있습니다: local_classification_audit/')
            if global_result.get('narrative_failures'):issues.append('일부 샷의 의미 요약 보정 실패')
            if global_result.get('narrative_scope_guards'):
                issues.append('미확인 부분을 언급한 자동 요약을 관측 목록으로 대체했습니다: '+', '.join(global_result['narrative_scope_guards']))
            missing=[dict(shot_id=s['shot_id'],objects=s['missing_objects']) for s in local['shots'] if s.get('missing_objects')]
            if missing:issues.append(f'{len(missing)}개 샷에 VLM이 지적한 검출 누락 후보가 남아 있습니다.')
            uncertain=[e['entity_id'] for e in global_result['entities'] if e.get('uncertain')]
            if uncertain:issues.append(f'{len(uncertain)}개 전역 Entity의 동일성/분류가 불확실합니다.')
            quality=dict(status='needs_review' if issues else 'awaiting_evaluation',
                success=False,issues=issues,missing_objects=missing,uncertain_entities=uncertain,
                export=exported,manual_evaluation=None,
                identity_reconciliation=audit,
                local_identity_audits=local_audits,
                local_classification_audits=classification_audits,
                note='실행 완료와 실제 식별 성공을 구분합니다. 독립 평가와 영상 검토가 필요합니다.')
            if structural_failure:
                quality['status']='failed_structure'
                quality['structurally_complete']=False
                atomic_json(output/'quality.json',quality)
                raise ValueError('Export structure validation failed; see quality.json. This run is not promoted to latest_run.')
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
            progress.close()
            final_status=read_json(output/'status.json')
            atomic_json(output/'run_timing.json',dict(final_status,elapsed_sec=invocation['elapsed_sec'],elapsed_human=format_duration(invocation['elapsed_sec']),stages=steps,analysis_frames=args.analysis_frames))
            lines=[f"상태: {final_status.get('status')}",f"분석 프레임: {args.analysis_frames}"]
            lines += [f"{row['stage']}: {row['elapsed_sec']:.2f}초 ({row['status']})" for row in steps]
            lines.append(f"04 전체 시간: {invocation['elapsed_sec']:.2f}초 ({invocation['elapsed_sec']/60:.2f}분)")
            lines.append('모델 적재·분석·출력 포함. 다운로드와 02/03 시간은 별도입니다.')
            (output/'timing_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            print('\n'+'\n'.join(lines)+f"\n기록: {output/'run_timing.json'}",flush=True)


if __name__=='__main__':main()
