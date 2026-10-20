#!/usr/bin/env python3
"""Reuse an existing cross-shot catalog and repair physical collisions/labels."""
import argparse
from pathlib import Path
import sys,time,hashlib,fcntl
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
from low_freq_event.storage import atomic_json
from low_freq_event.config import Settings
from story_semantics import StoryBackend,revise_narrative,annotate_identity_scopes
from story_export import export_story
from catalog_repair import read,repair,build_profile_cache
from run_timing import utc_now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Existing entities result whose cross-shot links should be retained')
    parser.add_argument('--profile-cache',type=Path,help='Optional previously checked appearance evidence from the same source pixels')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qwen-max-calls',type=int,default=100)
    parser.add_argument('--skip-narrative',action='store_true',help='Export local summaries with an explicit pending-review warning')
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if source==out:parser.error('Use a separate output; the source is never overwritten.')
    lock=(out/'.repair.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter();started=utc_now();history=read(out/'repair_execution_history.json') if (out/'repair_execution_history.json').is_file() else []
    provenance=dict(source=str(source),artifacts={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('reviewed_tracking.json','local_entities.json','global_entities.json')},profile_cache=str(args.profile_cache.resolve()) if args.profile_cache else None)
    if (out/'repair_source.json').is_file() and read(out/'repair_source.json')!=provenance:raise ValueError('Resume source changed; use a new output')
    atomic_json(out/'repair_source.json',provenance)
    config=read(source/'run_config.json');config.update(output=str(out),global_strategy='linked',qwen_max_calls=args.qwen_max_calls,global_review_max_calls=0)
    atomic_json(out/'run_config.json',config)
    atomic_json(out/'status.json',dict(status='running',pipeline='shot_story',phase='catalog_repair'))
    backend=StoryBackend(Settings(**config['settings']),out,args.qwen_max_calls)
    try:
        bundle=read(source/'reviewed_tracking.json');bundle.setdefault('asset_root',str(source))
        local=read(source/'local_entities.json');catalog=read(source/'global_entities.json')
        cache=args.profile_cache or build_profile_cache(bundle,local,out,backend)
        bundle,local,catalog=repair(bundle,local,catalog,out,backend,cache)
        catalog=annotate_identity_scopes(catalog,local)
        atomic_json(out/'reviewed_tracking.json',bundle);atomic_json(out/'local_entities.json',local);atomic_json(out/'global_entities.json',catalog)
        if not args.skip_narrative:
            catalog=revise_narrative(bundle,local,catalog,out,backend)
            atomic_json(out/'global_entities.json',catalog)
        # Free GPU memory before rendering; sampled frames stay the sole evidence.
        backend.model=None
        exported=export_story(bundle,local,catalog,out,Path(config['input']))
        validation=exported['validation']
        if not validation.get('structurally_complete') or validation.get('simultaneous_identity_conflicts'):
            raise ValueError('Repair export failed structure/identity validation; see story_validation.json')
        uncertain=sum(e.get('uncertain',False) for e in catalog['entities'])
        issues=[]
        if uncertain:issues.append(f'{uncertain}개 identity가 미확정입니다.')
        if args.skip_narrative or catalog.get('narrative_failures'):issues.append('샷 요약은 추가 검토가 필요합니다.')
        atomic_json(out/'quality.json',dict(status='needs_review' if issues else 'passed',issues=issues,export=exported))
        atomic_json(out/'status.json',dict(status='complete_with_warnings' if issues else 'complete',pipeline='shot_story',phase='complete'))
    except (Exception,KeyboardInterrupt) as exc:
        atomic_json(out/'status.json',dict(status='failed',pipeline='shot_story',phase='catalog_repair',error=str(exc)))
        raise
    finally:
        elapsed=time.perf_counter()-start
        history.append(dict(started_at=started,ended_at=utc_now(),elapsed_sec=elapsed,generations_through_attempt=backend.ledger['generations']))
        atomic_json(out/'repair_execution_history.json',history)
        atomic_json(out/'run_timing.json',dict(started_at=started,ended_at=utc_now(),elapsed_sec=elapsed,scope='Repair only; prior tracking/local/global inference reused'))
        print(f'복구 실행 시간: {elapsed:.2f}초 / 누적 실행 {sum(h["elapsed_sec"] for h in history):.2f}초 / 실제 모델 호출 {backend.ledger["generations"]}회',flush=True)
    from story_report import write_report
    write_report(out)
    from repair_report import write_repair_report
    write_repair_report(out)

if __name__=='__main__':main()
