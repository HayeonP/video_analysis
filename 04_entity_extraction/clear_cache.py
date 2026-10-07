#!/usr/bin/env python3
"""Remove stage-04 inference caches while retaining inputs, weights and final exports."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
CACHE_NAMES=('frames','grounding','features','crops','request_cache','traces','review_traces',
             'workers','sequential','temporal_review','temporal_retrieval','identity_galleries','yolo_proposals.json','run_config.json',
             'cache_provenance.json','identity_before_review.json','identity_review_progress.json',
             'identity_state.json','bounded','qwen_budget.json','tracklets.json','weak_tracklets.json',
             'features.meta.json','yolo_proposals.meta.json','detector_summary.json','anchored','duplicate_observations.json',
             'track_fusion.json','observation_fusion.json')


def cache_targets(root):
    folders={p.parent for p in root.rglob('run_config.json')}
    folders.update(p.parent for p in root.rglob('entities.json'))
    targets=[]
    for folder in sorted(folders):
        if not folder.resolve().is_relative_to(root.resolve()):
            raise ValueError('결과 폴더 외부의 경로는 삭제할 수 없습니다.')
        for name in CACHE_NAMES:
            path=folder/name
            if path.exists() or path.is_symlink():targets.append(path)
    return targets


def clear_caches(root, *, apply=False):
    root=root.resolve()
    # Hold every known run lock until deletion completes. No partial deletion when busy.
    with ExitStack() as stack:
        locks=set(root.rglob('.run.lock'))
        locks.add(root/'.run.lock')
        folders={p.parent for p in root.rglob('run_config.json')}
        folders.update(p.parent for p in root.rglob('entities.json'))
        for folder in folders:
            locks.add(folder/'.run.lock')
            if folder.name in ('with_qwen','without_qwen'):locks.add(folder.parent/'.run.lock')
        for path in sorted(locks):
            if not path.resolve().is_relative_to(root):
                raise ValueError('외부 경로의 잠금 파일은 허용하지 않습니다.')
            handle=stack.enter_context(path.open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        targets=cache_targets(root)
        if apply:
            for path in targets:
                if path.is_symlink() or path.is_file():path.unlink()
                else:shutil.rmtree(path)
        return [str(p.relative_to(root)) for p in targets]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--apply',action='store_true',help='실제 삭제. 생략하면 대상만 표시.')
    args=parser.parse_args()
    root=args.output.expanduser().resolve()
    if not root.is_dir() or not root.is_relative_to((ROOT/'outputs').resolve()):
        parser.error('04_entity_extraction/outputs 아래의 기존 결과 폴더만 지정하세요.')
    try:removed=clear_caches(root,apply=args.apply)
    except BlockingIOError:parser.exit(1,'분석이 실행 중입니다. 분석 종료 후 캐시를 삭제하세요.\n')
    report=dict(applied=args.apply,at_utc=datetime.now(timezone.utc).isoformat(),
                root=str(root),paths=removed,retained='input videos, model weights, final catalogs/images/videos')
    if args.apply:(ROOT/'CACHE_CLEAR_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
