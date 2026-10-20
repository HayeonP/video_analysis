"""Per-video benchmark launch/resume and selection of its last completed result."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VIDEOS = {
    'sintel': ROOT.parent / '00_prototype/videos/sintel_trailer_1080p.mp4',
    'frozen': ROOT.parent / 'videos/frozen_trailer_1080p.mp4',
}


def completed_result(folder, video):
    try:
        record = json.loads((folder / 'benchmark_timing.json').read_text())
        state = json.loads((folder / 'entities/status.json').read_text())
        if (record['status'] == 'complete' and Path(record['input']).resolve() == video.resolve()
                and record.get('policy', {}).get('analysis_frames') == 'sampled'
                and state['status'] in ('complete', 'complete_with_warnings')
                and (folder / 'entities/entities.json').is_file()):
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def latest_result(movie):
    pointer = ROOT / 'outputs' / f'latest_sampled_{movie}.json'
    candidates = []
    if pointer.is_file():
        try:
            candidates.append(Path(json.loads(pointer.read_text())['benchmark_output']))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    candidates.extend(p.parent for p in (ROOT / 'outputs').glob('*/benchmark_timing.json'))
    valid = [(r.get('ended_at', ''), p) for p in set(candidates)
             if (r := completed_result(p, VIDEOS[movie]))]
    if not valid:
        raise ValueError(f'{movie}의 완료된 sampled 결과가 없습니다. 먼저 ./analyze_{movie}.sh를 실행하세요.')
    return max(valid, key=lambda item: item[0])[1] / 'entities'


def comparison_result(movie, variant):
    """Explicit channels never fall back to another experiment's result."""
    pointer = ROOT / 'outputs' / f'{variant}_{movie}.json'
    try:
        folder = Path(json.loads(pointer.read_text())['benchmark_output'])
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(f'{variant} {movie} 결과가 아직 없습니다. 새 실험은 analyze_new_{movie}.sh로 실행하세요.') from None
    if not completed_result(folder, VIDEOS[movie]):
        raise ValueError(f'지정된 비교 결과가 완료 상태가 아닙니다: {folder}')
    return folder / 'entities'


def latest_failed_result(movie):
    candidates=[]
    for path in (ROOT/'outputs').glob('*/benchmark_timing.json'):
        try:
            record=json.loads(path.read_text())
            if (record.get('status') in ('failed','interrupted')
                    and Path(record['input']).resolve()==VIDEOS[movie].resolve()
                    and record.get('policy',{}).get('analysis_frames')=='sampled'):
                candidates.append((record.get('ended_at',record.get('started_at','')),path.parent))
        except (OSError,ValueError,KeyError,TypeError):pass
    if not candidates:raise ValueError(f'{movie}의 이어갈 실패/중단 실행이 없습니다.')
    return max(candidates,key=lambda item:item[0])[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['analyze', 'visualize'])
    parser.add_argument('movie', choices=VIDEOS)
    parser.add_argument('--resume', action='store_true', help='최근 실패/중단 결과 이어하기 (--output으로 직접 지정 가능)')
    parser.add_argument('--output', type=Path, help='분석 결과 폴더 (--resume이면 기존 실패/중단 폴더)')
    parser.add_argument('--variant', choices=['baseline','new'], help='고정된 기존 결과 또는 새 실험 결과')
    args, extra = parser.parse_known_args()
    if args.action == 'visualize':
        if args.resume:parser.error('--resume는 분석용 옵션입니다.')
        if args.output is not None:parser.error('시각화는 영상별 최근 완료 결과를 자동 선택합니다.')
        try:result = comparison_result(args.movie,args.variant) if args.variant else latest_result(args.movie)
        except ValueError as exc:
            parser.exit(1, str(exc)+'\n')
        port = '5006' if args.variant == 'new' else ('5004' if args.movie == 'sintel' else '5005')
        # Let VS Code open the forwarded address after establishing its tunnel.
        # Opening a raw localhost URL is not proof that forwarding succeeded.
        os.execv(sys.executable, [sys.executable, str(ROOT/'viewer_server.py'), str(result), '--port', port, *extra])
    if args.variant=='baseline':parser.error('baseline은 시각화 전용입니다.')
    if args.variant=='new':
        if '--global-strategy' in extra:parser.error('새 실험의 전략은 batched로 고정됩니다.')
        extra=['--global-strategy','batched',*extra]
        if args.resume and args.output is None:parser.error('새 실험 재개에는 --output을 명시하세요.')
    if args.resume and args.output is None:
        try:args.output=latest_failed_result(args.movie)
        except ValueError as exc:parser.exit(1,str(exc)+'\n')
    if args.resume:extra=['--resume',*extra]
    output = (args.output or ROOT/'outputs'/f'{args.movie}_{"new" if args.variant=="new" else "sampled"}_{datetime.now():%Y%m%d_%H%M%S_%f}').expanduser().resolve()
    command = [sys.executable, str(ROOT/'benchmark_video.py'), '--video', str(VIDEOS[args.movie]),
               '--output', str(output), *extra]
    # --check is preparation only: never promote a result or run inference.
    try:code = subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:return 130
    if code == 0 and '--check' not in extra:
        if not completed_result(output, VIDEOS[args.movie]):
            print('완료 결과를 확인할 수 없어 영상별 최근 결과를 변경하지 않았습니다.', file=sys.stderr)
            return 1
        pointer = ROOT/'outputs'/(f'new_{args.movie}.json' if args.variant=='new' else f'latest_sampled_{args.movie}.json')
        pointer.parent.mkdir(parents=True, exist_ok=True)
        temporary = pointer.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(benchmark_output=str(output)), indent=2)+'\n')
        temporary.replace(pointer)
        print(f'\n다음 명령으로 이 결과를 여세요: {ROOT / ("visualize_"+("new_" if args.variant=="new" else "")+args.movie+".sh")}', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
