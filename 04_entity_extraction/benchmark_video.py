#!/usr/bin/env python3
"""02 -> 03 -> 04 benchmark with explicit failed-run resume; --check does not analyze."""
import argparse
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parent
GIB = 1024 ** 3


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save_report(root, report):
    temporary = root / 'benchmark_timing.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(root / 'benchmark_timing.json')
    lines = [f"실행 상태: {report['status']}", f"입력: {report['input']}",
             f"시작(UTC): {report['started_at']}",
             (report.get('scope_description') or ('범위: 중단 실행 이어하기. 전체 경과는 각 실행 시간의 합계(중단 후 대기 제외).' if report.get('resumed') else
              '범위: 새 결과 캐시로 02 → 03 → 04 한 번. 다운로드/설치/사전 검사는 제외.')),
             '기존 모델 가중치는 재사용; 각 프로세스의 모델 적재·추론·검증·출력은 포함.', '']
    for row in report['stages']:
        seconds = row.get('elapsed_sec')
        lines.append(f"{row['name']}: {seconds:.2f}초 / {row['status']}" if seconds is not None
                     else f"{row['name']}: 실행 중")
    if 'elapsed_sec' in report:
        lines.extend(['', f"전체 경과: {report['elapsed_sec']:.2f}초 ({report['elapsed_sec']/60:.2f}분)"])
    if report.get('analysis_status'):
        lines.append(f"04 분석 상태: {report['analysis_status']}; 품질: {report.get('quality_status', 'unknown')}")
    if 'qwen_generations' in report:
        lines.append(f"실제 Qwen 생성: {report['qwen_generations']} / {report['qwen_limit']}")
    if report.get('error'):
        lines.append(f"오류: {report['error']}")
    if report['status'] != 'complete':
        lines.append('미완료 실행의 시간은 전체 처리 시간 측정값으로 사용하지 마세요.')
    lines.extend(['', '04 세부 시간: entities/run_timing.json (실패 시 execution_history.json도 확인)',
                  '품질 판정: entities/quality.json; 완료와 정확도는 별개입니다.'])
    (root / 'benchmark_summary.txt').write_text('\n'.join(lines) + '\n')


def run_stage(name, command, root, report, env):
    row = dict(name=name, command=list(map(str, command)), started_at=utc_now(), status='running',
               log=f'{name}' + (f'_attempt_{len(report["stages"])+1}' if report.get('resumed') else '') + '.log')
    report['stages'].append(row)
    save_report(root, report)
    started = time.perf_counter()
    process = None
    try:
        print(f'\n[{name}] {shlex.join(row["command"])}', flush=True)
        with (root / row['log']).open('w') as log:
            process = subprocess.Popen(row['command'], cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors='replace',
                                       start_new_session=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        row['returncode'] = code
        if code:
            raise RuntimeError(f'{name} 실패 (exit {code}); {root / row["log"]}')
        row['status'] = 'complete'
    except BaseException as exc:
        row['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        row['error'] = str(exc) or type(exc).__name__
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        row.update(ended_at=utc_now(), elapsed_sec=time.perf_counter() - started)
        save_report(root, report)


def preflight(video, output, minimum_gib, resume=False):
    if not video.is_file():
        raise FileNotFoundError(f'입력 영상이 없습니다: {video}')
    if output.exists() and not resume:
        raise FileExistsError(f'새 실행만 허용합니다. 기존 출력 폴더: {output}')
    for binary in ('ffmpeg', 'ffprobe'):
        if not shutil.which(binary):
            raise RuntimeError(f'{binary}가 필요합니다.')
    # Import and inspect resources only. Do not load a model or analyze a frame.
    import av
    import cv2
    import numpy
    import scenedetect
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('04 분석에 사용할 CUDA GPU를 찾을 수 없습니다.')
    weights = [REPO / '02_shot_boundary_detection/models/transnetv2.pth',
               PROJECT / 'assets/models/yolo11x.pt',
               PROJECT / 'models/Qwen3-VL-8B-Thinking-NF4/config.json',
               PROJECT / 'models/Qwen3-VL-8B-Thinking-NF4/model.safetensors.index.json']
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(f'측정 전에 모델을 준비하세요: {path}')
    from huggingface_hub import snapshot_download
    for name, revision in [
        ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
        ('facebook/dinov2-small', 'ed25f3a31f01632728cabb09d1542f84ab7b0056')]:
        snapshot_download(name, revision=revision, local_files_only=True)
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    available = shutil.disk_usage(parent).free / GIB
    print(f'입력: {video}\n출력: {output}\n출력 디스크 여유: {available:.2f} GiB', flush=True)
    if available < minimum_gib:
        raise RuntimeError(f'저장 공간 부족: 최소 {minimum_gib:g} GiB를 확보하거나 '
                           '--output으로 여유 있는 디스크의 새 폴더를 지정하세요. '
                           '(10 GiB는 예고편용 시작 기준이며 최대 사용량 보장은 아닙니다.)')
    metadata = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,avg_frame_rate,nb_frames,duration', '-of', 'json', str(video)], text=True))
    return dict(video=metadata['streams'][0], gpu=torch.cuda.get_device_name(0),
                disk_free_gib=available, python=sys.version,
                packages={name: importlib.metadata.version(name) for name in
                          ('torch', 'transformers', 'ultralytics', 'av', 'scenedetect-headless')})


def commands(video, output, calls, global_calls, strategy="linked", context=None):
    return [
        ('02_shots', [sys.executable, str(REPO / '02_shot_boundary_detection/split_shots.py'),
                      str(video), '--output', str(output / 'shots'), '--detector', 'transnet']),
        ('03_sampling', [sys.executable, str(REPO / '03_adaptive_sampling/adaptive_sampling.py'),
                         '--input-list', str(output / 'shots/shots.json'),
                         '--output', str(output / 'adaptive')]),
        ('04_entities', [str(PROJECT / 'run.sh'), '--input', str(output / 'adaptive'),
                         '--original-video', str(video), '--output', str(output / 'entities'),
                         '--stage', 'all', '--analysis-frames', 'sampled', '--skip-mask-refinement',
                         '--global-initialization', 'local', '--global-strategy', strategy, '--qwen-max-calls', str(calls),
                         '--global-review-max-calls', str(global_calls),
                         '--model-id', 'Qwen/Qwen3-VL-8B-Instruct',
                         '--model-revision', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                         '--global-model-id', str(PROJECT/'models/Qwen3-VL-8B-Thinking-NF4'),
                         '--global-model-revision', '92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b'] + (['--context',str(context)] if context else []))]


def acquire_run_lock(output):
    lock = (output/'.benchmark.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('같은 결과 폴더의 분석이 이미 실행 중입니다.') from None
    return lock


def resume_report(output, video, calls, global_calls):
    report = read_json(output / 'benchmark_timing.json')
    if report.get('status') not in ('failed', 'interrupted'):
        raise ValueError('실패/중단 상태의 실행만 이어갈 수 있습니다.')
    if Path(report['input']).resolve() != video or report['input_sha256'] != digest(video):
        raise ValueError('이전 실행과 입력 영상이 다릅니다.')
    policy = report.get('policy', {})
    if (policy.get('analysis_frames') != 'sampled' or policy.get('qwen_limit') != calls
            or policy.get('global_review_limit') != global_calls):
        raise ValueError('이전 실행과 분석 모드/예산이 다릅니다.')
    return report


def resume_plan(plan, report, output):
    completed = {row['name'] for row in report['stages'] if row['status'] == 'complete'}
    required = {'02_shots': output/'shots/shots.json', '03_sampling': output/'adaptive/summary.json'}
    for name, path in required.items():
        if name in completed and not path.is_file():
            raise ValueError(f'이어하기에 필요한 중간 결과가 없습니다: {path}')
    return [(name, command) for name, command in plan if name not in completed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True, help='분석할 원본 영상 파일')
    parser.add_argument('--output', type=Path, help='새 결과 폴더; --resume와 함께 쓰면 기존 실패/중단 폴더')
    parser.add_argument('--qwen-max-calls', type=int, default=320)
    parser.add_argument('--global-review-max-calls', type=int, default=None)
    parser.add_argument('--min-free-gib', type=float, default=10)
    parser.add_argument('--resume', action='store_true', help='--output의 실패/중단 실행을 캐시와 예산 기록을 유지하여 이어하기')
    parser.add_argument('--check', action='store_true', help='의존성·모델·디스크만 검사. 분석하지 않음')
    parser.add_argument('--context',type=Path,help='영상 종류와 출처가 있는 선택적 사전 정보 JSON')
    parser.add_argument('--global-strategy', choices=['linked','batched'], default='linked')
    args = parser.parse_args()
    if args.context is not None:args.context=args.context.expanduser().resolve()
    if args.global_review_max_calls is None:
        args.global_review_max_calls = 10 if args.global_strategy=='batched' else 96
        if args.resume and args.output:
            try:
                args.global_review_max_calls = read_json(args.output.expanduser().resolve()/'benchmark_timing.json')['policy']['global_review_limit']
            except (OSError, KeyError, ValueError):
                pass  # Normal resume validation below reports the missing/invalid record.
    if args.qwen_max_calls < 1 or not 0 <= args.global_review_max_calls < args.qwen_max_calls:
        parser.error('예산은 total > global >= 0, total >= 1이어야 합니다.')
    if not 0 < args.min_free_gib < float('inf'):
        parser.error('--min-free-gib는 양의 유한수여야 합니다.')
    video = args.video.expanduser().resolve()
    output = (args.output or PROJECT / 'outputs' /
              f'{video.stem}_benchmark_{datetime.now():%Y%m%d_%H%M%S_%f}').expanduser().resolve()
    prior = None
    run_lock = None
    try:
        if args.resume:
            if args.output is None:raise ValueError('--resume에는 --output이 필요합니다.')
            run_lock = acquire_run_lock(output)
            prior = resume_report(output, video, args.qwen_max_calls, args.global_review_max_calls)
            if (args.global_strategy=='batched') != (prior.get('policy',{}).get('global_strategy')=='batched'):
                raise ValueError('기존 결과와 새 묶음 실험은 서로 다른 폴더에서 실행해야 합니다.')
        environment = preflight(video, output, args.min_free_gib, resume=args.resume)
    except Exception as exc:
        print(f'사전 검사 실패: {exc}', file=sys.stderr)
        return 2
    plan = commands(video, output, args.qwen_max_calls, args.global_review_max_calls,args.global_strategy,args.context)
    if prior:
        if prior.get('policy',{}).get('global_strategy','pairwise')!='linked':
            old=next((r['command'] for r in reversed(prior['stages']) if r['name']=='04_entities'),None)
            if old:plan=[(name,old if name=='04_entities' else command) for name,command in plan]
        try:plan = resume_plan(plan, prior, output)
        except (ValueError, OSError) as exc:
            print(f'이어하기 검사 실패: {exc}', file=sys.stderr);return 2
    from video_context import load_video_context
    media_context=load_video_context(args.context)
    if prior and media_context!=prior.get('policy',{}).get('video_context',{}):
        print('이어하기의 영상 사전 정보가 변경되었습니다.',file=sys.stderr);return 2
    if args.check:
        for name, command in plan:
            print(name, shlex.join(command))
        print('사전 검사 통과. 분석은 실행하지 않았습니다.')
        return 0
    # Input hashing and resource preparation are explicitly outside the timer.
    source_sha = digest(video)
    output.mkdir(parents=True, exist_ok=args.resume)
    if run_lock is None:run_lock = acquire_run_lock(output)
    env = dict(os.environ, ENTITY_PYTHON=sys.executable, PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    started = time.perf_counter()
    report = dict(status='running', started_at=utc_now(), input=str(video), input_sha256=source_sha,
                  output=str(output), environment=environment, stages=[],
                  policy=dict(fresh_analysis_cache=True, existing_model_weights=True,
                              downloads_and_preflight_excluded=True, model_load_included=True,
                              analysis_frames="sampled", mask_refinement=False, external_context=bool(args.context),
                              qwen_limit=args.qwen_max_calls, global_review_limit=args.global_review_max_calls,global_strategy=args.global_strategy,video_context=media_context))
    previous_elapsed = 0.0
    if prior:
        previous_elapsed = prior.get('elapsed_sec', 0.0)
        history = output/'benchmark_attempts'
        history.mkdir(exist_ok=True)
        (history/f'{time.time_ns()}.json').write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n')
        report = prior
        report.update(status='running', resumed=True, resumed_at=utc_now(), environment=environment)
        report.pop('error', None)
        report.pop('ended_at', None)
    save_report(output, report)
    exit_code = 0
    try:
        from tqdm.auto import tqdm
        for name, command in tqdm(plan,total=len(plan),desc='전체 02 → 03 → 04',unit='stage'):
            run_stage(name, command, output, report, env)
        state = read_json(output / 'entities/status.json')
        if state.get('status') not in ('complete', 'complete_with_warnings'):
            raise RuntimeError('04가 전체 분석을 완료하지 않았습니다.')
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = str(exc) or type(exc).__name__
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        report.update(ended_at=utc_now(), elapsed_sec=previous_elapsed + time.perf_counter() - started)
        for relative, fields in [
            ('entities/status.json', {'status': 'analysis_status'}),
            ('entities/quality.json', {'status': 'quality_status'}),
            ('entities/qwen_budget.json', {'generations': 'qwen_generations', 'limit': 'qwen_limit'})]:
            path = output / relative
            if path.is_file():
                try:
                    data = read_json(path)
                    report.update({dest: data.get(src) for src, dest in fields.items()})
                except (OSError, ValueError):
                    pass
        save_report(output, report)
        print(f'\n시간 기록: {output / "benchmark_summary.txt"}', flush=True)
        print((output / 'benchmark_summary.txt').read_text(), flush=True)
        if not exit_code:
            print('뷰어: ' + shlex.join([str(PROJECT / 'visualize.sh'), str(output / 'entities')]))
    return exit_code


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
