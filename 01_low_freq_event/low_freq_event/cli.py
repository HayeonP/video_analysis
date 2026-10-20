import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess

from .config import ROOT, Settings


def existing(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists() or candidate.is_absolute():
        return candidate.resolve()
    return (ROOT / candidate).resolve()


def default_analysis_output(video, settings, outputs=None):
    """Resume compatible output; preserve older runs when code/settings change."""
    from .config import PIPELINE_VERSION
    from .pipeline import code_fingerprint
    from .storage import file_fingerprint, read_json
    suffix = '_unified' if settings.perception_device == 'cpu' else '_unified_gpu'
    base = (outputs or ROOT / 'outputs') / (video.stem + suffix)
    manifest = base / 'manifest.json'
    if not manifest.exists():
        return base
    identity = dict(sha256=file_fingerprint(video)['sha256'], settings=settings.model_dump(),
                    pipeline_version=PIPELINE_VERSION, code_sha256=code_fingerprint())
    if read_json(manifest).get('identity') == identity:
        return base
    revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    result = base.with_name(base.name + '_' + revision)
    logging.info('이전 코드/설정의 결과를 보존하고 별도 결과 경로 사용: %s', result)
    return result


def main():
    parser = argparse.ArgumentParser(description="저빈도 영상 이벤트 추출 · Stage 1 전용")
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("analyze", help="분석하거나 저장된 중단 지점에서 재개")
    run.add_argument("video", help="입력 영상 파일 경로")
    run.add_argument("--output", type=Path, help="생략하면 outputs/<영상 이름>_unified_gpu (CPU 지정 시 _unified)")
    run.add_argument("--config", type=Path)
    run.add_argument('--tracking-only', action='store_true', help='추적까지만 실행; VLM 없이 같은 폴더에서 후속 분석 가능')
    run.add_argument('--tracking-fps', type=float, help='추적 FPS; VLM 샘플 FPS와 별개, 기본 10')
    run.add_argument("--device", choices=['auto', 'cuda', 'cpu'], help="검출·특징 모델 장치; auto는 GPU 우선, 메모리 부족 시 CPU")
    run.add_argument("--max-windows", type=int, help="이번 실행에서 처리할 윈도 수; 이후 같은 명령으로 재개")
    segments = commands.add_parser('events-from-tracks', help='저장된 10fps 추적으로 5초 비중첩 인물·배경·구도 요약')
    segments.add_argument('result', nargs='?', default=str(ROOT/'outputs/260929_pretty_girl_person_tracking_v4'))
    segments.add_argument('--output', type=Path, help='기본: 입력 폴더 이름 뒤 _focused_summaries_10fps_<픽셀예산>px')
    segments.add_argument('--max-windows', type=int, help='이번 실행에서 추가 분석할 구간 수; 각 윈도는 독립적으로 저장')
    segments.add_argument('--max-pixels', type=int, default=65536, help='VLM 프레임 해상도 예산; FPS는 10 유지')
    segments.add_argument('--plan', action='store_true', help='입력·프레임 수만 확인; 모델 로드 및 결과 생성 없음')
    tracking = commands.add_parser('review-tracking', help='저장된 검출 특징으로 CPU 추적 비교; 모델 실행 없음')
    tracking.add_argument('result')
    tracking.add_argument('--output', type=Path, required=True)
    revise = commands.add_parser('reanalyze-events', help='저장된 bbox·통합 ID를 보존하고 프레임 근거로 이벤트만 재분석')
    revise.add_argument('result')
    revise.add_argument('--output', type=Path, required=True)
    revise.add_argument('--max-windows', type=int)
    viewer = commands.add_parser("visualize", help="저장된 결과를 표시; 모델을 실행하지 않음")
    viewer.add_argument("result", nargs="?", help="결과 폴더/JSON 경로; 생략하면 outputs의 가장 최근 완료된 분석/추적")
    viewer.add_argument("--source-name", help="이 원본 영상 파일명의 가장 최근 완료된 결과만 선택")
    viewer.add_argument("--video", type=Path, help="영상 이동 시 경로 재지정")
    viewer.add_argument("--port", type=int, default=5001)
    viewer.add_argument("--host", default="0.0.0.0")
    viewer.add_argument("--generate-only", action="store_true")
    trim = commands.add_parser("trim", help="입력 영상의 앞부분을 독립적인 mp4로 저장")
    trim.add_argument("source", type=Path)
    trim.add_argument("--seconds", type=float, default=10)
    trim.add_argument("--output", type=Path, default=ROOT / "outputs/sintel_10s.mp4")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        if args.command in {None, "analyze"}:
            if args.command is None:
                args = parser.parse_args(["analyze"])
            from .pipeline import analyze
            if args.max_windows is not None and args.max_windows < 1:
                parser.error("--max-windows는 1 이상이어야 합니다.")
            settings = Settings.load(args.config)
            if args.tracking_fps is not None:
                settings = Settings.model_validate(dict(settings.model_dump(), tracking_fps=args.tracking_fps))
            if args.device is not None:
                settings.perception_device = args.device
            if args.output is None:
                args.output = default_analysis_output(existing(args.video), settings)
            result = analyze(existing(args.video), args.output, settings, max_windows=args.max_windows, tracking_only=args.tracking_only)
            print(f"결과: {args.output.resolve() / 'timeline.json'}")
            print(f"누적 수행시간: {result['timing'].get('total_active_sec', 0):.2f}초")
        elif args.command == 'events-from-tracks':
            from .segment_events import analyze_segments, load_plan
            if args.max_windows is not None and args.max_windows < 1:
                parser.error('--max-windows는 1 이상이어야 합니다.')
            if args.max_pixels < 4096:
                parser.error('--max-pixels는 4096 이상이어야 합니다.')
            source = existing(args.result)
            output = existing(str(args.output)) if args.output else source.with_name(source.name+f'_focused_summaries_10fps_{args.max_pixels}px')
            if args.plan:
                original, frames, windows = load_plan(source)
                print(json.dumps(dict(source=str(source), output=str(output), fps=10, window_sec=5, overlap_sec=0, max_pixels=args.max_pixels,
                    total_frames=len(frames), windows=[{k:v for k,v in w.items() if k!='frames'} | {'frame_count':len(w['frames'])} for w in windows],
                    boundary_review=False, merge_windows=False, action_timestamps=False, analysis_strategy='full_frame_per_person_then_scene', crop_people=False, event_scope="people_and_scene", model_loaded=False),ensure_ascii=False,indent=2))
            else:
                result=analyze_segments(source,output,max_windows=args.max_windows,max_pixels=args.max_pixels)
                print(f"상태: {result['status']} · 결과: {output/'timeline.json'}")
                print(f"시각화: ./visualize.sh {output}")
        elif args.command == 'review-tracking':
            from .tracking_review import review_tracking
            print(json.dumps(review_tracking(existing(args.result), args.output), ensure_ascii=False, indent=2))
        elif args.command == 'reanalyze-events':
            from .reanalyze import reanalyze_events
            if args.max_windows is not None and args.max_windows < 1:
                parser.error('--max-windows는 1 이상이어야 합니다.')
            result = reanalyze_events(existing(args.result), args.output, max_windows=args.max_windows)
            print(f"결과: {args.output.resolve() / 'timeline.json'}")
            print(f"누적 수행시간: {result['timing']['total_active_sec']:.2f}초")
        elif args.command == "visualize":
            from .viewer import generate_viewer, latest_completed_result
            from .viewer_server import serve
            if args.result is not None and args.source_name:
                parser.error('결과 경로와 --source-name은 함께 지정할 수 없습니다.')
            result = existing(args.result) if args.result is not None else latest_completed_result(ROOT / "outputs", source_name=args.source_name)
            if result.is_file():
                result = result.parent
            print(f"선택한 분석 결과: {result}")
            page = generate_viewer(result, video=args.video)
            print(f"시각화: {page}")
            if not args.generate_only:
                # Serve only results, crops and copied input video; not source, credentials or venv.
                serve(result, args.port, args.host, home=page)
        elif args.command == "trim":
            if args.seconds <= 0:
                parser.error("--seconds는 양수여야 합니다.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(args.source),
                            "-t", str(args.seconds), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                            "-movflags", "+faststart", str(args.output)], check=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.exception("실행 실패: %s", exc)
        return 1
