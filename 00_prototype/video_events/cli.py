from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

from .config import Settings
from .paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def doctor() -> int:
    import av
    import bitsandbytes
    import torch
    import transformers

    report = {
        "python": sys.executable,
        "virtual_environment": sys.prefix,
        "project_environment_active": Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__, "pyav": av.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if report["cuda_available"]:
        free, total = torch.cuda.mem_get_info()
        report.update(gpu=torch.cuda.get_device_name(0), vram_gib=round(total / 2**30, 2),
                      free_vram_gib=round(free / 2**30, 2), cuda_runtime=torch.version.cuda)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["project_environment_active"] and report["cuda_available"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL 4B·4bit 영상 이벤트 분석 (프로젝트 가상환경 자동 사용)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analysis = subparsers.add_parser("analyze", help="전체 영상 이벤트 분석 / 같은 명령으로 자동 이어하기")
    analysis.add_argument("video", type=Path)
    analysis.add_argument("--output", "-o", type=Path, help="결과 폴더 (기본: 프로젝트 outputs/영상이름)")
    analysis.add_argument("--config", type=Path, help="JSON 설정 파일")
    analysis.add_argument("--limit-seconds", type=float, help="앞부분만 실행할 때 분석 길이")
    analysis.add_argument("--fail-fast", action="store_true", help="디버깅용: 오류 발생 시 중단 (기본값은 기록 후 계속)")
    viewer = subparsers.add_parser("visualize", help="완료된 결과로 이벤트·영상·객체 박스 HTML 생성 (GPU 사용 없음)")
    viewer.add_argument("result", type=Path, help="분석 결과 폴더 또는 timeline.json")
    viewer.add_argument("--output", "-o", type=Path, help="HTML 저장 경로 (기본: 결과 폴더/viewer.html)")
    viewer.add_argument("--video", type=Path, help="원본 영상이 이동한 경우 새 경로")
    first_viewer = subparsers.add_parser("first-round-visualize", help="저장된 1차 후보만 시각화하고 서버 실행 (재추론 없음)")
    first_viewer.add_argument("result", type=Path, nargs="?", default=ROOT / "outputs/sintel", help="결과 폴더 또는 candidates.json (기본: outputs/sintel)")
    first_viewer.add_argument("--output", "-o", type=Path, help="HTML 경로 (기본: 결과 폴더/first_round.html)")
    first_viewer.add_argument("--video", type=Path, help="원본 영상이 이동한 경우 새 경로")
    first_viewer.add_argument("--generate-only", action="store_true", help="HTML만 생성하고 서버는 시작하지 않음")
    first_viewer.add_argument("--port", type=int, default=5001, help="서버 포트 (기본: 5001)")
    first_viewer.add_argument("--host", default="0.0.0.0", help="접속 주소 (기본: 같은 네트워크에서 접속 가능)")
    server = subparsers.add_parser("serve", help="같은 네트워크에서 접속하는 뷰어 서버 (GPU 사용 없음)")
    server.add_argument("--port", type=int, default=5000, help="서버 포트 (기본: 5000)")
    server.add_argument("--host", default="0.0.0.0", help="접속 허용 주소 (기본: 0.0.0.0, 같은 네트워크에서 접속 가능)")
    server.add_argument("--directory", type=Path, default=ROOT, help="제공할 폴더 (기본: 프로젝트 폴더)")
    subparsers.add_parser("doctor", help="가상환경·패키지·GPU 확인")
    download = subparsers.add_parser("download-model", help="모델을 미리 다운로드")
    download.add_argument("--config", type=Path)
    tests = subparsers.add_parser("test", help="가상환경에서 테스트 실행 (모델 불필요)")
    tests.add_argument("pytest_args", nargs=argparse.REMAINDER)
    demo = subparsers.add_parser("make-demo", help="추론 점검용 짧은 합성 영상 생성")
    demo.add_argument("--output", type=Path, default=ROOT / "outputs/demo.mp4")
    demo.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        if args.command == "first-round-visualize":
            from .first_round_viewer import generate_first_round_viewer
            page = generate_first_round_viewer(args.result, output=args.output, video=args.video)
            print(f"1차 후보 시각화 생성 완료: {page}", flush=True)
            if not args.generate_only:
                from .viewer_server import serve
                serve(ROOT, args.port, args.host, home=page)
            return 0
        if args.command == "serve":
            from .viewer_server import serve
            serve(args.directory, args.port, args.host)
            return 0
        if args.command == "visualize":
            from .viewer import generate_viewer
            page = generate_viewer(args.result, output=args.output, video=args.video)
            print(f"시각화 생성 완료: {page}\n브라우저에서 열기: {page.as_uri()}")
            return 0
        if args.command == "doctor":
            return doctor()
        if args.command == "test":
            return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), *args.pytest_args], cwd=ROOT)
        if args.command == "make-demo":
            from .demo import make_demo
            if args.duration <= 0:
                parser.error("--duration은 0보다 커야 합니다.")
            if args.output.exists():
                raise ValueError(f"이미 있는 파일입니다. 다른 --output 경로를 지정하세요: {args.output}")
            make_demo(args.output, args.duration)
            print(args.output.resolve())
            return 0
        settings = Settings.load(args.config)
        if args.command == "download-model":
            from huggingface_hub import snapshot_download
            print(snapshot_download(settings.model_id, revision=settings.model_revision,
                                    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]))
            return 0
        from .pipeline import analyze
        output = (args.output or ROOT / "outputs" / args.video.stem).resolve()
        output.mkdir(parents=True, exist_ok=True)
        log = logging.FileHandler(output / "run.log", encoding="utf-8")
        log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(log)
        result = analyze(args.video, output, settings, limit_seconds=args.limit_seconds, fail_fast=args.fail_fast)
        if result["video"].get("error_count", 0):
            print(f"오류 {result['video']['error_count']}건을 기록하고 나머지 분석을 완료했습니다: {output / 'analysis_errors.json'}")
        print(f"결과: {output / 'timeline.json'}\n영상과 함께 보기: {output / 'timeline.html'}")
        return 0
    except KeyboardInterrupt:
        print("\n중단했습니다. 같은 명령을 실행하면 마지막 완료 구간부터 이어집니다.", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.error("%s", exc)
        return 1
