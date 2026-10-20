#!/usr/bin/env python3
"""Serve an explicit allowlist of entity result artifacts, with video seeking."""
import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from viewer_process import bind_replacing_viewer

PROJECT = Path(__file__).resolve().parent
DOWNLOADS = ("entities.json", "story.json", "quality.json", "tracking_summary.json",
             "entity_shot_presence.csv", "entity_shot_presence.json", "shot_entities.json",
             "REPORT.html", "REPORT.md", "RESULT_REPORT.md", "entity_gallery.jpg", "qwen_budget.json",
             "detector_summary.json", "duplicate_observations.json", "observation_fusion.json",
             "track_fusion.json", "run_timing.json", "requested_context_validation.json",
             "context_reassignment/person.json", "context_reassignment/animal.json", "shot_memory/memory.json",
             "anchored/quality.json", "anchored/coverage.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def optional_json(path, default=None):
    """An absent or partly written optional artifact must not hide usable results."""
    if not path.is_file():
        return {} if default is None else default
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {} if default is None else default


def number(value, default=0.0):
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else default
    except (TypeError, ValueError):
        return default


def normalize_shots(timeline, story):
    story_shots = (story.get("shots") or []) if isinstance(story, dict) else []
    timeline_shots = (timeline.get("shots") or []) if isinstance(timeline, dict) else []
    indexed = {str(s.get("shot_id", s.get("id", ""))): s
               for s in story_shots if isinstance(s, dict)}
    sources = list(timeline_shots)
    seen = {str(s.get("shot_id", s.get("id", ""))) for s in sources if isinstance(s, dict)}
    sources.extend(s for key, s in indexed.items() if key not in seen)
    shots, cursor = [], 0.0
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        shot_id = str(source.get("shot_id", source.get("id", f"shot_{index + 1:04d}")))
        detail = indexed.get(shot_id, {})
        start = number(source.get("full_start_sec", source.get("start_sec", detail.get("start_sec", cursor))))
        duration = number(source.get("duration_sec"), -1)
        end = number(source.get("end_sec", detail.get("end_sec")), -1)
        if duration < 0:
            duration = max(0.0, end - start) if end >= 0 else 0.0
        shots.append({"id": shot_id, "full_start_sec": start, "duration_sec": duration,
                      "end_sec": start + duration, **{key: detail.get(key, source.get(key)) for key in
                       ("summary", "initial_summary", "events", "entities", "uncertainties", "status", "transition", "transition_evidence")}})
        cursor = start + duration
    return shots


def collect_results(root):
    root = root.resolve()
    text_guide = PROJECT / "viewer_guide.html"
    method_slides = PROJECT / "METHOD_SLIDES.html"
    current_guide = method_slides if method_slides.is_file() else text_guide
    routes = {"/": PROJECT / "viewer.html", "/guide": current_guide,
              "/slides": current_guide, "/guide-text": text_guide}
    for url, relative in (
        ("/model-flow", "MODEL_FLOW_SIMPLE.html"),
        ("/analysis-guide", "CURRENT_ANALYSIS_DETAILED_REPORT.html"),
        ("/CURRENT_ANALYSIS_DETAILED_REPORT.html", "CURRENT_ANALYSIS_DETAILED_REPORT.html"),
        ("/context-guide", "CONTEXT_ANALYSIS_REPORT.html"),
        ("/context-guide.pdf", "CONTEXT_ANALYSIS_REPORT.pdf"),
        ("/analysis-guide.pdf", "CURRENT_ANALYSIS_DETAILED_REPORT.pdf"),
        ("/model-flow.pdf", "MODEL_FLOW_SIMPLE.pdf"),
        ("/architecture", "PIPELINE_STUDY_REPORT.html"),
        ("/architecture.pdf", "PIPELINE_STUDY_REPORT.pdf"),
        ("/PIPELINE_STUDY_REPORT.pdf", "PIPELINE_STUDY_REPORT.pdf"),
        ("/assets/pipeline_study_architecture.svg", "assets/pipeline_study_architecture.svg"),
    ):
        if (PROJECT / relative).is_file():
            routes[url] = PROJECT / relative
    korean_font = PROJECT / "assets/NanumGothic-Regular.ttf"
    if korean_font.is_file():
        routes["/assets/NanumGothic-Regular.ttf"] = korean_font
    method_pdf = PROJECT / "presentation/output/pdf/METHOD_SLIDES.pdf"
    if method_pdf.is_file():
        routes["/slides.pdf"] = method_pdf
    if (PROJECT / "ENTITY_EXTRACTION_SLIDES.html").is_file():
        routes["/slides-legacy"] = PROJECT / "ENTITY_EXTRACTION_SLIDES.html"
    data = {"name": root.parent.name if root.name == "entities" else root.name, "modes": [], "comparison": None, "guide": "/guide"}
    if (root / "IDENTITY_HISTORY.html").is_file():
        routes["/identity-history"] = root / "IDENTITY_HISTORY.html"
        data["identity_history"] = "/identity-history"
    if (root / "LEGACY_IDENTITY_HISTORY.html").is_file():
        routes["/legacy-identity-history"] = root / "LEGACY_IDENTITY_HISTORY.html"
        data["legacy_identity_history"] = "/legacy-identity-history"
    if (root / "REPAIR_REPORT.html").is_file():
        routes["/repair-report"] = root / "REPAIR_REPORT.html"
        data["repair_report"] = "/repair-report"
    root_report = root / "REPORT.html"
    if root_report.is_file() and root_report.resolve().is_relative_to(root):
        routes["/guide"] = routes["/report"] = root_report
    candidates = [(name, root / name) for name in ("with_qwen", "without_qwen")]
    if (root / "entities.json").is_file():
        candidates = [(root.name, root)]
    for name, folder in candidates:
        if not (folder / "entities.json").is_file():
            continue
        folder = folder.resolve()

        def register(url, relative):
            if not isinstance(relative, str) or not relative:
                return None
            target = (folder / relative).resolve()
            if target.is_relative_to(folder) and target.is_file():
                routes[url] = target
                return url
            return None

        prefix = f"/results/{name}"
        entity_data = read_json(folder / "entities.json")
        entities = entity_data.get("entities", []) if isinstance(entity_data, dict) else entity_data
        if not isinstance(entities, list):
            entities = []
        story = optional_json(folder / "story.json")
        if not isinstance(story, dict):
            story = {"shots": story} if isinstance(story, list) else {}
        timeline = optional_json(folder / "timeline.json")
        if not isinstance(timeline, dict):
            timeline = {"shots": timeline} if isinstance(timeline, list) else {}
        shots = normalize_shots(timeline, story)
        public_entities = []
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                continue
            item = {key: entity.get(key) for key in (
                "entity_id", "label", "kind", "appearance", "observations", "shots",
                "display_intervals", "first_seen_sec", "last_seen_sec", "identity_status", "reference_key",
                "confidence", "evidence", "uncertainties", "aliases", "stable_traits")}
            item["entity_id"] = str(entity.get("entity_id", entity.get("id", f"entity_{index + 1}")))
            item["shots"] = [str(s.get("shot_id", s.get("id", ""))) if isinstance(s, dict) else str(s)
                             for s in (entity.get("shots") or [])]
            if not item["shots"]:
                item["shots"] = [shot["id"] for shot in shots if item["entity_id"] in (shot.get("entities") or [])]
            representative = entity.get("representative")
            if isinstance(representative, dict):
                representative = representative.get("path", representative.get("image"))
            item["image"] = (register(f"{prefix}/images/{index}", representative)
                             if representative else None)
            item["gallery"] = []
            gallery = entity.get("gallery") or entity.get("representatives") or entity.get("crops") or []
            if isinstance(gallery, list):
                for gallery_index, crop in enumerate(gallery):
                    crop_data = crop if isinstance(crop, dict) else {"path": crop}
                    relative = crop_data.get("path", crop_data.get("image", crop_data.get("crop")))
                    url = register(f"{prefix}/images/{index}/{gallery_index}", relative)
                    if url:
                        item["gallery"].append({"image": url, "shot_id": crop_data.get("shot_id"),
                                                "time_sec": crop_data.get("time_sec", crop_data.get("full_time_sec"))})
            public_entities.append(item)
        status = optional_json(folder / "status.json")
        if not isinstance(status, dict):
            status = {}
        warnings = status.get("warnings") or []
        warnings = list(warnings) if isinstance(warnings, list) else [str(warnings)]
        summary_path=folder/'anchored/summary.json'
        if summary_path.is_file():
            identity_summary=read_json(summary_path)
            if identity_summary.get('version',0)<3 and identity_summary.get('tracklets',0)>0 and identity_summary.get('canonical_anchors')==0:
                warnings.append('기준 대상 생성 실패로 샷 간 Qwen 매칭이 실행되지 않은 결과입니다. 표시된 미확정 ID는 서로 다른 개체를 뜻하지 않습니다.')
        report = register(f"{prefix}/report", "REPORT.html")
        if report and "/report" not in routes:
            routes["/guide"] = routes["/report"] = routes[report]
        quality = optional_json(folder / "quality.json")
        if not quality:
            quality = optional_json(folder / "anchored/quality.json")
        data["modes"].append({
            "name": name, "entities": public_entities, "shots": shots,
            "duration": number(timeline.get("duration_sec"), max((s["end_sec"] for s in shots), default=0)),
            "sampled_frames": status.get("sampled_frames", status.get("samples")),
            "analysis_status": ('complete_with_warnings' if warnings and status.get("status") == "complete"
                                else status.get("status", "unknown")),
            "warnings": warnings,
            "pipeline": status.get("pipeline", "legacy"), "qwen_generations": status.get("qwen_generations"),
            "qwen_limit": status.get("qwen_limit", 8),
            "video": register(f"{prefix}/video.mp4", "adaptive_entities_full.mp4"),
            "original_video": register(f"{prefix}/original_entities_full.mp4", "original_entities_full.mp4"),
            "story": story, "quality": quality,
            "tracking_summary": optional_json(folder / "tracking_summary.json"),
            "report": report or ("/report" if "/report" in routes else None),
            "elapsed_sec": status.get("elapsed_sec", status.get("elapsed_seconds")),
            "downloads": {file: register(f"{prefix}/{file}", file) for file in DOWNLOADS},
        })
    comparison = (root / "qwen_comparison_full.mp4").resolve()
    if comparison.is_relative_to(root) and comparison.is_file():
        data["comparison"] = "/comparison.mp4"
        routes[data["comparison"]] = comparison
    return routes, json.dumps(data, ensure_ascii=False).encode("utf-8")


def handler_for(routes, payload):
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            self.respond(head=False)

        def respond(self, head=False):
            route = urlsplit(self.path).path
            if route == "/api/data":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if not head:
                    try:
                        self.wfile.write(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                return
            path = routes.get(route)
            if path is None or not path.is_file():
                self.send_error(404)
                return
            try:
                with path.open("rb") as stream:
                    size = path.stat().st_size
                    start, end = 0, size - 1
                    requested = self.headers.get("Range")
                    if requested:
                        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
                        try:
                            if not match or not any(match.groups()):
                                raise ValueError()
                            left, right = match.groups()
                            if left:
                                start = int(left)
                                end = min(int(right), end) if right else end
                            else:
                                length = int(right)
                                if length <= 0:
                                    raise ValueError()
                                start = max(0, size - length)
                            if start > end or start >= size:
                                raise ValueError()
                        except ValueError:
                            self.send_response(416)
                            self.send_header("Content-Range", f"bytes */{size}")
                            self.send_header("Content-Length", "0")
                            self.end_headers()
                            return
                    self.send_response(206 if requested else 200)
                    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                    if mime.startswith("text/"):
                        mime += "; charset=utf-8"
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    if requested:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    if head:
                        return
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = stream.read(min(256 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def open_viewer_browser(host, port):
    """Ask the connected VS Code client to resolve and open a remote URL."""
    # localhost lets VS Code resolve the URL through its tunnel provider.
    url = f'http://localhost:{port}' if host in ('127.0.0.1', '0.0.0.0', 'localhost') else f'http://{host}:{port}'
    code = shutil.which('code')
    if not code or not os.environ.get('VSCODE_IPC_HOOK_CLI'):
        print('자동 열기: VS Code Remote 터미널에서 실행해야 합니다. 접속 주소: ' + url, flush=True)
        return
    try:
        result = subprocess.run([code, '--openExternal', url], capture_output=True,
                                text=True, timeout=20)
        if result.returncode:
            print('VS Code 브라우저 열기 요청 실패: ' + (result.stderr or result.stdout).strip(), flush=True)
        else:
            print('VS Code에 포트 전달 및 브라우저 열기를 요청했습니다. '
                  '화면이 열리지 않으면 VS Code 원격 연결을 다시 연결하세요.', flush=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f'VS Code 브라우저 열기 요청 실패: {exc}', flush=True)


def main():
    parser = argparse.ArgumentParser(description="Entity 결과 웹 뷰어 (VS Code 포트 전달 지원)")
    parser.add_argument("output", nargs="?", type=Path, help="결과 폴더; 생략 시 최근 완료한 분석")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5004)
    parser.add_argument('--open-browser', action=argparse.BooleanOptionalAction, default=False,
                        help='VS Code Remote 클라이언트에 브라우저 열기 요청')
    args = parser.parse_args()
    try:
        if args.output is None:
            latest=PROJECT/'outputs/latest_run.json'
            args.output=Path(read_json(latest)['output']) if latest.is_file() else PROJECT/'outputs'
        routes, payload = collect_results(args.output.expanduser().resolve())
        server = bind_replacing_viewer(
            ThreadingHTTPServer, (args.host, args.port), handler_for(routes, payload))
        server.daemon_threads = True
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"뷰어 시작 실패: {exc}\n포트가 사용 중이면 --port로 다른 포트를 지정하세요.\n")
    print(f"Entity viewer: http://{args.host}:{args.port}", flush=True)
    if "/architecture" in routes:
        print(f"전체 구조 학습 보고서: http://{args.host}:{args.port}/architecture", flush=True)
    print(f"VS Code → 포트(Ports) → {args.port} 전달 → 브라우저에서 열기", flush=True)
    print(f"결과: {args.output}\n결과를 다시 분석한 뒤에는 뷰어를 재시작하세요. 종료: Ctrl+C", flush=True)
    if args.open_browser:
        threading.Thread(target=open_viewer_browser, args=(args.host, args.port), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
