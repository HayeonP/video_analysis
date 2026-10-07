#!/usr/bin/env python3
"""Download the Sintel trailer from W3C and prepare its first 30 seconds."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
from urllib.request import Request, urlopen

from video_utils import probe, require_tools, sha256, write_json

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "videos/sintel_trailer_30s.mp4"
SOURCE_URL = "https://media.w3.org/2010/05/sintel/trailer_hd.mp4"
SOURCE_SHA256 = "34bbd52a4b89fdf63c8ace50b268da26653a59508288100cd3c23de276db7931"


def download_demo() -> Path:
    require_tools()
    DEMO.parent.mkdir(parents=True, exist_ok=True)
    metadata = DEMO.with_suffix(".source.json")
    if DEMO.exists():
        if not metadata.is_file():
            raise FileExistsError(f"출처 기록이 없는 파일이 있습니다: {DEMO}")
        import json
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved.get("sha256") != sha256(DEMO):
            raise ValueError(f"샘플 파일이 변경되었습니다: {DEMO}")
        print(f"기존 30초 샘플 사용: {DEMO}", flush=True)
        return DEMO
    original = DEMO.parent / "sintel_trailer_full.mp4"
    if not original.exists():
        partial = original.with_suffix(".part")
        print(f"예고편 다운로드: {SOURCE_URL}", flush=True)
        try:
            request = Request(SOURCE_URL, headers={"User-Agent": "shot-boundary-demo/1.0"})
            with urlopen(request, timeout=60) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target)
            if sha256(partial) != SOURCE_SHA256:
                raise ValueError("예고편 체크섬이 예상과 다릅니다. 다운로드 출처를 확인하세요.")
            partial.replace(original)
        finally:
            partial.unlink(missing_ok=True)
    if sha256(original) != SOURCE_SHA256:
        raise ValueError(f"원본 예고편이 변경되었습니다: {original}")
    temporary = DEMO.with_name(".sintel_trailer_30s.tmp.mp4")
    try:
        # Re-encode to cut at exactly 30 seconds, retaining the original frame rate.
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(original), "-t", "30", "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(temporary),
        ], check=True)
        info = probe(temporary)
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        duration = float(video["duration"])
        if abs(duration - 30) > 0.05:
            raise ValueError(f"샘플 영상 길이가 30초가 아닙니다: {duration}")
        details = dict(
            title="Sintel — Official Trailer (first 30 seconds)",
            source_url=SOURCE_URL, project_url="https://durian.blender.org/download/",
            attribution="© Blender Foundation | durian.blender.org",
            license="Creative Commons Attribution 3.0",
            license_url="https://durian.blender.org/sharing/",
            modification="First 30 seconds extracted; H.264/AAC re-encoded. Full trailer retained alongside sample.",
            source_file=original.name, source_sha256=SOURCE_SHA256,
            file=DEMO.name, sha256=sha256(temporary),
            source_start_sec=0, duration_sec=duration,
            width=video["width"], height=video["height"], fps=video["avg_frame_rate"],
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
        temporary.replace(DEMO)
        write_json(metadata, details)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"30초 샘플 준비 완료: {DEMO}", flush=True)
    return DEMO


if __name__ == "__main__":
    download_demo()
