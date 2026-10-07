import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

from .config import ROOT
from .storage import read_json


def latest_completed_result(outputs: Path, *, source_name: str | None = None) -> Path:
    """Choose completed analysis or tracking by completion time, not by when a viewer was regenerated."""
    candidates = []
    for timeline in outputs.glob("*/timeline.json"):
        try:
            data = read_json(timeline)
            if data.get("status") not in {"complete", "tracking_ready"}:
                continue
            if source_name and Path(data.get('video', {}).get('source', {}).get('path', '')).name != source_name:
                continue
            finished = []
            for session in data.get("timing", {}).get("sessions", []):
                if session.get("status") in {"complete", "tracking_ready"} and session.get("ended_at"):
                    stamp = datetime.fromisoformat(session["ended_at"].replace("Z", "+00:00"))
                    finished.append(stamp.replace(tzinfo=timezone.utc).timestamp() if stamp.tzinfo is None else stamp.timestamp())
            completed_at = max(finished) if finished else timeline.stat().st_mtime
            candidates.append((completed_at, str(timeline.parent), timeline.parent))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    if not candidates:
        raise ValueError(f"완료된 분석/추적 결과가 없습니다: {outputs}. ./visualize.sh <결과 폴더>로 직접 지정하세요.")
    return max(candidates)[2].resolve()


def resolve_video(source: dict, base: Path, override: Path | None):
    if override is not None:
        choices = [override.resolve()]
    else:
        choices = [(base / source.get("relative_path", "__missing__")).resolve(),
                   ROOT / "videos" / Path(source["path"]).name, Path(source["path"]), base / "video.mp4"]
    for path in choices:
        if not path.is_file():
            continue
        if path.stat().st_size != source["size"]:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]:
            return path
    raise ValueError("분석에 사용된 영상을 찾지 못했습니다. --video로 같은 영상의 새 경로를 지정하세요.")


def tracking_preview(base, data):
    """CPU-only preview from already decoded tracking frames; never loads models."""
    if data['status'] != 'tracking_ready' or not (base/'tracking'/'result.json').exists():
        return
    from fractions import Fraction
    import av
    from PIL import Image
    tracking = read_json(base/'tracking'/'result.json')
    frames = read_json(base/'tracking'/'frames.json')
    fps = tracking['tracking_fps']
    destination = base/'tracking.mp4'
    if not destination.exists():
        fd, name = tempfile.mkstemp(prefix='.tracking-',suffix='.mp4',dir=base)
        os.close(fd)
        temporary = Path(name)
        try:
            with av.open(str(temporary),'w') as container:
                stream = container.add_stream('libx264',rate=Fraction(str(fps)))
                with Image.open(base/frames[0]['image']) as first:
                    stream.width=first.width+first.width%2
                    stream.height=first.height+first.height%2
                stream.pix_fmt='yuv420p'
                stream.options={'crf':'20','preset':'fast'}
                for index,frame in enumerate(frames):
                    with Image.open(base/frame['image']) as image:
                        image=image.convert('RGB')
                        if image.size!=(stream.width,stream.height):
                            padded=Image.new('RGB',(stream.width,stream.height));padded.paste(image,(0,0));image=padded
                        encoded=av.VideoFrame.from_image(image);encoded.pts=index
                        for packet in stream.encode(encoded): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    data['tracking_video_url']='tracking.mp4'
    data['tracking_fps']=fps
    data['tracking_samples']=[dict(time_sec=f['time_sec'],playback_sec=i/fps,image=f['image']) for i,f in enumerate(frames)]
    data['tracks']=frames


def generate_viewer(base: Path, *, video: Path | None = None):
    base = base.resolve()
    data = read_json(base / "timeline.json")
    source = resolve_video(data["video"]["source"], base, video)
    destination = base / "video.mp4"
    if source != destination and (not destination.exists() or destination.stat().st_size != source.stat().st_size):
        shutil.copy2(source, destination)
    if data['video'].get('samples'):
        from .video import VideoSource, ensure_sampled_video
        ensure_sampled_video(VideoSource(source, data['video']['settings']['max_pixels']),
            base, data['video']['samples'], data['video']['settings']['sample_fps'])
    tracking_preview(base, data)
    data["video_url"] = "video.mp4"
    data["result_name"] = base.name
    if (base / "timing.json").exists():
        data["timing"] = read_json(base / "timing.json")
    template = Path(__file__).with_name("viewer.html").read_text()
    serialized = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = base / "viewer.html"
    fd, temporary = tempfile.mkstemp(prefix=".viewer-", suffix=".tmp", dir=base)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(template.replace("__DATA__", serialized))
        os.replace(temporary, page)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return page
