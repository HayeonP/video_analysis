from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image


@dataclass
class Frame:
    time_sec: float
    image: Image.Image
    path: str = ""


@dataclass
class Clip:
    start_sec: float
    end_sec: float
    source_fps: float
    frames: list[Frame]

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def tensor_array(self) -> np.ndarray:
        return np.stack([np.asarray(frame.image) for frame in self.frames])

    def metadata(self) -> dict:
        # Sampling is disabled downstream. Fractional indices retain exact PTS
        # for variable-frame-rate input when Qwen divides indices by source fps.
        return {
            "fps": self.source_fps,
            "total_num_frames": max(1, math.ceil(self.duration * self.source_fps)),
            "duration": self.duration,
            "frames_indices": [(frame.time_sec - self.start_sec) * self.source_fps for frame in self.frames],
        }

    def save_frames(self, output: Path) -> None:
        folder = output / "frames"
        folder.mkdir(parents=True, exist_ok=True)
        for frame in self.frames:
            frame.path = f"frames/t{round(frame.time_sec * 1_000_000):012d}.jpg"
            destination = output / frame.path
            if not destination.exists():
                frame.image.save(destination, quality=90)


def sliding_windows(duration: float, size: float, stride: float):
    """Cover every part of the video, including a short final window."""
    if duration <= 0 or size <= 0 or not 0 < stride <= size:
        raise ValueError("Expected positive duration/window and 0 < stride <= window")
    index = 0
    while True:
        start = round(index * stride, 9)
        end = min(duration, start + size)
        yield start, end
        if end >= duration - 1e-9:
            break
        index += 1


def fit_image(image: Image.Image, max_pixels: int) -> Image.Image:
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    width = max(32, int(image.width * scale) // 32 * 32)
    height = max(32, int(image.height * scale) // 32 * 32)
    return image.resize((width, height), Image.Resampling.LANCZOS)


class VideoSource:
    def __init__(self, path: Path, max_pixels: int):
        self.path = path.resolve()
        self.max_pixels = max_pixels
        with av.open(str(self.path)) as container:
            if not container.streams.video:
                raise ValueError("영상 스트림이 없습니다.")
            stream = container.streams.video[0]
            self.fps = float(stream.average_rate or stream.guessed_rate or 30)
            self.origin = float((stream.start_time or 0) * stream.time_base)
            if stream.duration is not None:
                self.duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                self.duration = float(container.duration / av.time_base)
            else:
                last = None
                for frame in container.decode(stream):
                    if frame.pts is not None:
                        last = float(frame.pts * stream.time_base) - self.origin
                if last is None:
                    raise ValueError("영상 길이를 확인할 수 없습니다.")
                self.duration = last + 1 / self.fps
            self.width, self.height = stream.width, stream.height
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("유효하지 않은 영상 길이입니다.")

    def original_images(self, clip: Clip):
        """Decode only the requested samples at source resolution for detection."""
        wanted = {round(f.time_sec * 1_000_000) for f in clip.frames}
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            container.seek(max(0, int((clip.start_sec + self.origin) / stream.time_base)), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * stream.time_base) - self.origin
                key = round(t * 1_000_000)
                if key in wanted:
                    wanted.remove(key)
                    yield key, frame.to_image().convert('RGB')
                if not wanted:
                    return
                if t > clip.end_sec:
                    break
        if wanted:
            raise ValueError(f'원본 샘플 프레임을 찾을 수 없습니다: {sorted(wanted)}')

    def sample(self, start: float, end: float, fps: float) -> Clip:
        if not 0 <= start < end <= self.duration + 1e-6 or fps <= 0:
            raise ValueError("Invalid sample interval or fps")
        targets = np.arange(start, end - 1e-9, 1 / fps)
        frames: list[Frame] = []
        target_index = 0
        last_decoded = None
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            seek = max(0, int((start + self.origin) / stream.time_base))
            container.seek(seek, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                time_sec = float(frame.pts * stream.time_base) - self.origin
                if time_sec < start - 1e-7:
                    continue
                if time_sec >= end - 1e-7:
                    break
                last_decoded = (time_sec, frame)
                if time_sec + 1e-7 < targets[target_index]:
                    continue
                frames.append(Frame(time_sec, fit_image(frame.to_image().convert("RGB"), self.max_pixels)))
                while target_index < len(targets) and targets[target_index] <= time_sec + 1e-7:
                    target_index += 1
                if target_index >= len(targets):
                    break
        if not frames and last_decoded:
            frames.append(Frame(last_decoded[0], fit_image(last_decoded[1].to_image().convert("RGB"), self.max_pixels)))
        if not frames:
            raise ValueError(f"구간 {start:.3f}~{end:.3f}초에서 프레임을 읽지 못했습니다.")
        return Clip(start, end, self.fps, frames)


def prepare_samples(source: VideoSource, output: Path, fps: float):
    """One global sampling grid shared by tracking, overlapping windows and playback."""
    from .storage import atomic_json, read_json
    manifest = output / 'samples.json'
    if manifest.exists() and (output / 'sampled.mp4').exists():
        return read_json(manifest)
    samples = []
    # VLM tensors stay small; playback is encoded separately from source pixels.
    for offset in range(0, math.ceil(source.duration * fps), 32):
        start, end = offset / fps, min(source.duration, (offset + 32) / fps)
        clip = source.sample(start, end, fps)
        clip.save_frames(output)
        for frame in clip.frames:
            index = len(samples)
            samples.append(dict(sample_index=index, time_sec=frame.time_sec,
                image=frame.path, playback_sec=index / fps))
    render_sampled_video(source, output, samples, fps)
    atomic_json(manifest, samples)
    return samples


def sampled_clip(source, output, samples, start, end):
    frames = []
    for sample in samples:
        if start - 1e-7 <= sample['time_sec'] < end - 1e-7:
            with Image.open(output / sample['image']) as image:
                frames.append(Frame(sample['time_sec'], fit_image(image.convert('RGB'), source.max_pixels), sample['image']))
    return Clip(start, end, source.fps, frames)


def render_sampled_video(source, output, samples, fps):
    """Encode exact analysis PTS at source resolution, independent of VLM pixel limits."""
    from fractions import Fraction
    import os
    import tempfile
    if not samples:
        raise ValueError('샘플이 없는 영상은 인코딩할 수 없습니다.')
    fd, name = tempfile.mkstemp(prefix='.sampled-', suffix='.mp4', dir=output)
    os.close(fd)
    temporary = Path(name)
    try:
        wanted = Clip(samples[0]['time_sec'], source.duration, source.fps,
            [Frame(s['time_sec'], None) for s in samples])
        with av.open(str(temporary), 'w') as container:
            stream = container.add_stream('libx264', rate=Fraction(str(fps)))
            stream.width = source.width + source.width % 2
            stream.height = source.height + source.height % 2
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '18', 'preset': 'medium'}
            count = 0
            for index, (key, image) in enumerate(source.original_images(wanted)):
                sample = samples[index]
                if key != round(sample['time_sec'] * 1_000_000):
                    raise ValueError('샘플 영상의 프레임 순서가 분석 시각과 다릅니다.')
                # Keep still-frame review at native resolution as well.
                path = output / sample['image']
                path.parent.mkdir(parents=True, exist_ok=True)
                replace = not path.exists()
                if not replace:
                    with Image.open(path) as existing:
                        replace = existing.size != image.size
                if replace:
                    image.save(path, quality=95)
                if image.size != (stream.width, stream.height):
                    padded = Image.new('RGB', (stream.width, stream.height))
                    padded.paste(image, (0, 0))
                    image = padded
                encoded = av.VideoFrame.from_image(image)
                encoded.pts = index
                for packet in stream.encode(encoded):
                    container.mux(packet)
                count += 1
            if count != len(samples):
                raise ValueError('샘플 영상의 프레임 수가 분석 샘플 수와 다릅니다.')
            for packet in stream.encode():
                container.mux(packet)
        temporary.replace(output / 'sampled.mp4')
    finally:
        temporary.unlink(missing_ok=True)


def ensure_sampled_video(source, output, samples, fps):
    """Repair old low-resolution viewer videos without changing analysis results."""
    path = output / 'sampled.mp4'
    if path.exists():
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) == (source.width + source.width % 2, source.height + source.height % 2):
                return
    render_sampled_video(source, output, samples, fps)
