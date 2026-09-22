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
