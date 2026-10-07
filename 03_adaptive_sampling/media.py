"""Exact-timestamp VFR output: fewer coded frames, unchanged video duration."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import av


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format',
                                              '-of','json',str(path)], text=True))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class VideoWriter:
    def __init__(self, path, width, height, fps, holds=None, crf=18):
        self.fps = Fraction(fps)
        self.time_base = 1 / self.fps
        self.holds = holds
        self.container = av.open(str(path), 'w', options={'movflags': '+faststart'})
        self.stream = self.container.add_stream('libx264', rate=self.fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.time_base = self.time_base
        self.stream.codec_context.time_base = self.time_base
        self.stream.codec_context.thread_count = 2
        self.stream.options = {'preset':'fast', 'crf':str(crf), 'bf':'0'}

    def mux(self, packet):
        frame_index = Fraction(packet.pts) * packet.time_base * self.fps
        if frame_index.denominator != 1:
            raise RuntimeError('인코더 PTS가 원본 프레임 격자와 다릅니다.')
        duration = 1 if self.holds is None else self.holds[int(frame_index)]
        ticks = Fraction(duration, 1) / self.fps / packet.time_base
        if ticks.denominator != 1:
            raise RuntimeError('샘플 유지 시간을 정확히 표현할 수 없습니다.')
        packet.duration = int(ticks)
        self.container.mux(packet)

    def write(self, index, image):
        frame = av.VideoFrame.from_ndarray(image, format='bgr24')
        frame.pts, frame.time_base = index, self.time_base
        for packet in self.stream.encode(frame):
            self.mux(packet)

    def close(self):
        try:
            for packet in self.stream.encode():
                self.mux(packet)
        finally:
            self.container.close()


def copy_audio(video, original, output):
    # Retain original audio timing; do not shorten the output to the last sampled PTS.
    subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-i',str(video),'-i',str(original),
        '-map','0:v:0','-map','1:a:0?','-c','copy','-map_metadata','1',
        '-movflags','+faststart',str(output)], check=True)
    Path(video).unlink()


def verify_video(path, indices, total_frames, fps, source_info):
    fps = Fraction(fps)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        duration = stream.duration * stream.time_base
        timestamps = [frame.pts * frame.time_base for frame in container.decode(stream)]
    expected = [Fraction(index, 1)/fps for index in indices]
    if timestamps != expected or duration != Fraction(total_frames, 1)/fps:
        raise RuntimeError(f'출력 프레임 수/PTS/재생시간 불일치: {path}')
    result_info = probe(path)
    audio_before = [s for s in source_info['streams'] if s['codec_type']=='audio']
    audio_after = [s for s in result_info['streams'] if s['codec_type']=='audio']
    if bool(audio_before) != bool(audio_after):
        raise RuntimeError('오디오 스트림이 보존되지 않았습니다.')
    format_difference = abs(float(result_info['format']['duration']) - float(source_info['format']['duration']))
    if format_difference > .002:
        raise RuntimeError(f'컨테이너 재생시간 차이가 큽니다: {format_difference:.6f}s')
    return dict(passed=True, decoded_frames=len(timestamps), expected_sampled_frames=len(indices),
                exact_sample_timestamps=True, video_duration_sec=float(duration), video_duration_error_sec=0.,
                container_duration_error_sec=format_difference, audio_preserved=bool(audio_after))
