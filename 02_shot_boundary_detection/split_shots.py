#!/usr/bin/env python3
"""Detect shot boundaries and export separate H.264/AAC MP4 clips."""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import tempfile

import cv2
import scenedetect
from scenedetect import AdaptiveDetector, ContentDetector, detect, split_video_ffmpeg

from download_demo import DEMO, download_demo
from video_utils import probe, require_tools, sha256, write_json

ROOT = Path(__file__).resolve().parent


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 유한한 숫자를 입력하세요.")
    return number


def split_shots(source: Path, output: Path, *, detector="transnet", threshold=None,
                min_shot_seconds=0.25, show_progress=True) -> dict:
    require_tools()
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"입력 영상이 없습니다: {source}")
    if output.exists():
        raise FileExistsError(f"결과 폴더가 이미 있습니다. --output으로 새 폴더를 지정하세요: {output}")
    if detector not in {"transnet", "adaptive", "content"}:
        raise ValueError(f"지원하지 않는 검출기: {detector}")
    threshold = threshold if threshold is not None else ({"transnet": 0.5, "adaptive": 3.0, "content": 27.0}[detector])
    if not math.isfinite(threshold) or threshold <= 0 or not math.isfinite(min_shot_seconds) or min_shot_seconds <= 0:
        raise ValueError("threshold와 min_shot_seconds는 0보다 큰 유한한 숫자여야 합니다.")
    if detector == "transnet" and threshold > 1:
        raise ValueError("TransNetV2 임계값은 0~1 범위여야 합니다.")
    source_info = probe(source, count_frames=True)
    if not any(s["codec_type"] == "video" for s in source_info["streams"]):
        raise ValueError("입력 파일에 비디오 스트림이 없습니다.")
    source_video = next(s for s in source_info['streams'] if s['codec_type'] == 'video')
    source_frames = int(source_video['nb_read_frames'])
    source_fps = Fraction(source_video['avg_frame_rate'])
    # Bound CPU use on the shared development machine.
    cv2.setNumThreads(2)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        staging = Path(temp)
        print(f"컷 검출: {source.name} ({detector}, threshold={threshold})", flush=True)
        if detector == "transnet":
            from gradual import detect_gradual
            scenes, detection_details = detect_gradual(source, staging / "frame_stats.csv",
                threshold=threshold, min_shot_seconds=min_shot_seconds)
        else:
            algorithm = (AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_shot_seconds)
                         if detector == "adaptive"
                         else ContentDetector(threshold=threshold, min_scene_len=min_shot_seconds))
            scenes = detect(str(source), algorithm, stats_file_path=str(staging / "frame_stats.csv"),
                            start_in_scene=True, show_progress=show_progress)
            detection_details = {}
        if not scenes:
            raise RuntimeError("디코딩된 영상 프레임이 없습니다.")
        cursor = 0
        for start, end in scenes:
            if start.frame_num != cursor or end.frame_num <= start.frame_num:
                raise RuntimeError("분할 경계에 누락/중복이 있어 저장을 중단합니다.")
            cursor = end.frame_num
        if cursor != source_frames:
            raise RuntimeError(f"원본 프레임 범위 불일치: 분할 끝 {cursor}, 원본 {source_frames}")
        rows = []
        for index, (start, end) in enumerate(scenes, 1):
            rows.append(dict(
                shot_id=index, file=f"shots/shot_{index:04d}.mp4",
                start_frame=start.frame_num, end_frame_exclusive=end.frame_num,
                frame_count=end.frame_num - start.frame_num,
                start_timecode=start.get_timecode(), end_timecode=end.get_timecode(),
                start_sec=float(Fraction(start.frame_num, 1) / source_fps), end_sec=float(Fraction(end.frame_num, 1) / source_fps),
                duration_sec=float(Fraction(end.frame_num - start.frame_num, 1) / source_fps),
            ))
        print(f"{len(scenes) - 1}개 경계 검출 → {len(scenes)}개 MP4 저장", flush=True)
        status = split_video_ffmpeg(
            str(source), scenes, output_dir=staging / "shots",
            formatter=lambda video, scene: f"shot_{scene.index + 1:04d}.mp4",
            arg_override="-map 0:v:0 -map 0:a:0? -c:v libx264 -preset fast -crf 18 "
                         "-threads 2 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart",
            show_progress=show_progress and detector != "transnet",
        )
        if status != 0:
            raise RuntimeError(f"FFmpeg 분할 실패 (종료 코드 {status})")
        verification = []
        for row in rows:
            clip = staging / row["file"]
            if not clip.is_file() or clip.stat().st_size == 0:
                raise RuntimeError(f"분할 파일이 생성되지 않았습니다: {clip}")
            info = probe(clip, count_frames=True)
            video = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
            decoded = int(video['nb_read_frames'])
            if decoded != row['frame_count'] or Fraction(video['avg_frame_rate']) != source_fps:
                raise RuntimeError(f"분할 MP4의 프레임 수/속도가 원본과 다릅니다: {clip}")
            verification.append(dict(file=row['file'], decoded_frames=decoded,
                                     video_duration_sec=float(video['duration'])))
        metadata = dict(
            source=str(source), source_sha256=sha256(source),
            created_at=datetime.now(timezone.utc).isoformat(),
            scenedetect_version=scenedetect.__version__,
            detector=detector, threshold=threshold, min_shot_seconds=min_shot_seconds,
            adaptive_min_content_val=15.0 if detector == "adaptive" else None,
            frame_index_convention="zero-based, [start_frame, end_frame_exclusive)",
            duration_sec=round(scenes[-1][1].seconds, 9),
            total_frames=scenes[-1][1].frame_num,
            shot_count=len(rows), cut_count=len(rows) - 1,
            cut_seconds=[row["start_sec"] for row in rows[1:]], shots=rows,
            exported_frame_count=sum(row["frame_count"] for row in rows),
            extraction_mode="contiguous",
            fps=float(source_fps), source_fps_fraction=str(source_fps),
            excluded_ranges=[], excluded_frame_count=0,
            timeline_mapping="source_frame = start_frame + local_frame; source_sec = start_sec + local_video_sec",
            **{k: v for k, v in detection_details.items() if k not in {"total_frames", "duration_sec", "fps", "excluded_ranges", "excluded_frame_count"}},
        )
        metadata.update({k: detection_details[k] for k in ("total_frames", "duration_sec") if k in detection_details})
        metadata['total_frames'] = source_frames
        metadata['duration_sec'] = float(Fraction(source_frames, 1) / source_fps)
        if metadata['exported_frame_count'] != source_frames:
            raise RuntimeError("저장된 전체 프레임 수가 원본과 다릅니다.")
        write_json(staging / "validation.json", dict(
            source_frames=source_frames, decoded_exported_frames=sum(v['decoded_frames'] for v in verification),
            missing_frames=0, overlapping_frames=0, all_clips_match_source_fps=True,
            source_video_duration_sec=metadata['duration_sec'], clips=verification))
        write_json(staging / "shots.json", metadata)
        with (staging / "shots.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        provenance = source.with_suffix(".source.json")
        if provenance.is_file():
            write_json(staging / "source.json", json.loads(provenance.read_text(encoding="utf-8")))
        if output.exists():
            raise FileExistsError(f"작업 중 결과 폴더가 생겼습니다: {output}")
        staging.rename(output)
    print(f"완료: {output}\n경계 목록: {output / 'shots.csv'}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="PySceneDetect로 샷 경계를 찾고 샷별 MP4를 생성합니다.")
    parser.add_argument("video", nargs="?", type=Path, help="입력 영상. 생략하면 Sintel 예고편 30초 샘플을 다운로드합니다.")
    parser.add_argument("--output", type=Path, help="새 결과 폴더. 기본: outputs/<입력 파일명>")
    parser.add_argument("--detector", choices=("transnet", "adaptive", "content"), default="transnet")
    parser.add_argument("--threshold", type=positive_float, help="검출 임계값. transnet 기본 0.5, adaptive 기본 3.0, content 기본 27.0. 낮출수록 민감합니다.")
    parser.add_argument("--min-shot-seconds", type=positive_float, default=0.25,
                        help="최소 샷 길이 기준(초), 기본 0.25. 짧은 조각은 이웃 구간에 합쳐 모든 프레임을 보존합니다.")
    args = parser.parse_args()
    try:
        # Check the destination before downloading or analyzing anything.
        source = args.video.expanduser().resolve() if args.video else DEMO
        output = (args.output or ROOT / "outputs" / source.stem).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"결과 폴더가 이미 있습니다. --output으로 새 폴더를 지정하세요: {output}")
        if args.video is None:
            source = download_demo()
        split_shots(source, output, detector=args.detector, threshold=args.threshold,
                    min_shot_seconds=args.min_shot_seconds)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"오류: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
