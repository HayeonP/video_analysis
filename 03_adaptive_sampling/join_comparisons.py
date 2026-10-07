#!/usr/bin/env python3
"""Join per-shot comparison videos in the input manifest order, without re-encoding."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile

import av

from media import probe, write_json


def join_comparisons(root, videos):
    root = Path(root).resolve()
    output = root / 'comparison_full.mp4'
    report_path = root / 'comparison_full.json'
    if output.exists() or report_path.exists():
        raise FileExistsError(f'전체 비교 영상 결과가 이미 있습니다: {output}')
    if not videos:
        raise ValueError('이어 붙일 비교 영상이 없습니다.')

    clips, segments = [], []
    total_frames, signature, fps = 0, None, None
    for row in videos:
        metadata = root / row['samples']
        meta = json.loads(metadata.read_text(encoding='utf-8'))
        if not meta.get('comparison_video'):
            raise ValueError(f'{row["id"]}: --comparison으로 샷별 비교 영상을 먼저 생성하세요.')
        path = metadata.parent / meta['comparison_video']
        stream = next(s for s in probe(path)['streams'] if s['codec_type'] == 'video')
        current_fps = Fraction(stream['avg_frame_rate'])
        current_signature = (stream['codec_name'], stream['width'], stream['height'],
                             stream['pix_fmt'], current_fps, stream['time_base'])
        if signature is None:
            signature, fps = current_signature, current_fps
        elif signature != current_signature:
            raise ValueError('전체 비교 영상은 해상도/FPS/코덱/time_base가 같은 비교 영상끼리 연결합니다.')
        count = int(meta['source_frames'])
        if int(stream['nb_frames']) != count or Fraction(stream['duration_ts']) * Fraction(stream['time_base']) != count / fps:
            raise ValueError(f'{row["id"]}: 비교 영상 길이/프레임 수가 원본과 다릅니다.')
        segments.append(dict(id=row['id'], comparison=str(path.relative_to(root)),
            start_frame=total_frames, end_frame=total_frames+count,
            start_sec=float(total_frames/fps), end_sec=float((total_frames+count)/fps),
            original_video=meta['original_video'], original_start_frame=meta['original_start_frame'],
            original_fps=meta['original_fps']))
        clips.append((path, count))
        total_frames += count

    with tempfile.TemporaryDirectory(prefix='.join-comparisons-', dir=root) as name:
        work = Path(name)
        lines = ['ffconcat version 1.0']
        # Controlled relative names avoid escaping user paths in ffconcat syntax.
        for index, (path, count) in enumerate(clips):
            link = work / f'part_{index:06d}.mp4'
            link.symlink_to(path)
            lines.extend([f"file '{link.name}'", f'duration {float(count/fps):.12f}'])
        manifest = work / 'inputs.ffconcat'
        manifest.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        pending = work / 'comparison_full.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'concat',
            '-safe', '1', '-i', str(manifest), '-map', '0:v:0', '-c:v', 'copy',
            '-an', '-movflags', '+faststart', str(pending)], check=True)
        # Validate the actual output timeline before making the result visible.
        with av.open(str(pending)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            duration = stream.duration * stream.time_base
            decoded = 0
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts * frame.time_base != index/fps:
                    raise RuntimeError('전체 비교 영상에 불연속적인 프레임 시각이 있습니다.')
                decoded += 1
        if decoded != total_frames or duration != total_frames/fps:
            raise RuntimeError('전체 비교 영상의 길이/프레임 수가 원본 합계와 다릅니다.')
        report = dict(video=output.name, order='summary.json videos order',
            duration_sec=float(duration), fps=str(fps), frames=decoded,
            audio=False, reencoded=False, exact_contiguous_timestamps=True,
            segments=segments)
        pending.rename(output)
        write_json(report_path, report)
    return dict(video=output.name, timeline=report_path.name,
                duration_sec=float(duration), frames=decoded)


def main():
    parser = argparse.ArgumentParser(description='기존 샷별 비교 영상들을 입력 순서대로 연결합니다.')
    parser.add_argument('result_dir', type=Path, help='summary.json이 있는 결과 폴더')
    args = parser.parse_args()
    try:
        summary_path = args.result_dir / 'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary['full_comparison'] = join_comparisons(args.result_dir, summary['videos'])
        write_json(summary_path, summary)
        print(f'완료: {args.result_dir / summary["full_comparison"]["video"]}')
        print(f'{summary["full_comparison"]["frames"]} frames, '
              f'{summary["full_comparison"]["duration_sec"]:.6f}s')
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'오류: {exc}\n')


if __name__ == '__main__':
    main()
