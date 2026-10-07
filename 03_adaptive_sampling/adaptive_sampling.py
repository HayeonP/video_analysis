#!/usr/bin/env python3
"""Motion/appearance-adaptive frame sampling with original-time VFR export."""
import argparse
import csv
from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import av
import cv2
import numpy as np

from sampling import Config, Selector
from media import VideoWriter, copy_audio, probe, sha256, verify_video, write_json
from join_comparisons import join_comparisons

ROOT = Path(__file__).resolve().parent
DEFAULT_LIST = ROOT / 'inputs/sintel_shots.json'


def load_inputs(list_path=None, videos=None):
    if videos:
        base, raw = Path.cwd(), [dict(path=str(p)) for p in videos]
    else:
        list_path = Path(list_path or DEFAULT_LIST).expanduser().resolve()
        base = list_path.parent
        data = json.loads(list_path.read_text(encoding='utf-8'))
        if isinstance(data, dict) and 'shots' in data:
            raw = [dict(id=Path(r['file']).stem, path=r['file'], original_start_frame=r['start_frame'],
                        original_fps=data.get('source_fps_fraction', str(data.get('fps', 24))),
                        original_video=data.get('source')) for r in data['shots']]
        else:
            raw = data.get('videos') if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise ValueError('비어 있지 않은 JSON 영상 목록이 필요합니다.')
    result, names = [], set()
    for value in raw:
        value = dict(path=value) if isinstance(value, str) else dict(value)
        path = Path(value['path']).expanduser()
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        name = value.get('id', path.stem)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.','..'} or name in names:
            raise ValueError(f'중복되거나 잘못된 영상 id: {name}')
        if not path.is_file():
            raise FileNotFoundError(path)
        names.add(name)
        result.append(dict(value, id=name, path=str(path)))
    return result


def analyze(path, config):
    info = probe(path)
    video = next(s for s in info['streams'] if s['codec_type']=='video')
    fps = Fraction(video['avg_frame_rate'])
    if fps <= 0 or video['width'] % 2 or video['height'] % 2:
        raise ValueError('양수 FPS와 짝수 해상도의 입력이 필요합니다.')
    selector = Selector(fps, config)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        first_time = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError('PTS가 없는 영상은 지원하지 않습니다.')
            stamp = frame.pts * frame.time_base
            if first_time is None:
                first_time = stamp
            if abs(float(stamp-first_time-Fraction(index,1)/fps)) > 1e-6:
                raise ValueError('입력은 CFR 영상이어야 합니다. 불규칙 PTS를 감지했습니다.')
            width = min(config.analysis_width, frame.width)
            height = max(32, round(frame.height * width / frame.width))
            small = frame.reformat(width=width, height=height, format='bgr24').to_ndarray()
            selector.push(small, index)
    indices, rows = selector.finish()
    if video.get('nb_frames', 'N/A') != 'N/A' and int(video['nb_frames']) != len(rows):
        raise RuntimeError('입력 프레임을 끝까지 디코딩하지 못했습니다.')
    return indices, rows, info, fps


def render(item, destination, indices, total, fps, info, comparison=False):
    destination.mkdir(parents=True)
    (destination/'frames').mkdir()
    video = next(s for s in info['streams'] if s['codec_type']=='video')
    boundaries = [*indices[1:], total]
    holds = {index: end-index for index,end in zip(indices,boundaries)}
    silent = destination / '.silent.mp4'
    writer = VideoWriter(silent, video['width'], video['height'], fps, holds)
    compare = None
    if comparison:
        width = min(480, video['width'])
        height = round(video['height'] * width / video['width'] / 2)*2
        compare = VideoWriter(destination/'comparison.mp4', width*2, height+28, fps, crf=23)
    thumbnails = []
    held, held_index = None, None
    seen = 0
    try:
        with av.open(item['path']) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            for index, frame in enumerate(container.decode(stream)):
                seen += 1
                if index not in holds and not comparison:
                    continue
                full = frame.to_ndarray(format='bgr24')
                if index in holds:
                    writer.write(index, full)
                    image_path = destination/'frames'/f'frame_{index:08d}.jpg'
                    if not cv2.imwrite(str(image_path),full,[cv2.IMWRITE_JPEG_QUALITY,92]):
                        raise RuntimeError(f'이미지 저장 실패: {image_path}')
                    tile = cv2.resize(full,(192,108))
                    tile = cv2.copyMakeBorder(tile,24,0,0,0,cv2.BORDER_CONSTANT)
                    cv2.putText(tile,f'f{index} / {float(Fraction(index,1)/fps):.3f}s',(3,17),0,.43,(255,255,255),1)
                    thumbnails.append(tile)
                    if comparison:
                        held, held_index = cv2.resize(full,(width,height)), index
                if comparison:
                    left = cv2.copyMakeBorder(cv2.resize(full,(width,height)),28,0,0,0,cv2.BORDER_CONSTANT)
                    right = cv2.copyMakeBorder(held,28,0,0,0,cv2.BORDER_CONSTANT)
                    cv2.putText(left,f'Original  t={float(Fraction(index,1)/fps):.3f}s',(6,20),0,.5,(255,255,255),1)
                    cv2.putText(right,f'Adaptive  sample f{held_index}  hold {float(Fraction(index-held_index,1)/fps):.2f}s',(6,20),0,.5,(255,255,255),1)
                    compare.write(index, np.hstack((left,right)))
    finally:
        writer.close()
        if compare:
            compare.close()
    if seen != total:
        raise RuntimeError('렌더링 중 원본 프레임 수가 달라졌습니다.')
    copy_audio(silent, Path(item['path']), destination/'sampled.mp4')
    blank = np.zeros_like(thumbnails[0])
    padded = thumbnails + [blank] * ((5-len(thumbnails)%5)%5)
    cv2.imwrite(str(destination/'selected_frames.jpg'),np.vstack([np.hstack(padded[i:i+5]) for i in range(0,len(padded),5)]))
    return holds


def process(item, destination, config, comparison=False):
    began = time.perf_counter()
    indices, rows, info, fps = analyze(Path(item['path']), config)
    total = len(rows)
    original_fps = Fraction(str(item.get('original_fps', fps)))
    if original_fps != fps:
        raise ValueError('원본 프레임 매핑 FPS가 샷 FPS와 다릅니다.')
    offset = int(item.get('original_start_frame',0))
    if offset < 0:
        raise ValueError('original_start_frame은 0 이상이어야 합니다.')
    holds = render(item, destination, indices, total, fps, info, comparison)
    validation = verify_video(destination/'sampled.mp4',indices,total,fps,info)
    samples = [dict(sample_id=number, source_frame=index, source_time_sec=float(Fraction(index,1)/fps),
                    original_frame=offset+index, original_time_sec=float(Fraction(offset+index,1)/fps),
                    hold_until_frame=index+holds[index], hold_duration_sec=float(Fraction(holds[index],1)/fps),
                    image=f'frames/frame_{index:08d}.jpg', reason=rows[index]['reason'])
               for number,index in enumerate(indices)]
    duration = float(Fraction(total,1)/fps)
    meta = dict(id=item['id'],source=item['path'],source_sha256=sha256(item['path']),source_fps=str(fps),
                source_frames=total,source_duration_sec=duration,original_start_frame=offset,
                original_video=item.get('original_video') or item['path'],original_fps=str(original_fps),
                sampled_frames=len(indices),effective_fps=len(indices)/duration,
                frame_reduction_fraction=1-len(indices)/total,
                output_video='sampled.mp4',output_timing='VFR: selected frame holds until the next sample; original duration retained',
                config=asdict(config),samples=samples,validation=validation,
                comparison_video='comparison.mp4' if comparison else None,
                max_hold_seconds=max(float(Fraction(v,1)/fps) for v in holds.values()),
                processing_seconds=time.perf_counter()-began)
    write_json(destination/'samples.json',meta)
    write_json(destination/'validation.json',validation)
    with (destination/'motion.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        out = csv.DictWriter(stream,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    return meta


def run_batch(items, output, config, comparison=False):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'새 결과 폴더를 지정하세요: {output}')
    for name in ('ffmpeg','ffprobe'):
        if not shutil.which(name):
            raise RuntimeError(f'{name}가 필요합니다.')
    cv2.setNumThreads(2);cv2.setRNGSeed(0)
    config.validate()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-',dir=output.parent) as name:
        staging = Path(name);results=[]
        for index,item in enumerate(items,1):
            print(f'[{index}/{len(items)}] {item["id"]} 분석·샘플링',flush=True)
            meta = process(item,staging/item['id'],config,comparison)
            results.append(dict(id=item['id'],source_frames=meta['source_frames'],sampled_frames=meta['sampled_frames'],
                duration_sec=meta['source_duration_sec'],effective_fps=meta['effective_fps'],
                frame_reduction_fraction=meta['frame_reduction_fraction'],
                video=f'{item["id"]}/sampled.mp4',samples=f'{item["id"]}/samples.json'))
            print(f'  {meta["source_frames"]} → {meta["sampled_frames"]} frames '
                  f'({meta["effective_fps"]:.2f} fps), 길이 {meta["source_duration_sec"]:.6f}s 유지',flush=True)
        source_count=sum(r['source_frames'] for r in results);sample_count=sum(r['sampled_frames'] for r in results)
        summary=dict(config=asdict(config),source_frames=source_count,sampled_frames=sample_count,
                     frame_reduction_fraction=1-sample_count/source_count,
                     total_duration_sec=sum(r['duration_sec'] for r in results),videos=results)
        if comparison:
            print('전체 비교 영상 연결', flush=True)
            summary['full_comparison'] = join_comparisons(staging, results)
        write_json(staging/'summary.json',summary)
        write_json(staging/'input_list.json',dict(videos=items))
        with (staging/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    print(f'완료: {output}\n프레임 {source_count} → {sample_count}, {100*(1-sample_count/source_count):.1f}% 감소',flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description='누적 optical flow + visual change 기반 adaptive sampling. 원본 시간 길이를 유지합니다.')
    parser.add_argument('videos',nargs='*',type=Path)
    parser.add_argument('--input-list',type=Path,help='JSON 파일 목록 또는 02 단계의 shots.json. 경로는 목록 파일 기준입니다.')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/sintel_adaptive')
    parser.add_argument('--comparison',action='store_true',help='샷별 비교 영상과 전체를 이어 붙인 comparison_full.mp4 생성 (검토용 CFR, 무음)')
    for key in ('change_budget','min_fps','max_fps','visual_guard','transient_guard'):
        parser.add_argument('--'+key.replace('_','-'),type=float,default=getattr(Config(),key))
    parser.add_argument('--analysis-width',type=int,default=256)
    args=parser.parse_args()
    if args.videos and args.input_list:
        parser.error('직접 영상 경로와 --input-list 중 하나만 사용하세요.')
    try:
        config=Config(**{key:getattr(args,key) for key in ('change_budget','min_fps','max_fps','visual_guard','transient_guard','analysis_width')})
        run_batch(load_inputs(args.input_list,args.videos),args.output,config,args.comparison)
    except (ValueError,KeyError,OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        parser.exit(1,f'오류: {exc}\n')


if __name__ == '__main__':
    main()
