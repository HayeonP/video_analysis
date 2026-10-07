"""Compare independent Qwen-on/off outputs on exactly the same adaptive timeline."""
from collections import Counter
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw
from low_freq_event.storage import atomic_json, read_json
from media import VideoWriter
from entity_output import font


def compare_results(root,with_result,without_result):
    root=Path(root).resolve()
    folders=[root/'without_qwen',root/'with_qwen']
    samples=[read_json(folder/'samples.json') for folder in folders]
    timelines=[read_json(folder/'timeline.json') for folder in folders]
    if samples[0]!=samples[1] or timelines[0]!=timelines[1]:
        raise ValueError('두 실행의 입력 샘플/타임라인이 다르므로 비교할 수 없습니다.')
    proposals=[read_json(folder/'yolo_proposals.json') for folder in folders]
    if proposals[0]!=proposals[1]: raise ValueError('두 실행의 YOLO 후보가 다릅니다.')
    details={}
    for folder in folders:
        observations=read_json(folder/'observations.json')['observations']
        state=read_json(folder/'identity_before_review.json')
        review=read_json(folder/'identity_review.json')
        details[folder.name]=dict(
            entities=read_json(folder/'entities.json')['entity_count'],
            detections=len(observations),
            frames_with_detections=len({o['frame_key'] for o in observations}),
            detection_classes=dict(Counter(o['detector_class'] for o in observations)),
            grounding_sources=dict(Counter(o['grounding_source'] for o in observations)),
            ids_before_qwen_review=len({a['entity_id'] for a in state['assignments'].values()}),
            qwen_id_merges=len(review['joins']),
            qwen_merge_methods=dict(Counter(j['method'] for j in review['joins'])),
            result_dir=folder.name,
            video=f'{folder.name}/adaptive_entities_full.mp4',
            gallery=f'{folder.name}/entity_gallery.jpg')
    timeline=timelines[0];fps=Fraction(timeline['fps']);frames=samples[0]
    holds={f['full_frame']:f['hold_frames'] for f in frames}
    width=960;height=round(timeline['size'][1]*width/timeline['size'][0]/2)*2
    pending=root/'.qwen_comparison_full.mp4'
    writer=VideoWriter(pending,2*width,height+42,fps,holds,crf=20)
    containers=[av.open(str(folder/'adaptive_entities_full.mp4')) for folder in folders]
    decoders=[iter(c.decode(video=0)) for c in containers]
    title_font=font(21)
    try:
        for sample in frames:
            images=[]
            for decoder,label in zip(decoders,['Qwen OFF | IDs independent','Qwen ON | IDs independent']):
                frame=next(decoder)
                if frame.pts*frame.time_base!=sample['full_frame']/fps:
                    raise ValueError('두 시각화 영상의 프레임 시각이 다릅니다.')
                resized=cv2.resize(frame.to_ndarray(format='rgb24'),(width,height))
                image=Image.new('RGB',(width,height+42),(18,20,26))
                image.paste(Image.fromarray(resized),(0,42))
                ImageDraw.Draw(image).text((12,10),label,font=title_font,fill='white')
                images.append(np.asarray(image))
            writer.write(sample['full_frame'],np.hstack(images)[:,:,::-1].copy())
        for decoder in decoders:
            if next(decoder,None) is not None: raise ValueError('예상보다 많은 출력 프레임이 있습니다.')
    finally:
        writer.close()
        for container in containers: container.close()
    with av.open(str(pending)) as c:
        stream=c.streams.video[0]
        duration=stream.duration*stream.time_base
        stamps=[f.pts*f.time_base for f in c.decode(stream)]
    if stamps!=[f['full_frame']/fps for f in frames] or duration!=timeline['total_frames']/fps:
        raise RuntimeError('나란히 보기 영상의 시간축이 다릅니다.')
    pending.replace(root/'qwen_comparison_full.mp4')
    summary=dict(status='complete_with_warnings' if with_result.get('warnings') else 'complete',
        warnings=with_result.get('warnings',[]),same_input_frames=True,same_yolo_proposals=True,
        sampled_frames=len(frames),duration_sec=float(duration),duration_error_sec=0,
        comparison_video='qwen_comparison_full.mp4',left='without_qwen',right='with_qwen',
        identity_note='IDs are assigned independently per mode. Equal ID strings/colors do not imply the same physical entity across modes.',
        interpretation=('Entity/detection counts are not accuracy scores. Efficient ON/OFF share detector boxes and feature policy; Qwen reviews tracklet labels, rejection and identity links.' if with_result.get('pipeline')=='efficient' else 'Entity/detection counts are not accuracy scores; Qwen changes detections, descriptions, crop embeddings and final merges.'),
        results=details)
    atomic_json(root/'comparison_summary.json',summary)
    from result_report import write_run_reports
    write_run_reports(root,'both')
    print(f'비교 완료: {root / "qwen_comparison_full.mp4"}',flush=True)
    return summary
