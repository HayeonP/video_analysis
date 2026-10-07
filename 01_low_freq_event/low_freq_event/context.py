"""The same ID/coordinate contract is used by every Qwen analysis request."""
import json


def entity_token(eid):
    for prefix, token in [('person_', 'P'), ('object_', 'O'), ('entity_', 'E')]:
        if eid.startswith(prefix) and eid[len(prefix):].isdigit():
            return token + str(int(eid[len(prefix):]))
    return eid


def entity_context(clip, entities, grounded):
    payload = {'coordinate_system': 'xyxy normalized 0..1000', 'similarity_is_probability': False,
        'entities': [{'id': e.local_id, 'token': entity_token(e.local_id), 'kind': e.kind, 'label': e.label,
                      'appearance': e.appearance} for e in entities], 'frames': []}
    for i, frame in enumerate(clip.frames):
        observations = []
        for box in grounded[i]['boxes']:
            observations.append({k: v for k, v in dict(box, token=entity_token(box['local_id'])).items()
                                 if k in {'local_id','token','bbox','detection_confidence','identity_method','identity_similarity','identity_uncertain','grounding_source'}})
        payload['frames'].append({'frame_index': grounded[i].get('frame_index', i), 'time_sec': round(frame.time_sec, 6), 'observations': observations})
    return payload


def context_prompt(clip, entities, grounded):
    return ('\n검출 및 동일성 비교에서 미리 배정한 대상 정보입니다. 인물은 이 ID를 그대로 사용하고 새로 합치거나 나누지 마세요. '
            '박스와 번호는 설명용 표시이며 영상 속 물체가 아닙니다. 클래스 이름과 유사도는 모델 추정값입니다. '
            '목록에 없는 배경·물체도 원본 영상에서 직접 확인하세요.\n'
            + json.dumps(entity_context(clip, entities, grounded), ensure_ascii=False, separators=(',', ':')))


def annotated_clip(clip, grounded):
    from PIL import ImageDraw, ImageFont
    from .video import Clip, Frame
    font = ImageFont.load_default(size=15)
    frames = []
    for frame, ground in zip(clip.frames, grounded):
        image = frame.image.copy()
        draw = ImageDraw.Draw(image)
        for box in ground['boxes']:
            if not box['local_id'].startswith(('person_', 'object_')):
                continue
            token = entity_token(box['local_id'])
            x1, y1, x2, y2 = box['bbox']
            coords = (x1*image.width/1000, y1*image.height/1000, x2*image.width/1000, y2*image.height/1000)
            color = '#ffff00' if token.startswith('P') else '#00ffff'
            draw.rectangle(coords, outline=color, width=2)
            x, y = coords[:2]
            draw.rectangle((x, y, x+45, y+18), fill='black')
            draw.text((x+1, y), token, fill=color, font=font)
        frames.append(Frame(frame.time_sec, image, frame.path))
    return Clip(clip.start_sec, clip.end_sec, clip.source_fps, frames)
