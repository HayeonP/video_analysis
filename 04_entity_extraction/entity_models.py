"""04 adapters around the detector, appearance encoders and VLM from stage 01."""
import json
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from low_freq_event.perception import PersonEncoder


class Detection(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    kind: Literal['person', 'animal', 'object']
    category: str = Field(min_length=1, max_length=60)
    label: str = Field(min_length=1, max_length=100)
    appearance: str = Field(min_length=1, max_length=250)
    bbox: list[float] = Field(min_length=4, max_length=4)
    uncertain: bool = False

    @model_validator(mode='before')
    @classmethod
    def normalize_kind(cls, value):
        # Canonicalize an explicit subclass, without inventing detections or boxes.
        if isinstance(value, dict):
            aliases={'dragon':'animal', 'creature':'animal', 'bird':'animal',
                     'character':'person', 'human':'person', 'humanoid':'person',
                     'weapon':'object', 'staff':'object'}
            kind=value.get('kind')
            if kind in aliases:
                value=dict(value, kind=aliases[kind])
        return value

    @model_validator(mode='after')
    def coordinates(self):
        x1,y1,x2,y2 = self.bbox
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError('bbox must be positive-area xyxy in 0..1000')
        return self


class FrameDetections(BaseModel):
    model_config = ConfigDict(extra='forbid')
    frame_key: str
    detections: list[Detection]


class DetectionBatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    frames: list[FrameDetections]


class MergeGroup(BaseModel):
    ids: list[str] = Field(min_length=2)
    reason: str


class MergeReview(BaseModel):
    groups: list[MergeGroup]


DETECT_PROMPT = '''Inspect EACH labeled image independently. Return only actually visible, individually distinguishable foreground people/animated humanoids, animals/fantasy creatures, and salient physical objects (e.g. held weapons).
Do not include scenery, rooms, terrain, sky, text, subtitles, logos, body parts or clothing as separate entities. A dragon/creature is an animal even though the detector may call it bird. Humanoid animated characters are person. Detect each individual separately. Do not invent objects from dark patches or motion blur. An empty/black/title frame can have no detections.
YOLO proposals below are hints: refine, remove clear false positives, and add visible foreground entities that YOLO missed. Return the COMPLETE final list, not just additions. Give tight boxes around the visible extent of each instance, never around an entire scene. Avoid duplicate boxes for the same instance. Do not transfer a detection to a different frame just because it appears in a neighboring image.
Coordinates are [x1,y1,x2,y2] normalized 0..1000 in each supplied full image. Preserve frame_key exactly and return every frame exactly once in order. category is a short English object class (person, dragon, staff, etc.); label and appearance are concise Korean visual descriptions. Do not infer character names, intentions or identities from movie knowledge. Identity will be linked separately. uncertain=true when visual evidence is ambiguous.
JSON: {"frames":[{"frame_key":"...","detections":[{"kind":"person","category":"person","label":"짧은 머리의 인물","appearance":"갈색 옷과 짧은 머리","bbox":[10,20,300,900],"uncertain":false}]}]}
YOLO proposals:
'''


def detect_batch(backend, frames, images, proposals):
    keys = [f['frame_key'] for f in frames]
    hints = {key: proposals[key] for key in keys}
    def validate(result):
        if [f.frame_key for f in result.frames] != keys:
            raise ValueError(f'Return these frames exactly once in order: {keys}')
        return result
    result = backend.request(DETECT_PROMPT+json.dumps(hints,ensure_ascii=False), DetectionBatch,
        images=list(zip(keys,images)), validate=validate,
        max_tokens=4096, image_pixels=196608)
    return [f.model_dump() for f in result.frames]


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:(x[2]-x[0])*(x[3]-x[1])
    return overlap/max(area(a)+area(b)-overlap,1e-8)


def observations(encoder, image, frame, detections, proposals, output, *, use_qwen=True):
    # Keep trusted YOLO geometry when a VLM instance aligns with its proposal.
    records=[]
    for det in detections:
        d=dict(det)
        if not use_qwen:
            # Detector-only mode preserves accepted YOLO boxes and confidence.
            d['grounding_source']='YOLO11s'
            records.append(d)
            continue
        candidates=[p for p in proposals if (p['category']=='person')==(d['kind']=='person')]
        candidate=max(candidates,key=lambda p:iou(p['bbox'],d['bbox']),default=None)
        if candidate and iou(candidate['bbox'],d['bbox'])>=.55:
            d.update(bbox=candidate['bbox'],detection_confidence=candidate['confidence'],grounding_source='YOLO11s+Qwen3-VL')
        else:
            d.update(detection_confidence=None,grounding_source='Qwen3-VL')
        # Same-frame duplicate suppression, never identity matching across frames.
        if any(d['kind']==old['kind'] and iou(d['bbox'],old['bbox'])>.8 for old in records):
            continue
        records.append(d)
    people=[d for d in records if d['kind']=='person']
    pixel_boxes=[(np.array(d['bbox'])*np.array([image.width,image.height]*2)/1000,1.) for d in people]
    base=PersonEncoder.extract(encoder,image,frame['frame_key'],frame['original_time_sec'],output,detections=pixel_boxes) if people else []
    people.sort(key=lambda d:d['bbox'][0])
    result=[]
    for obs,d in zip(base,people):
        obs.update(kind='person',entity_kind='person',detector_class='person',label=d['label'],
            appearance=d['appearance'],detection_confidence=d['detection_confidence'],
            grounding_source=d['grounding_source'],detection_uncertain=d['uncertain'])
        result.append(obs)
    objects=[d for d in records if d['kind']!='person']
    crops=[]
    for i,d in enumerate(objects):
        x1,y1,x2,y2=np.array(d['bbox'])*np.array([image.width,image.height]*2)/1000
        crops.append(image.crop((int(x1),int(y1),max(int(x1)+1,int(np.ceil(x2))),max(int(y1)+1,int(np.ceil(y2))))))
    features=encoder.object_features(crops)
    for i,(d,crop,feature) in enumerate(zip(objects,crops,features)):
        path=f'crops/{frame["frame_key"]}_object_{i:03d}.jpg'
        (output/'crops').mkdir(exist_ok=True)
        crop.save(output/path,quality=95)
        result.append(dict(detection_id=f'{frame["frame_key"]}_object_{i:03d}',frame_key=frame['frame_key'],
            time_sec=frame['original_time_sec'],bbox=d['bbox'],crop=path,source_size=list(image.size),
            kind='object',entity_kind=d['kind'],detector_class=d['category'],label=d['label'],
            appearance=d['appearance'],object_feature=feature,detection_confidence=d['detection_confidence'],
            grounding_source=d['grounding_source'],detection_uncertain=d['uncertain']))
    for obs in result:
        obs.update(shot_id=frame['shot_id'],source_frame=frame['source_frame'],
            original_frame=frame['original_frame'],source_time_sec=frame['source_time_sec'],
            original_time_sec=frame['original_time_sec'],full_frame=frame['full_frame'],
            full_time_sec=frame['full_time_sec'],hold_frames=frame['hold_frames'],image=frame['image'])
    return result
