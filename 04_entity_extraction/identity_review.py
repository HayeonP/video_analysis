"""Hierarchical, co-visibility-constrained visual review of appearance tracklets."""
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from low_freq_event.backend import QwenBackend
from low_freq_event.identity import canonical
from low_freq_event.detection import select_references
from low_freq_event.storage import atomic_json
from entity_models import MergeReview

LOG=logging.getLogger('entities')


class ReviewBackend(QwenBackend):
    def request(self, prompt, schema, **kwargs):
        before = self.call_number
        result = super().request(prompt, schema, **kwargs)
        LOG.info('[Qwen review] %s / %s', schema.__name__,
                 'CACHE HIT: 저장된 응답 재사용' if self.call_number == before else '신규 모델 추론 완료')
        return result

    def load(self):
        if self.model is not None: return
        if not torch.cuda.is_available(): raise RuntimeError('Identity review requires CUDA.')
        if torch.cuda.get_device_properties(0).total_memory < 16*2**30:
            super().load();return
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        LOG.info('Gallery review model: native %s, %s',dtype,torch.cuda.get_device_name(0))
        self.processor=AutoProcessor.from_pretrained(self.settings.model_id,revision=self.settings.model_revision)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(self.settings.model_id,
            revision=self.settings.model_revision,dtype=dtype,device_map={'':0},attn_implementation='sdpa').eval()


def review_identities(registry,visual,raw_all,output,backend,*,cross_shot=True):
    joins=[];reviews=[]
    if not raw_all: return joins
    def assigned(o): return canonical(registry,visual['assignments'][o['detection_id']]['entity_id'])
    def review(keys,tag,shot=None):
        ids=list(dict.fromkeys(canonical(registry,eid) for eid in keys))
        if len(ids)<2: return
        history={eid:[o for o in raw_all if assigned(o)==eid] for eid in ids}
        sets={eid:{o['frame_key'] for o in samples} for eid,samples in history.items()}
        kinds={eid:Counter(o['entity_kind'] for o in samples).most_common(1)[0][0] for eid,samples in history.items()}
        forbidden=[[a,b] for i,a in enumerate(ids) for b in ids[i+1:] if sets[a]&sets[b] or kinds[a]!=kinds[b]]
        if len(forbidden)==len(ids)*(len(ids)-1)//2: return
        # A single labeled sheet gives every crop a well-behaved image aspect ratio.
        sheet=Image.new('RGB',(768,204*((len(ids)+2)//3)),(35,35,35))
        draw=ImageDraw.Draw(sheet)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        descriptions=[]
        for index,eid in enumerate(ids):
            samples=history[eid]
            local=[o for o in samples if o['shot_id']==shot] if shot else samples
            pool=local or samples
            refs=select_references(pool,limit=4)
            refs=sorted(refs,key=lambda o:(o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]),reverse=True)[:2]
            x=(index%3)*256;y=(index//3)*204
            draw.text((x+7,y+5),eid,font=font,fill='white')
            for j,o in enumerate(refs):
                image=ImageOps.contain(Image.open(output/o['crop']).convert('RGB'),(120,168))
                sheet.paste(image,(x+4+j*126+(120-image.width)//2,y+30+(168-image.height)//2))
            descriptions.append(dict(id=eid,kind=kinds[eid],labels=list(dict.fromkeys(o['label'] for o in samples))[:3]))
        sheet_path=output/'identity_galleries'/f'{tag}.jpg';sheet_path.parent.mkdir(exist_ok=True)
        sheet.save(sheet_path,quality=95)
        def validate(result):
            used=set()
            for group in result.groups:
                if not set(group.ids)<=set(ids) or len(set(group.ids))!=len(group.ids) or used&set(group.ids):
                    raise ValueError('Use listed IDs once in disjoint groups.')
                for a,b in forbidden:
                    if {a,b}<=set(group.ids): raise ValueError(f'{a} and {b} cannot merge.')
                used.update(group.ids)
            return result
        prompt=('이미지의 각 칸은 ID 하나의 서로 다른 관찰 crop입니다. 같은 칸 안의 두 이미지는 같은 후보 ID입니다. '
            '서로 다른 칸 중 실제로 동일한 개체를 보여주는 ID만 묶으세요. 자세/시점/크기 변화는 허용하지만, '
            '같은 종류나 비슷한 색상만으로 다른 개체를 합치지 마세요. 보이는 구별 특징을 비교하세요. '
            'forbidden_pairs는 실제 동시 등장 또는 종류 불일치이므로 절대로 합치지 마세요. 불확실한 쌍은 생략하세요. '
            'JSON 하나만 간결하게 반환: {"groups":[{"ids":["person_0001","person_0002"],"reason":"시각적 근거 한 문장"}]}. '
            '병합할 것이 없으면 {"groups":[]}입니다. 반드시 제공된 ID 철자를 사용하세요.\n'
            +json.dumps(dict(entities=descriptions,forbidden_pairs=forbidden),ensure_ascii=False))
        LOG.info('Identity review %s: %d IDs',tag,len(ids))
        result=backend.request(prompt,MergeReview,images=[('ID gallery',sheet)],validate=validate,
            max_tokens=1400,image_pixels=786432)
        reviews.append(dict(tag=tag,ids=ids,groups=result.model_dump()['groups']))
        for group in result.groups:
            target=min(group.ids)
            for old in group.ids:
                if old==target: continue
                registry[old]['alias_of']=target
                registry[target]['identity_status']='matched'
                registry[target]['first_seen_sec']=min(registry[target]['first_seen_sec'],registry[old]['first_seen_sec'])
                registry[target]['last_seen_sec']=max(registry[target]['last_seen_sec'],registry[old]['last_seen_sec'])
                joins.append(dict(target=target,retired=old,reason=group.reason,method='Qwen3-VL visual gallery review',review=tag))
        atomic_json(output/'identity_review_progress.json',dict(joins=joins,reviews=reviews))
    # First consolidate nearby appearances within a shot; retain anchors between pages.
    for shot in dict.fromkeys(o['shot_id'] for o in raw_all):
        original=list(dict.fromkeys(assigned(o) for o in raw_all if o['shot_id']==shot))
        seen=[]
        for offset in range(0,len(original),6):
            anchors=list(dict.fromkeys(canonical(registry,e) for e in seen))[-4:]
            batch=anchors+original[offset:offset+6]
            review(batch,f'{shot}_{offset:03d}',shot=shot)
            seen.extend(original[offset:offset+6])
    # Then compare the reduced gallery across shots, keeping people and objects distinct.
    active=list(dict.fromkeys(assigned(o) for o in raw_all))
    pages=[active[i:i+5] for i in range(0,len(active),5)]
    if cross_shot:
        for a,b in combinations_with_replacement(range(len(pages)),2):
            review(pages[a]+pages[b],f'global_{a:02d}_{b:02d}')
    for proof in visual['assignments'].values():
        old=proof['entity_id'];proof['entity_id']=canonical(registry,old)
        if proof['entity_id']!=old:
            proof['identity_merge_method']='Qwen3-VL visual gallery review'
    return joins
