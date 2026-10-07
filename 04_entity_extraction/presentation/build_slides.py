#!/usr/bin/env python3
"""Build a portable HTML review deck from the recorded stage-04 run."""
import base64
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/sintel_sequential'
ON = RUN / 'with_qwen'
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def uri(p, mime='image/jpeg'):
    return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode('ascii')
def main():
    entities = [{k:e[k] for k in ('entity_id','label','kind','shots','observations')}
                | {'image':uri(ON/e['representative'])}
                for e in read(ON/'entities.json')['entities']]
    frames=read(ON/'samples.json')
    shots=read(ON/'timeline.json')['shots']
    events=read(ON/'sequential/forward.json')['events']
    states=[]; previous=0
    for shot in shots:
        sid=shot['id']; sf=[f for f in frames if f['shot_id']==sid]
        folder=ON/'sequential/forward'/sid
        memory=read(folder/'memory_after.json')
        result=read(folder/'result.json')
        states.append(dict(id=sid,before=previous,after=len(memory),frame_count=len(sf),
            image=uri(ON/sf[len(sf)//2]['image']),summary=result['scene_summary'],
            events=[{k:e[k] for k in ('local_id','label','decision','assigned_id','co_visible_blocked')}
                    for e in events if e['shot_id']==sid],
            memory=[dict(id=k,label=v['label']) for k,v in memory.items()]))
        previous=len(memory)
    data=dict(entities=entities,states=states,presence=read(ON/'entity_shot_presence.json'),
        comparison=read(ON/'sequential/direction_comparison.json'),
        timing=read(RUN/'timing.json'),
        pair_image=uri(ON/'sequential/final/direction_pairs/S005_001__S008_004.jpg'))
    template=(ROOT/'presentation/slides_template.html').read_text(encoding='utf-8')
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    html=template.replace('__RUN_DATA__',encoded).replace('__FONT_DATA__',uri(ROOT/'assets/NanumGothic-Regular.ttf','font/ttf'))
    target=ROOT/'ENTITY_EXTRACTION_SLIDES.html'
    target.write_text(html,encoding='utf-8')
    print(f'Created {target} ({target.stat().st_size/2**20:.1f} MiB)')
if __name__=='__main__':
    main()

