"""Render recorded compact-review checkpoints; never rerun inference."""
import base64
import json
from pathlib import Path


def write_history(output):
    output=Path(output);folder=output/'compact_identity'
    if not (folder/'audit.json').is_file():return
    audit=json.loads((folder/'audit.json').read_text())
    ledger=json.loads((output/'qwen_budget.json').read_text())
    snapshots={int(p.stem.split('_')[-1]):json.loads(p.read_text()) for p in folder.glob('snapshot_*.json')}
    events=[e for e in ledger['events'] if e.get('phase','').startswith('global_catalog:')]
    states=[]
    for call in range(len(events)+1):
        available=[n for n in snapshots if n<=call]
        snapshot=snapshots[max(available)] if available else dict(groups={})
        event=events[call-1] if call else {}
        batch=event.get('phase','').split(':')[-1]
        images=[]
        if batch:
            for path in sorted((folder/batch).glob('evidence_*.jpg')):
                images.append('data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode())
        states.append(dict(call=call,status=event.get('status','initial'),elapsed=event.get('elapsed_sec'),
            batch=batch,groups=snapshot['groups'],images=images,
            accepted_checkpoint=call in snapshots))
    catalog_path=output/'global_entities.json'
    if catalog_path.is_file():
        catalog=json.loads(catalog_path.read_text())
        if catalog.get('compact_validation'):
            states.append(dict(call=len(events),label='최종 검증',status='deterministic validation; no model call',elapsed=None,batch='',groups={e['entity_id']:e for e in catalog['entities']},images=[],accepted_checkpoint=True))
    (output/'IDENTITY_HISTORY.html').write_text(render_history(states,audit))


def render_history(states,audit):
    data=json.dumps(dict(states=states,audit=audit),ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>전역 검토 이력</title>
<style>body{font:16px/1.6 sans-serif;max-width:1250px;margin:30px auto;padding:20px;background:#f4f6fa;color:#243449}button{padding:10px;margin:4px;cursor:pointer}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccc}img{max-width:100%;display:block;margin:20px auto}small{color:#53647a}pre{white-space:pre-wrap}button[aria-pressed=true]{background:#173c69;color:white}</style>
<h1>전역 검토: 호출별 변화</h1><p>각 버튼은 재시도를 포함한 실제 모델 호출 횟수입니다. 검증을 통과한 응답만 목록에 반영됩니다. 실패한 호출에서는 직전 목록을 유지합니다. C 번호는 검토용 그룹 ID이며 최종 G 번호와 다릅니다. 최종 검증은 추가 모델 호출 없이 추적 전환 후보를 분리한 출력 목록입니다.</p>
<div id="buttons"></div><h2 id="title"></h2><p id="status"></p><table><thead><tr><th>검토 ID</th><th>레이블</th><th>포함된 Local ID</th></tr></thead><tbody id="rows"></tbody></table><h2>이 호출에 사용한 실제 이미지</h2><div id="images"></div><details><summary>전체 검토 기록</summary><pre id="audit"></pre></details>
<script>const data=DATA;
const $=id=>document.getElementById(id);function show(i){const s=data.states[i];$('title').textContent=`${s.label||(s.call+'회 이후')} · ${Object.keys(s.groups).length}개 그룹`;$('status').textContent=`상태: ${s.status} / ${s.accepted_checkpoint?'반영된 체크포인트':'목록 변경 없음'}${s.elapsed!=null?' / '+s.elapsed.toFixed(1)+'초':''}`;$('rows').replaceChildren();for(const [id,g] of Object.entries(s.groups)){const tr=document.createElement('tr');for(const value of [id,g.label,g.local_ids.join(', ')]){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('rows').append(tr)}$('images').replaceChildren();for(const src of s.images){const img=document.createElement('img');img.src=src;img.loading='lazy';$('images').append(img)}[...$('buttons').children].forEach((b,n)=>b.setAttribute('aria-pressed',String(n===i)))}
data.states.forEach((s,i)=>{const b=document.createElement('button');b.textContent=s.label||(s.call+'회');b.onclick=()=>show(i);$('buttons').append(b)});$('audit').textContent=JSON.stringify(data.audit,null,2);show(data.states.length-1);</script>'''.replace('DATA;',data+';')
    return page


def write_legacy_history(source,destination):
    """Replay saved accepted decisions and verify the recorded final partition."""
    from copy import deepcopy
    from global_identity import apply_review,FocusedDecision
    from story_semantics import local_lookup,cannot_links
    source,destination=Path(source),Path(destination)
    read=lambda name:json.loads((source/name).read_text())
    local=read('local_entities.json');bundle=read('reviewed_tracking.json')
    audit=read('identity_reconciliation/audit.json');provisional=read('global_provisional.json')
    groups={f'C{i+1:04d}':dict(e,source_clusters=[f'C{i+1:04d}']) for i,e in enumerate(provisional['entities'])}
    entries=local_lookup(local);forbidden=cannot_links(bundle,local);objections={}
    events=[e for e in read('qwen_budget.json')['events'] if e.get('phase','').startswith('global_focused:')]
    last={e['phase']:i for i,e in enumerate(events,1)}
    decisions={r['batch']:r['decision'] for r in audit['reviews']}
    states=[dict(call=0,status='initial',elapsed=None,batch='',groups=deepcopy(groups),images=[],accepted_checkpoint=True)]
    for call,event in enumerate(events,1):
        batch=int(event['phase'].split(':')[-1])
        if call==last[event['phase']] and batch in decisions:
            groups,forbidden,_,_=apply_review(groups,FocusedDecision.model_validate(decisions[batch]),forbidden,entries,soft_objections=objections)
        if call%10==0 or call==len(events):
            states.append(dict(call=call,status='saved-decision replay',elapsed=None,batch=str(batch),groups=deepcopy(groups),images=[],accepted_checkpoint=True))
    actual=sorted(sorted(e['local_ids']) for e in read('global_entities.json')['entities'])
    replay=sorted(sorted(g['local_ids']) for g in groups.values())
    if actual!=replay:raise ValueError('Legacy replay does not match the saved final identity partition')
    record=dict(source=str(source.resolve()),model_calls=len(events),comparison_attempts=len({e['phase'] for e in events}),accepted_comparisons=audit['completed_requests'],
                final_partition_verified=True,note='Saved accepted decisions replayed; no inference. Checkpoints count actual calls including retries.')
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'LEGACY_IDENTITY_HISTORY.html').write_text(render_history(states,record).replace('전역 검토: 호출별 변화','이전 실행: 10회 단위 변화').replace('이 호출에 사용한 실제 이미지','이전 비교 이미지는 원본 identity_reconciliation 폴더에 보존되어 있습니다'))
    (destination/'legacy_history_10.json').write_text(json.dumps(dict(metadata=record,states=states),ensure_ascii=False,indent=2))
    return record
