from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from urllib.parse import quote

from .schema import Event
from .storage import atomic_json


def export_results(output: Path, video: Path, metadata: dict, events: list[Event], entities: list[dict], cuts: list[float],
                   errors: list[dict] | None = None):
    by_id = {entity["entity_id"]: entity for entity in entities}
    rows = []
    for event in events:
        row = event.model_dump()
        related = [by_id[key]["identity_status"] for key in (event.actor_id, event.object_id) if key in by_id]
        row["identity_status"] = ("unresolved" if event.unresolved_local_ids else
                                  "provisional" if "provisional" in related else "assigned")
        row["evidence_times_sec"] = [item.time_sec for item in event.evidence]
        rows.append(row)
    payload = {"schema_version": 1, "video": metadata, "events": rows,
               "entities": entities, "scene_cuts_sec": sorted(set(round(value, 3) for value in cuts)),
               "analysis_errors": errors or []}
    atomic_json(output / "timeline.json", payload)
    atomic_json(output / "entities.json", entities)
    fields = ["event_id", "start_sec", "end_sec", "event_type", "actor_id", "object_id", "action",
              "description", "state_before", "state_after", "uncertain", "identity_status", "evidence_times_sec",
              "review_reason", "unresolved_local_ids"]
    with (output / "events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe = dict(row)
            safe["evidence_times_sec"] = json.dumps(safe["evidence_times_sec"])
            safe["unresolved_local_ids"] = json.dumps(safe["unresolved_local_ids"], ensure_ascii=False)
            # Prevent model text from being interpreted as spreadsheet formulas.
            for field in fields:
                if isinstance(safe.get(field), str) and safe[field].startswith(("=", "+", "-", "@")):
                    safe[field] = "'" + safe[field]
            writer.writerow(safe)
    payload["video_url"] = quote(os.path.relpath(video, output), safe="/")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (output / "timeline.html").write_text(HTML.replace("__DATA__", data), encoding="utf-8")


HTML = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>영상 이벤트 타임라인</title><style>
body{font:15px system-ui,sans-serif;background:#10151c;color:#e6edf3;max-width:1180px;margin:auto;padding:28px}
h1{font-size:26px}video{width:100%;max-height:440px;background:#000;border-radius:12px}button,select,input{font:inherit;border:1px solid #3b485c;border-radius:7px;background:#202b3a;color:inherit;padding:9px;cursor:pointer}
.filters{display:flex;gap:12px;margin:20px 0;flex-wrap:wrap}.event{border:1px solid #344052;border-radius:12px;padding:18px;margin:12px 0}.muted{color:#a7b5c9;font-size:13px}.gallery{display:flex;gap:10px;overflow:auto;margin:12px 0}.gallery img{height:120px;max-width:220px;object-fit:contain;border-radius:5px}.entities{display:flex;gap:12px;overflow:auto}.entity{min-width:160px;max-width:240px;background:#18222e;padding:12px;border-radius:9px}.entity img{height:100px;max-width:150px;object-fit:contain}a{color:#9dcaff}summary{cursor:pointer}
</style><h1>영상 이벤트 타임라인</h1><p id="stats" class="muted"></p><details id="errors" hidden><summary id="error-summary"></summary><div id="error-list"></div></details>
<video id="video" controls preload="metadata"></video>
<div class="filters"><input id="search" placeholder="이벤트 검색"><select id="entity"><option value="">모든 대상</option></select></div>
<details><summary>등장 대상과 대표 이미지</summary><div id="entities" class="entities"></div></details>
<div id="events"></div><script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), video=document.getElementById('video');video.src=data.video_url;
const byId=Object.fromEntries(data.entities.map(e=>[e.entity_id,e]));
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function time(t){return Math.floor(t/60)+':'+(t%60).toFixed(2).padStart(5,'0')}
document.getElementById('stats').textContent=`${data.events.length}개 이벤트 · ${data.entities.length}개 대상 · 분석 길이 ${time(data.video.analyzed_duration_sec)}`;
if(data.analysis_errors?.length){document.getElementById('errors').hidden=false;document.getElementById('error-summary').textContent=`일부 분석 실패 · ${data.analysis_errors.length}건 확인 필요`;
const labels={sample:'프레임 추출 실패',scan:'영상 내용 분석 실패',localize:'대상 위치 확인 실패',identity:'대상 연결 실패',events:'이벤트 정리 실패',reconcile:'최종 대상 연결 실패'};
for(const error of data.analysis_errors){document.getElementById('error-list').append(node('p',`${time(error.start_sec)} → ${time(error.end_sec)} · ${labels[error.operation]||'구간 처리 실패'} · ${error.recovery==='skipped_window'?'해당 분석 구간을 건너뜀':'확인 가능한 결과를 보존함'}`,'muted'))}}
for(const e of data.entities){const option=node('option',e.entity_id+' · '+e.label);option.value=e.entity_id;document.getElementById('entity').append(option);
const card=node('div',undefined,'entity');if(e.gallery.length){const img=node('img');img.src=e.gallery[0];card.append(img)}card.append(node('p',e.entity_id+' · '+e.label),node('p',e.appearance,'muted'));document.getElementById('entities').append(card)}
function render(){const target=document.getElementById('events');target.replaceChildren();const query=document.getElementById('search').value.toLowerCase(),id=document.getElementById('entity').value;
for(const e of data.events){if(id&&e.actor_id!==id&&e.object_id!==id)continue;if(query&&!JSON.stringify(e).toLowerCase().includes(query))continue;
const card=node('article',undefined,'event'),jump=node('button',time(e.start_sec)+' → '+time(e.end_sec));jump.onclick=()=>{video.currentTime=e.start_sec;video.play().catch(()=>{});video.scrollIntoView({block:'center',behavior:'smooth'})};
card.append(jump,node('p',e.description),node('p',[e.event_id,e.action,e.actor_id?byId[e.actor_id]?.label:'',e.object_id?byId[e.object_id]?.label:''].filter(Boolean).join(' · '),'muted'));
if(e.identity_status==='unresolved')card.append(node('p','대상 위치를 확인하지 못한 이벤트 · 근거 프레임 확인 필요','muted'));
if(e.review_reason==='refinement_failed')card.append(node('p','정밀 분석 실패로 보존한 1차 후보 · 재확인 필요','muted'));
if(e.state_before||e.state_after)card.append(node('p',e.state_before+' → '+e.state_after,'muted'));
const details=node('details');details.append(node('summary','근거 프레임'));const gallery=node('div',undefined,'gallery');for(const frame of e.evidence){const link=node('a');link.href=frame.image;link.target='_blank';const image=node('img');image.src=frame.image;image.loading='lazy';image.title=time(frame.time_sec);link.append(image);gallery.append(link)}details.append(gallery);card.append(details);target.append(card)}
if(!target.children.length)target.append(node('p','표시할 이벤트가 없습니다.'))}
document.getElementById('search').oninput=render;document.getElementById('entity').onchange=render;render();
</script></html>"""
