"""Self-contained browser UI, embedded into viewer.html without remote assets."""

HTML = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3R Video · 이벤트 리뷰</title>
<style>
:root{color-scheme:dark;--bg:#10151e;--panel:#182130;--line:#334158;--text:#edf2fa;--muted:#acbbd0;--accent:#91d6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,sans-serif}
header,main,footer{max-width:1600px;margin:auto;padding:20px 28px}header{display:flex;justify-content:space-between;gap:20px;align-items:center}
h1{font-size:23px;margin:0}h2{font-size:17px;margin:0 0 12px}h3{font-size:14px;margin:0 0 8px}p{margin:8px 0}
.eyebrow{color:var(--accent);font-size:11px;letter-spacing:.15em;font-weight:700}.muted{color:var(--muted);font-size:12px}.mono{font-variant-numeric:tabular-nums}
button,input,select{font:inherit;color:inherit;border:1px solid var(--line);border-radius:7px;background:#202c3f;padding:7px 11px}
button,select,summary{cursor:pointer}button:hover{border-color:var(--accent)}button:disabled{opacity:.45;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
button.primary{background:#b9e3ff;color:#112335;border-color:#b9e3ff;font-weight:650}input[type=checkbox]{accent-color:var(--accent)}input[type=range]{width:100%;padding:0;accent-color:var(--accent);cursor:pointer}
.layout{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(330px,1fr);gap:18px;align-items:start}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;min-width:0}.stack{display:grid;gap:18px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.between{justify-content:space-between}.player{position:relative;background:#000;line-height:0;border-radius:8px;overflow:hidden}
video{display:block;width:100%;height:auto}.overlay{position:absolute;inset:0;pointer-events:none;overflow:hidden}.box{position:absolute;border:2px solid var(--box-color);line-height:1.35}
.box-label{position:absolute;left:-2px;top:0;padding:2px 5px;background:var(--box-color);color:#08111c;font-size:11px;font-weight:700;max-width:250px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.controls{margin:12px 0}.badge{display:inline-block;border-radius:5px;font-size:11px;padding:2px 6px;background:#2b3c52;color:#d0e6ff}.warning{color:#ffcc80}.badge.warning{background:#4d3925;color:#ffcc80}
.notice{border:1px solid #775a34;background:#342b23;border-radius:8px;padding:10px 14px;margin-bottom:16px}.hidden,[hidden]{display:none!important}
.timeline-scroll{overflow:auto;max-height:230px;border:1px solid var(--line);border-radius:7px;margin-top:12px}.timeline{position:relative;min-height:65px;overflow:hidden;background:repeating-linear-gradient(to right,transparent 0,transparent calc(10% - 1px),#ffffff0b calc(10% - 1px),#ffffff0b 10%)}
.event-bar{position:absolute;height:22px;min-width:5px;padding:1px 5px;line-height:17px;border:1px solid #ffffff45;background:var(--event-color);color:#111c2c;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;text-align:left;border-radius:4px;font-weight:600}
.event-bar.selected{outline:2px solid white;outline-offset:1px;z-index:2}.event-bar.active{filter:brightness(1.22)}.playhead{position:absolute;top:0;bottom:0;width:2px;background:#fff;pointer-events:none;z-index:3}.cut{position:absolute;top:0;bottom:0;border-left:1px dashed #dce9ff55;pointer-events:none}.ticks{display:flex;justify-content:space-between;margin-top:4px;font-size:11px;color:var(--muted)}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;font-size:11px;color:var(--muted)}.dot{width:8px;height:8px;display:inline-block;border-radius:50%;margin-right:5px}
.filters{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}.filters input[type=search]{grid-column:1/-1;min-width:0}.filters select{min-width:0;width:100%}.filters label{grid-column:1/-1;font-size:12px;color:var(--muted)}
.event-list{height:590px;overflow:auto;padding:3px}.event-card{width:100%;display:block;text-align:left;margin-bottom:8px;padding:12px;background:#111b2a;border-left:3px solid var(--event-color)}.event-card.selected{background:#263d56;border-color:#a8dcff}.event-card.active .live{display:inline}.live{display:none;color:#b4e6bc;font-size:11px}.event-card .description{display:block;margin-top:7px;font-size:13px}.event-card .muted{display:block;margin-top:6px}
.entity-cards{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.entity-card{flex:1;min-width:200px;background:#111b2a;padding:12px;border-radius:8px;border-top:3px solid var(--box-color)}.entity-card img{float:left;width:54px;height:62px;object-fit:contain;margin:0 10px 6px 0;background:#050a11;border-radius:4px}.entity-card p{overflow-wrap:anywhere}.states{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.states div{border-left:2px solid var(--line);padding-left:10px}.frame-stage{position:relative;line-height:0;background:#060b12;border-radius:7px;overflow:hidden}.frame-stage>img{display:block;width:100%;height:auto}.frame-tools{margin:10px 0}.filmstrip{display:flex;gap:8px;overflow:auto;padding:6px 2px 10px}.thumb{padding:4px;flex:0 0 104px;background:#111b2a;font-size:11px}.thumb img{width:94px;height:56px;object-fit:contain;display:block}.thumb.selected{border-color:var(--accent);background:#29435c}.thumb.has-box{border-bottom:3px solid #91d6ff}.detail-description{font-size:17px;line-height:1.6}.error-row{border-top:1px solid var(--line);padding:9px 0}.error-row pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}.empty{padding:25px 12px;text-align:center;color:var(--muted)}
footer{color:var(--muted);font-size:12px;padding-top:0}@media(min-width:1100px){.sidebar{position:sticky;top:18px}}@media(max-width:900px){.layout{grid-template-columns:1fr}.event-list{height:350px}header,main,footer{padding:14px}.states{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div><div class="eyebrow">3R VIDEO / EVENT REVIEW</div><h1 id="title">영상 이벤트 리뷰</h1></div><div id="stats" class="muted"></div></header>
<main>
<div id="review-notice" class="notice" hidden></div>
<details id="errors" class="notice" hidden><summary id="error-summary"></summary><div id="error-list"></div></details>
<div class="layout">
<div class="stack">
<section class="panel" aria-label="동영상과 타임라인">
<div class="player" id="player"><video id="video" controls preload="metadata" playsinline></video><div id="video-boxes" class="overlay"></div></div>
<p id="media-error" class="warning" role="status" hidden>동영상을 열 수 없습니다. 원본 영상 경로와 브라우저의 코덱 지원을 확인하세요. 파일 선택으로 원본을 다시 연결할 수 있습니다.</p>
<div class="row between controls"><div class="row"><button id="previous" aria-label="이전 이벤트">← 이전</button><button id="play-event" class="primary" disabled>선택 구간 재생</button><button id="next" aria-label="다음 이벤트">다음 →</button></div><span id="clock" class="mono muted">0:00.00</span></div>
<div class="row between"><label class="muted"><input id="show-boxes" type="checkbox" checked> 객체 박스</label><label class="muted">재생 속도 <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label><button id="choose-video" class="muted">원본 영상 선택</button><input id="video-file" type="file" accept="video/*" hidden></div>
<p id="video-box-status" class="muted" aria-live="polite"></p>
<input id="scrubber" type="range" min="0" max="1" step="0.001" value="0" aria-label="영상 시간 이동">
<div class="timeline-scroll"><div id="timeline" class="timeline" aria-label="이벤트 구간 타임라인"><div id="playhead" class="playhead"></div></div></div>
<div id="ticks" class="ticks"></div><div id="legend" class="legend"></div>
<p class="muted">막대 클릭: 이벤트 선택 · 빈 공간 클릭: 시간 이동 · 흰 선: 재생 위치 · 점선: 장면 전환</p>
</section>
<section class="panel" id="detail" aria-label="선택 이벤트 상세"><p class="empty">이벤트를 선택하면 설명과 근거 프레임이 표시됩니다.</p></section>
</div>
<aside class="panel sidebar"><div class="row between"><h2>이벤트 목록</h2><span id="event-count" class="muted"></span></div>
<div class="filters"><input id="search" type="search" placeholder="설명·행동·대상 ID 검색" aria-label="이벤트 검색"><select id="entity-filter" aria-label="대상 필터"><option value="">모든 대상</option></select><select id="type-filter" aria-label="이벤트 종류"><option value="">모든 종류</option></select><label><input id="uncertain-only" type="checkbox"> 확인이 필요한 이벤트만</label></div>
<div id="event-list" class="event-list"></div></aside>
</div>
</main>
<footer>박스는 저장된 관찰 프레임에만 표시합니다. 프레임 사이의 위치를 추정하거나 추적하지 않습니다. 근거 프레임의 시간은 원본 영상 기준입니다.</footer>
<script type="application/json" id="data">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), video=$('video');
const events=[...data.events].sort((a,b)=>a.start_sec-b.start_sec||a.end_sec-b.end_sec);
const byId=new Map(data.entities.map(e=>[e.entity_id,e]));
const types={appear:['등장','#9acfff'],disappear:['사라짐','#beaaf3'],action:['행동','#8ddab8'],interaction:['상호작용','#ffcc87'],state_change:['상태 변화','#ffaeb7'],ongoing:['유지','#b1becf']};
const palette=['#91d6ff','#ffd38a','#c3afff','#96e3b2','#ffb8d4','#b8e2e7'];
const colors=new Map(data.entities.map((e,i)=>[e.entity_id,palette[i%palette.length]]));
const duration=Number(data.video.analyzed_duration_sec||data.video.duration_sec||Math.max(0,...events.map(e=>e.end_sec)))||1;
const frameTolerance=Math.max(0.001, Math.min(0.05,0.5/(Number(data.video.source_fps)||24)));
let visible=[...events],selected=null,frames=[],selectedFrame=null,stopAt=null,pendingSeek=null,objectUrl=null;
let frameSequence=0, cardNodes=[],barNodes=[],lastBoxKey='';
const observationMap=new Map();
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
function formatTime(t){t=Math.max(0,Number(t)||0);const centis=Math.round(t*100);return Math.floor(centis/6000)+':'+(Math.floor(centis/100)%60).toString().padStart(2,'0')+'.'+(centis%100).toString().padStart(2,'0')}
function kind(e){return types[e.event_type]||[e.event_type,'#b1becf']}
function label(id){return byId.get(id)?.label||id||'미확인'}
function related(e){return new Set([e?.actor_id,e?.object_id].filter(Boolean))}
function needsReview(e){return data.view_mode==='first_round'||e.uncertain||e.identity_status==='unresolved'||e.identity_status==='provisional'||!!e.unresolved_local_ids?.length||['refinement_failed','unlocalized_entities'].includes(e.review_reason)}
function setText(id,text){const el=$(id);if(el&&el.textContent!==text)el.textContent=text}
function validBox(b){return Array.isArray(b)&&b.length===4&&b.every(Number.isFinite)&&b[0]>=0&&b[1]>=0&&b[2]<=1000&&b[3]<=1000&&b[2]>b[0]&&b[3]>b[1]}
for(const entity of data.entities){
  for(const observation of entity.observations||[]){
    if(!Number.isFinite(observation.time_sec)||!validBox(observation.bbox))continue;
    const key=Math.round(observation.time_sec*1000000);
    if(!observationMap.has(key))observationMap.set(key,{time_sec:observation.time_sec,image:observation.frame,boxes:[]});
    const group=observationMap.get(key),signature=entity.entity_id+':'+observation.bbox.join(',');
    if(!group.boxes.some(b=>b.signature===signature))group.boxes.push({...observation,entity_id:entity.entity_id,signature});
  }
}
const observations=[...observationMap.values()].sort((a,b)=>a.time_sec-b.time_sec);
function observationAt(t,tolerance=0.00001){
  let lo=0,hi=observations.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(observations[mid].time_sec<t)lo=mid+1;else hi=mid}
  const choices=[observations[lo],observations[lo-1]].filter(Boolean).sort((a,b)=>Math.abs(a.time_sec-t)-Math.abs(b.time_sec-t));
  return choices.length&&Math.abs(choices[0].time_sec-t)<=tolerance?choices[0]:null;
}
function boxesFor(group){if(!group)return[];const ids=related(selected);return group.boxes.filter(b=>!selected||ids.has(b.entity_id))}
function drawBoxes(container,boxes){
  container.replaceChildren();if(!$('show-boxes').checked)return;
  for(const b of boxes){const [x1,y1,x2,y2]=b.bbox,rect=node('div',undefined,'box');
    rect.style.left=x1/10+'%';rect.style.top=y1/10+'%';rect.style.width=(x2-x1)/10+'%';rect.style.height=(y2-y1)/10+'%';
    rect.style.setProperty('--box-color',colors.get(b.entity_id)||palette[0]);
    rect.append(node('span',b.entity_id+' · '+label(b.entity_id),'box-label'));container.append(rect);
  }
}
function updateVideoBoxes(t){
  const group=observationAt(t,frameTolerance),boxes=boxesFor(group);
  const key=(group?.time_sec??'none')+':'+(selected?.event_id??'all')+':'+$('show-boxes').checked;
  if(key!==lastBoxKey){drawBoxes($('video-boxes'),boxes);lastBoxKey=key}
  setText('video-box-status',!$('show-boxes').checked?'객체 박스 표시 꺼짐':boxes.length?`저장된 관찰 ${formatTime(group.time_sec)} · ${boxes.length}개 박스`:'현재 시점에 저장된 대상 박스가 없습니다. 아래 근거 프레임에서 박스가 있는 시점을 선택할 수 있습니다.');
}
function seek(t){
  t=Math.max(0,Math.min(Number(t)||0,duration));stopAt=null;video.pause();
  if(video.readyState>=1){video.currentTime=Math.min(t,Number.isFinite(video.duration)?video.duration:t);pendingSeek=null}else pendingSeek=t;
  updatePlayback(t);
}
function updatePlayback(t){
  if(stopAt!==null&&t>=stopAt){const end=stopAt;stopAt=null;video.pause();video.currentTime=end;t=end}
  $('scrubber').value=Math.min(t,duration);$('playhead').style.left=Math.min(t/duration*100,100)+'%';
  setText('clock',formatTime(t)+' / '+formatTime(duration));updateVideoBoxes(t);
  for(const [el,e] of [...cardNodes,...barNodes])el.classList.toggle('active',t>=e.start_sec&&t<=e.end_sec);
}
function buildTimeline(){
  const timeline=$('timeline');timeline.replaceChildren();barNodes=[];const lanes=[];
  for(const cut of data.scene_cuts_sec||[]){if(cut<0||cut>duration)continue;const mark=node('div',undefined,'cut');mark.style.left=cut/duration*100+'%';timeline.append(mark)}
  for(const e of visible){let lane=lanes.findIndex(end=>end<=e.start_sec);if(lane<0)lane=lanes.length;lanes[lane]=Math.max(e.end_sec,e.start_sec+duration*.007)+duration*.002;
    const bar=node('button',e.action||kind(e)[0],'event-bar');bar.style.left=Math.max(0,e.start_sec/duration*100)+'%';bar.style.width=Math.max(.5,(e.end_sec-e.start_sec)/duration*100)+'%';bar.style.top=10+lane*29+'px';bar.style.setProperty('--event-color',kind(e)[1]);
    bar.title=`${formatTime(e.start_sec)} → ${formatTime(e.end_sec)} · ${e.description}`;bar.setAttribute('aria-label',bar.title);bar.onclick=ev=>{ev.stopPropagation();selectEvent(e)};bar.classList.toggle('selected',e===selected);timeline.append(bar);barNodes.push([bar,e]);
  }
  timeline.style.height=Math.max(65,lanes.length*29+20)+'px';const head=node('div',undefined,'playhead');head.id='playhead';timeline.append(head);
  timeline.onclick=ev=>{const rect=timeline.getBoundingClientRect();seek((ev.clientX-rect.left)/rect.width*duration)};
}
function renderList(){
  const query=$('search').value.trim().toLocaleLowerCase(),entity=$('entity-filter').value,type=$('type-filter').value;
  visible=events.filter(e=>(!entity||e.actor_id===entity||e.object_id===entity)&&(!type||e.event_type===type)&&(!$('uncertain-only').checked||needsReview(e))&&(!query||[e.event_id,e.description,e.action,e.actor_id,e.object_id,label(e.actor_id),label(e.object_id),e.state_before,e.state_after].join(' ').toLocaleLowerCase().includes(query)));
  const list=$('event-list');list.replaceChildren();cardNodes=[];
  for(const e of visible){const card=node('button',undefined,'event-card');card.style.setProperty('--event-color',kind(e)[1]);card.dataset.eventId=e.event_id;
    const top=node('div',undefined,'row between');top.append(node('span',formatTime(e.start_sec)+' → '+formatTime(e.end_sec),'mono'),node('span',kind(e)[0],'badge'));card.append(top,node('span',e.description,'description'));
    card.append(node('span',[e.actor_id?label(e.actor_id):'',e.object_id?label(e.object_id):'',e.action].filter(Boolean).join(' · '),'muted'));
    if(needsReview(e))card.append(node('span','확인 필요','badge warning'));card.append(node('span',' ● 현재 재생 구간','live'));card.onclick=()=>selectEvent(e);list.append(card);cardNodes.push([card,e]);
  }
  if(!visible.length)list.append(node('p','조건에 맞는 이벤트가 없습니다.','empty'));
  setText('event-count',`${visible.length} / ${events.length}개`);buildTimeline();
  if(selected&&!visible.includes(selected)){selected=null;frames=[];selectedFrame=null;frameSequence++;$('detail').replaceChildren(node('p','목록에서 이벤트를 선택하세요.','empty'));$('play-event').disabled=true}
  markSelection();updatePlayback(video.currentTime);
}
function markSelection(){for(const [el,e] of [...cardNodes,...barNodes]){el.classList.toggle('selected',e===selected);el.setAttribute('aria-pressed',String(e===selected))}const index=visible.indexOf(selected);$('previous').disabled=index<=0;$('next').disabled=!visible.length||index>=visible.length-1}
function eventFrames(e){
  const grouped=new Map();for(const f of e.evidence||[])grouped.set(Math.round(f.time_sec*1000000),{...f});
  const ids=related(e);
  for(const o of observations){if(o.time_sec>=e.start_sec-0.00001&&o.time_sec<=e.end_sec+0.00001&&o.boxes.some(b=>ids.has(b.entity_id)))grouped.set(Math.round(o.time_sec*1000000),{time_sec:o.time_sec,image:o.image})}
  return [...grouped.values()].sort((a,b)=>a.time_sec-b.time_sec);
}
function renderFrame(frame,jump=false){
  selectedFrame=frame;const request=++frameSequence,img=$('evidence-image'),overlay=$('frame-boxes'),boxes=boxesFor(observationAt(frame.time_sec));
  overlay.replaceChildren();img.hidden=false;$('frame-error').hidden=true;
  img.onload=()=>{if(request===frameSequence)drawBoxes(overlay,boxes)};
  img.onerror=()=>{if(request===frameSequence){img.hidden=true;$('frame-error').hidden=false;overlay.replaceChildren()}};
  img.src=frame.image;setText('frame-title',`근거 프레임 · ${formatTime(frame.time_sec)}`);
  setText('frame-info',`${boxes.length?'저장된 대상 박스 '+boxes.length+'개':'이 프레임에는 저장된 대상 박스가 없습니다.'}${frame.time_sec<selected.start_sec-0.00001||frame.time_sec>selected.end_sec+0.00001?' · 이벤트 전후 참고 프레임':''}`);
  for(const thumb of $('filmstrip').children)thumb.classList.toggle('selected',Number(thumb.dataset.time)===frame.time_sec);
  if(jump)seek(frame.time_sec);
}
function selectEvent(e){
  selected=e;markSelection();$('play-event').disabled=false;seek(e.start_sec);const detail=$('detail');detail.replaceChildren();
  const heading=node('div',undefined,'row between');heading.append(node('h2',e.event_id+' · '+kind(e)[0]),node('span',formatTime(e.start_sec)+' → '+formatTime(e.end_sec),'mono muted'));detail.append(heading,node('p',e.description,'detail-description'),node('p',`${e.action} · ${e.stage==='coarse'?'1차 분석 후보':(data.video.settings?.refine_fps||4)+'fps 정밀 분석 결과'}`,'muted'));
  if(needsReview(e)){const messages=[];if(data.view_mode==='first_round')messages.push('1차 후보 · 정밀 검증 전');if(e.uncertain)messages.push('분석 결과 불확실');if(e.identity_status==='unresolved'||e.unresolved_local_ids?.length)messages.push('대상 ID 또는 위치 미확인');if(e.identity_status==='provisional')messages.push('동일 대상 여부 미확정');if(e.review_reason==='refinement_failed')messages.push('정밀 분석 실패로 보존한 1차 후보');if(!messages.length)messages.push('추가 확인 필요');detail.append(node('p',messages.join(' · '),'warning'))}
  const cards=node('div',undefined,'entity-cards');
  for(const [role,id] of [['행동 주체',e.actor_id],['행동 대상',e.object_id]]){if(!id)continue;const entity=byId.get(id),card=node('div',undefined,'entity-card');card.style.setProperty('--box-color',colors.get(id)||palette[0]);
    if(entity?.gallery?.length){const img=node('img');img.src=entity.gallery[0];img.alt=label(id);img.onerror=()=>{img.hidden=true};card.append(img)}card.append(node('div',role,'muted'),node('strong',label(id)),node('p',id,'mono muted'),node('p',entity?.appearance||'' ,'muted'));
    if(entity?.identity_status==='provisional')card.append(node('span','동일 대상 여부 미확정','badge warning'));cards.append(card)}
  if(cards.children.length)detail.append(cards);
  if(e.unresolved_local_ids?.length)detail.append(node('p','연결되지 않은 윈도우 내부 ID: '+e.unresolved_local_ids.join(', '),'muted'));
  if(e.state_before||e.state_after){const states=node('div',undefined,'states');for(const [name,value]of[['이전 상태',e.state_before],['이후 상태',e.state_after]]){const part=node('div');part.append(node('h3',name),node('p',value||'기록 없음','muted'));states.append(part)}detail.append(states)}
  frames=eventFrames(e);selectedFrame=null;
  if(frames.length){
    const tools=node('div',undefined,'row between frame-tools'),title=node('h3');title.id='frame-title';const buttons=node('div',undefined,'row');
    for(const [text,direction]of[['← 박스 프레임',-1],['박스 프레임 →',1]]){const button=node('button',text);button.onclick=()=>{const boxed=frames.filter(f=>boxesFor(observationAt(f.time_sec)).length);const candidates=boxed.filter(f=>direction<0?f.time_sec<selectedFrame.time_sec:f.time_sec>selectedFrame.time_sec);const next=direction<0?candidates.at(-1):candidates[0];if(next)renderFrame(next,true)};buttons.append(button)}
    tools.append(title,buttons);detail.append(tools);const stage=node('div',undefined,'frame-stage'),img=node('img');img.id='evidence-image';img.alt='저장된 근거 프레임';const overlay=node('div',undefined,'overlay');overlay.id='frame-boxes';stage.append(img,overlay);detail.append(stage);
    const error=node('p','근거 이미지 파일을 찾을 수 없습니다. frames 폴더 위치를 확인하세요.','warning');error.id='frame-error';error.hidden=true;const info=node('p',undefined,'muted');info.id='frame-info';detail.append(error,info);
    const strip=node('div',undefined,'filmstrip');strip.id='filmstrip';
    for(const frame of frames){const hasBox=boxesFor(observationAt(frame.time_sec)).length>0,thumb=node('button',undefined,'thumb'+(hasBox?' has-box':''));thumb.dataset.time=frame.time_sec;const image=node('img');image.src=frame.image;image.alt=formatTime(frame.time_sec);image.loading='lazy';thumb.append(image,node('span',formatTime(frame.time_sec)+(hasBox?' · 박스':'')));thumb.onclick=()=>renderFrame(frame,true);strip.append(thumb)}
    detail.append(strip,node('p','프레임 클릭 시 영상도 해당 시점으로 이동합니다. 파란 밑줄은 저장된 대상 박스가 있는 프레임입니다.','muted'));
    const inside=frames.filter(f=>f.time_sec>=e.start_sec-0.00001&&f.time_sec<=e.end_sec+0.00001),firstBox=inside.find(f=>boxesFor(observationAt(f.time_sec)).length);renderFrame(firstBox||inside[0]||frames[0]);
  }else detail.append(node('p','저장된 근거 프레임이 없습니다.','empty'));
  const sources=node('details');sources.append(node('summary','분석 근거 정보'),node('p','원본 윈도우: '+(e.source_windows||[]).join(', '),'muted'));detail.append(sources);
}
function navigate(direction){const index=visible.indexOf(selected),next=visible[index+direction];if(next)selectEvent(next)}
$('previous').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);
$('play-event').onclick=()=>{if(!selected)return;seek(selected.start_sec);stopAt=Math.max(selected.end_sec,selected.start_sec+1/(Number(data.video.source_fps)||24));video.play().catch(()=>{stopAt=null;setText('video-box-status','영상을 재생할 수 없습니다. 영상 파일 또는 브라우저 재생 설정을 확인하세요.')})};
$('scrubber').max=duration;$('scrubber').oninput=()=>seek(Number($('scrubber').value));
$('speed').onchange=()=>{video.playbackRate=Number($('speed').value)};
$('show-boxes').onchange=()=>{lastBoxKey='';updateVideoBoxes(video.currentTime);if(selectedFrame)renderFrame(selectedFrame)};
for(const id of ['search','entity-filter','type-filter','uncertain-only'])$(id).addEventListener(id==='search'?'input':'change',renderList);
video.addEventListener('loadedmetadata',()=>{if(pendingSeek!==null){video.currentTime=Math.min(pendingSeek,video.duration);pendingSeek=null}$('media-error').hidden=true;updatePlayback(video.currentTime)});
video.addEventListener('error',()=>{$('media-error').hidden=false});
video.addEventListener('timeupdate',()=>updatePlayback(video.currentTime));video.addEventListener('seeked',()=>updatePlayback(video.currentTime));
// Video-frame callbacks use the presented frame timestamp, avoiding invented motion between observations.
if('requestVideoFrameCallback' in video){const tick=(_,meta)=>{updatePlayback(meta.mediaTime);video.requestVideoFrameCallback(tick)};video.requestVideoFrameCallback(tick)}
else{let ticking=false;const tick=()=>{if(video.paused){ticking=false;return}updatePlayback(video.currentTime);requestAnimationFrame(tick)};video.addEventListener('play',()=>{if(!ticking){ticking=true;requestAnimationFrame(tick)}})}
$('choose-video').onclick=()=>$('video-file').click();$('video-file').onchange=()=>{const file=$('video-file').files[0];if(!file)return;pendingSeek=selected?.start_sec||0;stopAt=null;if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(file);video.src=objectUrl;video.load()};
for(const entity of data.entities){const option=node('option',entity.entity_id+' · '+entity.label);option.value=entity.entity_id;$('entity-filter').append(option)}
for(const [key,[name,color]]of Object.entries(types)){const option=node('option',name);option.value=key;$('type-filter').append(option);const item=node('span'),dot=node('span',undefined,'dot');dot.style.background=color;item.append(dot,document.createTextNode(name));$('legend').append(item)}
for(let i=0;i<=4;i++)$('ticks').append(node('span',formatTime(duration*i/4)));
$('title').textContent=(data.view_mode==='first_round'?'1차 후보 · ':'')+data.video_name;$('stats').textContent=`${events.length}개 ${data.view_mode==='first_round'?'후보':'이벤트'} · ${data.entities.length}개 대상 · 분석 구간 ${formatTime(duration)}`;
if(data.review_notice){$('review-notice').hidden=false;$('review-notice').textContent=data.review_notice}
if(data.analysis_errors?.length){$('errors').hidden=false;$('error-summary').textContent=`분석 오류 ${data.analysis_errors.length}건 · 누락되거나 확인이 필요한 구간 보기`;
  const operations={sample:'프레임 추출',scan:'이벤트 추출',localize:'객체 위치 확인',identity:'대상 연결',events:'이벤트 정리',reconcile:'최종 대상 연결'};
  for(const e of data.analysis_errors){const row=node('div',undefined,'error-row'),jump=node('button',formatTime(e.start_sec)+' → '+formatTime(e.end_sec));jump.onclick=()=>seek(e.start_sec);row.append(jump,node('span',' '+(operations[e.operation]||e.operation)+' · '+(e.recovery==='skipped_window'?'해당 구간 분석 실패':'확인 가능한 결과 보존'),'warning'));const detail=node('details');detail.append(node('summary','오류 내용'),node('pre',e.message||''));row.append(detail);$('error-list').append(row)}
}
video.src=data.video_url;renderList();if(visible.length)selectEvent(visible[0]);
</script>
</body>
</html>'''
