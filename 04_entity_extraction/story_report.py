"""Generate an evidence-specific Korean report without inference or budget use."""
from __future__ import annotations
import base64
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path

# Logical roles stay stable; actual model names are read from the run records.
STEPS = [
    ('관측 수집','원본에서 박스를 연결','YOLO11x · BoT-SORT · DINOv2','원본 연속 프레임','시간순 원시 트랙',
     '한 프레임의 박스는 순간 관측입니다. 움직임과 외형으로 여러 프레임의 박스를 이어 임시 트랙을 만듭니다.',
     '전체 화면 검출에 작은 대상 타일 탐색을 더합니다. BoT-SORT가 위치·운동·DINO 외형으로 연결하고 샷 경계에서 초기화합니다. 현재 기본값은 카메라 전역 운동 보정(GMC)을 끕니다. person 같은 검출 클래스는 잠정 정보입니다.',
     '짧은 트랙 하나가 실제 개체 하나라는 뜻은 아닙니다. 배경이나 신체 일부가 후보로 섞일 수 있습니다.',
     'dense_tracking/bundle.json · tracking_summary.json'),
    ('관측 수집','각 트랙의 전경을 독립적으로 추적','SAM 2.1 video','원시 트랙의 관측 박스','프레임별 전경 마스크',
     '박스 안의 배경과 이웃 대상을 줄이기 위해, 대상의 픽셀 영역을 시간에 따라 추적합니다.',
     '실제 검출 중 품질과 시간적 다양성을 고려한 박스를 프롬프트로 선택합니다. 트랙마다 독립 세션으로 정방향·역방향 전파를 수행합니다. 관측 지지 구간·면적 급증·안정성·양방향 일치를 검사하며 다른 트랙과 겹치는 마스크도 이 단계에서는 보존합니다.',
     'SAM은 종류나 동일 인물을 판정하지 않습니다. 비어 있거나 거절된 마스크로 새 박스를 만들지 않습니다.',
     'raw_track_masks/bundle.json · summary.json'),
    ('관측 수집','중복과 혼합 가능성을 먼저 정리','마스크 그래프 · 규칙','마스크들의 시간 중첩','융합·분할 후보와 공간 근거',
     '같은 픽셀을 반복해서 가리키는 중복 후보를 줄이고, 다른 영역으로 관측된 근거를 다음 단계에 전달합니다.',
     '원본 조각 내부의 명확한 컷·검은 전환을 자동 검사하고, 원래 검출 근거 없이 그 경계를 넘은 마스크 관측을 제외합니다. 영상 프레임은 삭제하지 않습니다. 여러 프레임의 마스크 IoU로 중복을 비교합니다. 동시에 떨어져 보인 영역과 시간에 따른 대상 변경 가능성도 기록합니다. 필요한 경우 시간 구간 후보로 나누어 검토하고 모든 원시 트랙과 관측 출처를 보존합니다.',
     '공간적으로 떨어진 얼굴·몸·날개도 한 개체의 부분일 수 있습니다. 마스크 분리만으로 서로 다른 실제 개체임을 증명하지 않습니다.',
     'mask_track_graph/summary.json · shots/ · segments/'),
    ('샷 내부 정리','장면 문맥으로 샷의 등장 대상을 정리','LOCAL_MODEL','전체 장면 + 시간순 원본 crop','Local Entity · 사건 · 초기 요약',
     'VLM이 한 샷의 전체 화면과 대상별 여러 모습을 함께 보고, 어떤 후보가 같은 실제 대상인지 판단합니다.',
     '사람·동물/생물·물체·불명으로 종류를 다시 평가하고 이름과 외형 설명을 붙입니다. 끊어진 후보, 신체 일부, 붙잡기·운반 관계를 검토합니다. VLM가 소지 물체라고 분류해 놓고 사람 ID에 배정한 모순은 물체를 분리하여 보정합니다. 두 대상 모두 완전한 개체라는 판정과 반복된 동시 분리 마스크가 함께 있으면 병합 모순도 분리합니다. 얼굴·팔 같은 신체 부분은 이 규칙의 근거로 쓰지 않습니다. 각 트랙의 배정·제외·실패를 기록하고 초기 요약을 작성합니다.',
     '장면에서 보인 검출 누락 후보도 기록합니다. 텍스트만으로 실제 박스를 만들어 존재를 확정하지 않습니다.',
     'local_entities.json · reviewed_tracking.json · review_traces/'),
    ('샷 내부 정리','실제 대상마다 한 궤적으로 다시 추적','SAM 2.1 video','Local Entity의 관측 박스 모음','대상별 물리 트랙과 관측 공백',
     '여러 트랙이 한 대상이라고 정리되었다면, 그 대상의 실제 관측을 모아 마스크 추적을 다시 수행합니다.',
     'local ID별 여러 박스를 이용해 원본 프레임에서 전경을 전파합니다. source_track_ids와 원본 관측을 보존합니다. 다른 local ID가 거의 같은 전경 마스크로 붕괴하면 검출 근거로 되돌리고 경고를 남깁니다. 중첩 박스라는 이유만으로 운반자·탑승자를 합치지 않습니다.',
     '마스크가 실패한 시점은 실제 검출 박스가 있으면 fallback으로 보존하고, 검출도 없으면 빈 구간으로 남깁니다.',
     'mask_refinement/bundle.json · local.json · source_tracks.json'),
    ('샷 내부 정리','배경을 지운 외형으로 연결 후보 검색','DINOv2-small','전경만 남긴 대표 crop','샷 사이 외형 유사도 후보',
     '전경 바깥을 중립 회색으로 바꾸어 배경이나 함께 잡힌 다른 대상이 외형 비교를 지배하지 않게 합니다.',
     '여러 시점과 자세의 특징 벡터를 추출하고 코사인 유사도로 연결 후보를 제공합니다. 마스크가 없는 검출 fallback은 원본 crop임을 구분합니다. crop이 바뀌면 특징을 다시 계산합니다. 전경 특징은 앞선 geometry 후보 준비에도 사용됩니다.',
     '외형 유사도는 후보 검색 수치이며 동일 개체 확률이 아닙니다. 같은 종류의 두 생물을 자동으로 하나로 확정하지 않습니다.',
     'mask_refinement/masked_crops/ · appearance.json'),
    ('샷 간 연결','외형과 이야기를 정방향·역방향으로 검토','GLOBAL_MODEL','Local 목록 + 외형 + 샷 요약','Global Human / Other Entity',
     '샷을 가로질러 같은 대상을 묶되 앞뒤 사건과 여러 모습이 서로 맞는지 함께 확인합니다.',
     '정방향 검토가 전역 목록을 만들고 역방향 검토가 뒤에서 얻은 시각 근거로 앞쪽의 불명확한 대상을 재검토합니다. 운반·부분·동시 등장 관계도 제공합니다. 두 검토 단계의 형식 오류 재시도는 공통 호출 예산에 포함됩니다.',
     '모든 ID 쌍을 무한 반복 검토하는 Agent 루프는 없습니다. 검토 실패나 불확실성을 성공으로 치환하지 않습니다.',
     'global_entities.json · review_traces/ · qwen_budget.json'),
    ('샷 간 연결','전역 이름으로 장면 요약을 보정','NARRATIVE_MODEL','Global 목록 + 사건 순서 + 장면','보정된 요약과 사건',
     '같은 대상에게 같은 이름을 쓰면서, 누가 누구에게 무엇을 했는지 장면별로 다시 설명합니다.',
     '전역 ID와 전체 사건 순서, 장면 이미지를 함께 검토합니다. 초기 요약과 보정 요약을 모두 저장합니다. 사건은 주체·행동·상대·시각·근거 프레임을 갖고 관측·추론·불확실을 구분합니다. 최대 10샷 단위로 묶습니다. 17샷은 기본 요청 2회이며 전체 앞뒤 요약은 함께 제공합니다.',
     '보정된 요약으로 동일성을 다시 증명하는 피드백 순환은 없습니다. 영상에 없는 사건을 확정 사실로 다루면 안 됩니다.',
     'story.json · shot_entities.json · global_entities.json'),
    ('결과 확인','시간축을 보존한 영상과 검증 자료 저장','결정적 내보내기 · 독립 평가','관측 + Global ID + 샷 요약','영상 · 갤러리 · 등장표',
     '목록과 요약을 함께 확인하고 원하는 장면과 실제 관측 crop으로 바로 돌아갈 수 있게 합니다.',
     '원본 추적 박스 영상과 adaptive 선택 프레임의 유지 구간에 맞춘 영상을 저장합니다. 사람/기타 목록, 0/1 샷 등장표, 원시·전경 crop, 사건과 오류 기록을 내보냅니다. 파일·시각·ID 참조 검사와 독립 정답 평가를 구분합니다.',
     '등장표 0은 관측이 없다는 뜻입니다. 실제 영상에서 대상이 절대 없다는 보장은 아닙니다. 개체 수 감소 자체도 정확도 증거가 아닙니다.',
     'entities.json · story.json · entity_shot_presence.csv · quality.json')]


def _read(root,name):
    try:return json.loads((root/name).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return {}


def _text(value):
    return json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else str(value)


def _escape(value):return html.escape(_text(value))


def _duration(value):
    if not isinstance(value,(int,float)) or value<0:return '기록 없음'
    hours,seconds=divmod(round(value),3600);minutes,seconds=divmod(seconds,60)
    return (f'{hours}시간 ' if hours else '')+f'{minutes}분 {seconds}초'


def _table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{_escape(v)}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{_escape(v)}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def _md_table(headers,rows):
    clean=lambda v:_text(v).replace('|','\\|').replace('\n',' ')
    return ['| '+' | '.join(map(clean,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(clean,row))+' |' for row in rows]


def _overview():
    columns=[('A. 관측을 수집','#eaf1ff','#b9cff3',['원본 → YOLO · 추적','독립 SAM → 마스크 비교','중복 융합 · 시간 구간 후보'],'시각 근거와 후보'),
             ('B. 샷 안을 정리','#e8f6f0','#b6ddcd',['VLM → 대상 · 사건','SAM → 물리 트랙 보정','전경 DINO → 외형 후보'],'Local Entity 목록'),
             ('C. 샷 사이를 연결','#fff3e4','#e9cfaa',['VLM → 전역 동일성 검토','VLM → 장면 요약 보정','영상 · 등장표 · 근거 검증'],'Global Entity + 이야기')]
    parts=['<svg class="overview" viewBox="0 0 1000 306" role="img" aria-labelledby="overview-title"><title id="overview-title">관측 수집, 샷 내부 정리, 샷 간 연결의 세 단계</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#75889e"/></marker></defs>']
    for index,(title,fill,stroke,lines,result) in enumerate(columns):
        x=8+342*index
        parts+=[f'<rect x="{x}" y="12" width="300" height="280" rx="18" fill="{fill}" stroke="{stroke}"/><text x="{x+22}" y="62" font-size="23" font-weight="700" fill="#172d46">{title}</text>']
        for j,line in enumerate(lines):parts+=[f'<text x="{x+22}" y="{108+j*39}" font-size="17" fill="#334b64">{line}</text>']
        parts+=[f'<text x="{x+22}" y="253" font-size="15" fill="#536a82">산출물: {result}</text>']
    return ''.join(parts)+'<path d="M314 155H341M657 155H684" stroke="#75889e" stroke-width="2.5" marker-end="url(#arrow)"/></svg>'


def write_report(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);read=lambda name:_read(output,name)
    status=read('status.json');quality=read('quality.json');tracking=read('tracking_summary.json')
    local=read('local_entities.json');glob=read('global_entities.json');config=read('run_config.json')
    graph=read('mask_track_graph/summary.json');raw=read('raw_track_masks/summary.json');mask=read('mask_refinement/summary.json')
    raw_metrics=read('raw_track_masks/run_metrics.json');history=read('execution_history.json');ledger=read('qwen_budget.json')
    validation=read('story_validation.json');verification=read('verification_summary.json') or read('verification.json')
    work=read('work_timing.json');settings=config.get('settings') or {};stats=tracking.get('stats') or {}
    models={'LOCAL_MODEL':settings.get('model_id','Qwen3-VL · 설정 미기록'),
            'GLOBAL_MODEL':config.get('global_model_id','Qwen3-VL · 설정 미기록'),
            'NARRATIVE_MODEL':config.get('narrative_model_id',settings.get('model_id','Qwen3-VL · 설정 미기록'))}
    steps=[(*s[:2],models.get(s[2],s[2]),*s[3:]) for s in STEPS]
    shots=len(tracking.get('shots') or local.get('shots') or [])
    entities=len(glob['entities']) if 'entities' in glob else '미생성'
    calls=ledger.get('generations',status.get('qwen_generations','미기록'))
    limit=ledger.get('limit',config.get('qwen_max_calls',status.get('qwen_limit',24)))
    run_state=status.get('status','기록 없음');quality_state=quality.get('status','미평가')
    issues=list(quality.get('issues') or [])
    if status.get('error'):issues.append(status['error'])
    if not quality:issues.append('최종 품질 평가 파일이 아직 없습니다.')
    measured=[['샷',shots or '미기록'],['원본 처리 프레임',stats.get('dense_frames','미기록')],
              ['원시 추적 조각',stats.get('tracks','미기록')],['그래프 전 후보',graph.get('input_tracks','미기록')],
              ['그래프 후 후보',graph.get('output_tracks','미기록')],['독립 SAM 마스크 관측',raw.get('mask_frames','미기록')],
              ['독립 SAM 검출 fallback',raw.get('detector_fallback_frames','미기록')],
              ['검토된 local 대상',sum(len(s.get('entities',[])) for s in local.get('shots',[])) if local else '미생성'],
              ['최종 SAM 마스크 관측',mask.get('mask_frames','미기록')],['전역 Entity',entities]]
    model_rows=[[config.get('detector','yolo11x.pt'),'원본/타일 박스 검출','연속 프레임'],
                ['BoT-SORT','운동·외형에 따라 샷 내부 연결','GMC 기본값: 없음'],
                ['SAM 2.1 hiera-small','원시 마스크·local 대상 마스크','의미 동일성 판단 없음'],
                ['DINOv2-small','일반 외형 특징·전경 후보 검색','사람·비인간 공통'],
                [models['LOCAL_MODEL'],'샷 내부 종류·동일성·사건','샷별 검토'],
                [models['GLOBAL_MODEL'],'샷 간 동일성','정방향·역방향 두 검토 단계'],
                [models['NARRATIVE_MODEL'],'전체 문맥을 반영한 요약','장면 이미지 재확인']]
    events=ledger.get('events') or []
    actual_models=Counter((e.get('model_id','과거 이벤트: 모델 미기록'),e.get('precision','미기록')) for e in events)
    actual_rows=[(name,precision,count) for (name,precision),count in actual_models.items()] or [('미기록','미기록','미기록')]
    phase_counts=Counter(e.get('phase','미기록').split(':')[0] for e in events)
    revisions=[['Local Qwen',settings.get('model_revision','설정 기록 없음')],
               ['Global Qwen',config.get('global_model_revision','설정 기록 없음')],
               ['SAM',((mask or raw).get('model_identity') or {}).get('revision','이 실행 기록 없음')],
               ['DINO',(read('mask_refinement/appearance.json') or read('mask_track_graph/appearance.json')).get('revision','이 실행 기록 없음')]]
    finished=[r for r in history.get('invocations',[]) if isinstance(r.get('elapsed_sec'),(int,float))]
    times=[['현재/마지막 파이프라인 호출',_duration(status.get('elapsed_sec')),'status.json; 한 번의 프로세스 실행'],
           ['기록된 종료 호출 누적',_duration(sum(r['elapsed_sec'] for r in finished)) if finished else '기록 없음',f'execution_history.json; {len(finished)}개 종료 기록'],
           ['별도 원시 SAM 전체',_duration(raw_metrics.get('total_wall_sec_including_resumed_attempt')),'중단·재개 포함 별도 실행'],
           ['원시 SAM 마지막 생성 실행',_duration(raw_metrics.get('elapsed_sec',raw.get('elapsed_sec'))),'캐시 재사용 시간과 다름']]
    if work:times.append(['개발·다운로드·검증 전체',_duration(work.get('elapsed_sec')),'work_timing.json; 분석 실행과 별도'])
    time_note='분석 프로세스, 별도 SAM 작업, 개발·다운로드·검증 전체 시간은 서로 다른 범위입니다. 기록이 겹칠 수 있어 자동 합산하지 않습니다. 진행 중 호출은 종료 시간이 아직 없을 수 있습니다.'
    quality_note='실행 상태와 식별 품질은 별도입니다. 자동 구조 검사 통과만으로 같은 인물 연결이나 사건 이해가 정확하다고 판단하지 않습니다.'
    budget_note='기본 24회 상한을 모든 Qwen 모델이 공유합니다. 17샷의 정상 요청 계획은 local 17회 + global 두 단계 2회 + narrative 2회 = 21회이며 나머지 3회는 재시도 여유입니다. 더 긴 영상의 요청 수는 달라집니다. 전역 검토는 두 단계이지만 형식 오류 재시도로 실제 생성은 2회를 넘을 수 있습니다.'
    cache_note='호출 직전에 영속 원장을 기록합니다. 실패·재시도·중단도 사용량에 남고 재실행 시 같은 원장을 이어 씁니다. 캐시 적중은 새 생성이 아닙니다. 미검토 샷을 완료로 승격하지 않습니다. Hub 모델은 commit revision을 고정하며 SAM 로컬/HF 경로는 설정·가중치의 content hash를 검증한 경우에만 캐시를 공유합니다.'
    semantics=['검출 관측: 실제 YOLO 박스가 남아 있는 프레임입니다.',
               'SAM 관측: 검사를 통과한 양의 전경 픽셀에서 박스를 계산합니다. 모델 추정이며 수동 정답 마스크는 아닙니다.',
               '검출 fallback: SAM 근거가 실패하여 원래 검출 박스를 보존했습니다. 없는 전경 마스크를 표시하지 않습니다.',
               '관측 공백: 유효 마스크와 검출이 모두 없는 시점입니다. 이전 박스를 임의로 늘려 존재를 확정하지 않습니다.',
               'Adaptive 영상: 선택 프레임과 박스를 기존 유지 구간 동안 표시합니다. 원본 모드에서 연속 추적을 확인합니다.',
               'Global ID: 현재 모델이 같은 대상으로 연결한 판단입니다. 독립 평가와 관측 crop으로 확인해야 합니다.']
    formulas='마스크 IoU = |A ∩ B| / |A ∪ B|\n외형 코사인 유사도 = (a · b) / (‖a‖ ‖b‖)\n전체 시각 = 샷 시작 시각 + 원본 프레임 번호 / FPS'
    files=[['entities.json / global_human_entities.json / global_other_entities.json','전역 목록·종류·대표 crop'],
           ['shot_entities.json / entity_shot_presence.csv','샷별 대상과 등장 여부 0/1'],
           ['story.json','초기/보정 요약·사건·전역 ID'],
           ['adaptive_entities_full.mp4 / original_entities_full.mp4','adaptive 유지 영상 / 원본 연속 추적 영상'],
           ['quality.json / story_validation.json','검토 실패·누락·참조/시간/중복 검사'],
           ['qwen_budget.json / execution_history.json','실제 생성 예산·모델·실행 시간'],
           ['REPORT.html / REPORT.md','이 실행의 동작 설명과 수치']]
    research='시간에 따른 여러 모습·시각적 연결·장면 문맥을 함께 사용하는 CAST/PPCC/MuHPC의 문제 설정을 참고했습니다. 논문 모델 전체의 재현은 아닙니다. CAL의 옷 변화 재학습, 음성 화자 연결, YuNet/SFace/OSNet은 이 기본 파이프라인에 포함하지 않습니다.'
    sources=[('SAM 2','https://github.com/facebookresearch/sam2'),('BoT-SORT','https://github.com/NirAharon/BoT-SORT'),
             ('CAST','https://arxiv.org/abs/2201.07619'),('PPCC','https://movienet.github.io/projects/eccv18person.html'),
             ('MuHPC','https://www.robots.ox.ac.uk/~vgg/publications/2021/Brown21c/brown21c.pdf')]
    generated=datetime.now(timezone.utc).isoformat(timespec='seconds')
    md=['# 샷 추적 · 전역 Entity · 이야기 리포트','',f'- 결과 폴더: `{output.name}`',f'- 문서 생성 시각: {generated}','',
        '## 1. 먼저 이해할 컨셉','',
        '한 프레임의 박스를 바로 인물 ID로 만들지 않습니다. **관측을 시간으로 연결 → 샷 안에서 실제 대상을 정리 → 전체 샷에서 같은 대상을 연결 → 이름이 일관된 이야기 작성** 순서입니다.','',
        '03 단계의 adaptive 입력 메타데이터에서 원본 샷을 찾아 추적합니다. 원본 연속 프레임으로 추적하고, adaptive 출력에는 기존 선택 프레임과 유지 시간을 사용합니다.','',
        '```text','A. 관측 수집: 원본 → YOLO/BoT-SORT → 독립 SAM → 마스크 그래프',
        'B. 샷 내부: VLM → Local Entity/초기 요약 → SAM 보정 → 전경 DINO',
        'C. 샷 사이: VLM 정방향/역방향 → Global Entity → 요약 보정 → 검증/영상','```','',
        '## 2. 이번 실행의 실제 기록','',f'- 실행 상태: **{run_state}**',f'- 품질 상태: **{quality_state}**',
        f'- 실제 Qwen 생성: **{calls} / {limit}회**. 실패·재시도·중단 이력 포함.',quality_note,'']+_md_table(['항목','실제 기록'],measured)+['','## 3. 단계별 입력과 출력','']
    for index,(group,title,model,before,after,concept,detail,caution,evidence) in enumerate(steps,1):
        md += [f'### {index:02d}. {title}','',concept,'',f'- 입력: {before}',f'- 모델/처리: {model}',f'- 출력: {after}','',detail,'',f'해석 범위: {caution}','',f'근거 파일: `{evidence}`','']
    md+=['## 4. 어떤 DL 모델을 언제 사용하는가','']+_md_table(['모델','역할','범위'],model_rows)+['',
        'GPU 메모리 추정에 따라 큰 Qwen은 NF4 4bit / BF16 compute로 적재할 수 있습니다. 실제 정밀도는 아래 호출 원장 기록을 따릅니다.','']+_md_table(['실제 생성 모델','정밀도','횟수'],actual_rows)+['','### 고정 revision','']+_md_table(['모델','기록된 revision'],revisions)+['',
        '## 5. Agent 구조와 호출 상한','',
        'Python이 역할별 순서를 제어하는 유한 파이프라인입니다. 모델·도구를 자유롭게 반복 소환하는 Agent 루프는 없습니다.','',budget_note,'',cache_note,'',f'실제 단계별 생성: {_text(dict(phase_counts))}','',
        '## 6. 관측과 수식을 읽는 방법','']+[f'- {s}' for s in semantics]+['','```text',formulas,'```','',
        '마스크 IoU는 전경 교집합/합집합의 비율로 여러 프레임의 중복을 비교합니다. 외형 코사인 유사도는 후보 검색 수치이며 동일 개체 확률이 아닙니다. 특정 작품 이름·샷 ID·대상 수를 정답 규칙으로 사용하지 않습니다. 사용자 부분 정답은 독립 평가에만 사용합니다.','',
        '## 7. 실제 소요 시간','']+_md_table(['범위','시간','기준'],times)+['',time_note,'','## 8. 품질과 남은 항목','',quality_note,'']
    md += [f'- {_text(item)}' for item in issues] or ['- issues 목록은 비어 있습니다. 독립 식별 평가 완료를 뜻하지 않습니다.']
    extra=[('구조 검사',validation),('독립 부분 정답 평가',quality.get('manual_evaluation')),('별도 검증 기록',verification),('작업 시간과 중단 이력',work)]
    for title,data in extra:
        if data:md+=['',f'### {title}','','```json',json.dumps(data,ensure_ascii=False,indent=2),'```']
    md+=['','## 9. 확인 방법과 산출물','','```bash','./visualize.sh',f'./visualize.sh outputs/{output.name}','```','',
         'VS Code에서 5004 포트를 전달합니다. 결과 페이지에서 원본/adaptive 영상, Entity 필터, 샷별 요약 이동, 등장표와 근거 crop을 함께 확인합니다. /guide는 선택한 실행의 이 리포트를 표시합니다.','']+_md_table(['파일','내용'],files)+['','## 10. 연구 원칙','',research,'']+[f'- [{name}]({url})' for name,url in sources]+['']
    markdown='\n'.join(md)
    font=Path(__file__).resolve().parent/'assets/NanumGothic-Regular.ttf'
    font_css="@font-face{font-family:ReportKorean;src:url(data:font/ttf;base64,"+base64.b64encode(font.read_bytes()).decode()+") format('truetype');font-display:swap;}" if font.is_file() else ''
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f8;color:#193049;font:16px/1.85 ReportKorean,system-ui,sans-serif}a{color:#225bbb}main{max-width:1120px;margin:auto;padding:40px 24px 80px}h1{font-size:43px;line-height:1.4;letter-spacing:-1.5px}h2{font-size:29px;line-height:1.5;margin:0 0 18px}h3{font-size:19px;margin:16px 0 8px}p{margin:12px 0}header,.panel,.step{background:white;border:1px solid #dce4ef;border-radius:22px;padding:34px;margin-bottom:24px}header{padding:44px;background:linear-gradient(130deg,#fff,#edf4ff)}.eyebrow{font:700 12px/1.6 system-ui;letter-spacing:1.6px;color:#5276a3}.lead{font-size:19px;max-width:840px;color:#3c5774}.nav{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}.nav a{background:#e9f0fd;border-radius:8px;padding:7px 12px;font-size:14px;text-decoration:none}.muted{color:#60768e;font-size:14px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric{background:#fff;border:1px solid #dce4ef;border-radius:14px;padding:18px}.metric strong{display:block;font-size:25px;overflow-wrap:anywhere}.metric small{color:#60768e}.overview{display:block;width:100%;height:auto;margin:20px 0}.step{scroll-margin-top:20px;break-inside:avoid}.step-head{display:flex;align-items:baseline;gap:16px}.number{font:800 32px system-ui;color:#2e6bc5}.concept{font-size:20px;color:#233f5c;margin:18px 0 24px}.flow{display:grid;grid-template-columns:1fr 28px 1.2fr 28px 1fr;align-items:stretch;margin:24px 0}.flow div{border:1px solid #d6e0eb;background:#f7f9fc;border-radius:12px;padding:17px;font-size:15px;overflow-wrap:anywhere}.flow .model{background:#edf4ff;border-color:#bfd0ec}.flow span{align-self:center;text-align:center;color:#7085a0}.flow small{display:block;color:#68819a;font-size:12px;margin-bottom:6px}.note{background:#fff6e6;border-left:3px solid #d3a54f;padding:12px 16px;font-size:14px;border-radius:4px}.provenance{font-family:ui-monospace,monospace;font-size:12px;color:#536c88;overflow-wrap:anywhere}.table-wrap{overflow-x:auto;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;vertical-align:top;padding:13px 12px;border-bottom:1px solid #e1e8f0;overflow-wrap:anywhere}th{background:#f1f5fb;color:#526b87}pre{background:#f3f6fa;border-radius:10px;padding:20px;font:13px/1.7 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}details{margin-top:18px}summary{cursor:pointer;font-weight:700}.status{background:#f7f9fc;padding:15px;border-radius:10px}.status strong{color:#674b15}.bullets li{margin:10px 0}.sources{display:flex;gap:18px;flex-wrap:wrap}footer{color:#60768e;font-size:13px}@media(max-width:700px){main{padding:16px 12px}header,.panel,.step{padding:24px 18px;border-radius:16px}h1{font-size:31px}h2{font-size:23px}.metrics{grid-template-columns:repeat(2,1fr)}.metric strong{font-size:20px}.flow{grid-template-columns:1fr}.flow span{transform:rotate(90deg);padding:4px}.overview{min-width:800px}.diagram-scroll{overflow:auto}.step-head{gap:12px}.concept{font-size:18px}}@media print{body{background:white}main{max-width:none;padding:0}.nav{display:none}.step{page-break-before:always}header,.panel{break-inside:avoid}.overview{min-width:0}h1{font-size:33px}}'''
    page=['<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>샷 추적 · Entity · 이야기 리포트</title><style>'+font_css+css+'</style></head><body><main>',
          '<header><div class="eyebrow">04 ENTITY EXTRACTION / EVIDENCE TO STORY</div><h1>움직임으로 대상을 모으고,<br>같은 대상의 이야기로 연결합니다.</h1><p class="lead">박스 하나가 사람 하나는 아닙니다. 원본의 관측을 모아 샷별 대상을 정리하고, 여러 샷의 문맥과 외형으로 전역 이름을 붙입니다.</p>',
          f'<p class="muted">실행: {_escape(output.name)} · 문서 스냅샷: {_escape(generated)}</p>',
          '<nav class="nav"><a href="/">결과 화면</a><a href="#overview">전체 구조</a><a href="#step-01">단계별 설명</a><a href="#models">모델·호출</a><a href="#timing">소요 시간</a><a href="#quality">품질 기록</a></nav></header>',
          '<div class="metrics">'+''.join(f'<div class="metric"><small>{_escape(label)}</small><strong>{_escape(value)}</strong></div>' for label,value in [('샷',shots or '미기록'),('전역 Entity',entities),('Qwen 실제 생성',f'{calls} / {limit}'),('식별 품질',quality_state)])+'</div>',
          '<section class="panel" id="overview"><div class="eyebrow">THE BIG PICTURE</div><h2>전체를 세 단계로 보면 쉽습니다.</h2><p>원본에서 시각 근거를 모으고, 샷 안의 대상을 정리한 뒤, 샷 사이에서 같은 대상을 연결합니다.</p><div class="diagram-scroll">'+_overview()+'</div><p class="muted">추적에는 원본 연속 프레임을 사용합니다. Adaptive 메타데이터에서 원본 샷을 찾고 결과 영상에서는 기존 선택 프레임과 유지 시간을 보존합니다.</p></section>']
    for index,(group,title,model,before,after,concept,detail,caution,evidence) in enumerate(steps,1):
        page += [f'<article class="step" id="step-{index:02d}"><div class="eyebrow">{_escape(group)}</div><div class="step-head"><span class="number">{index:02d}</span><h2>{_escape(title)}</h2></div><p class="concept">{_escape(concept)}</p><div class="flow"><div><small>INPUT</small>{_escape(before)}</div><span>→</span><div class="model"><small>PROCESS</small>{_escape(model)}</div><span>→</span><div><small>OUTPUT</small>{_escape(after)}</div></div><p>{_escape(detail)}</p><p class="note">{_escape(caution)}</p><p class="provenance">{_escape(evidence)}</p></article>']
    page += ['<section class="panel" id="models"><div class="eyebrow">MODELS & BOUNDED ORCHESTRATION</div><h2>어떤 일을 어떤 모델에 맡기는가</h2>'+_table(['모델','역할','적용 범위'],model_rows),
             '<h3>호출을 제한한 역할별 파이프라인</h3><p>Python이 순서를 제어합니다. 자유롭게 반복하는 Agent 루프는 없습니다.</p><p>'+_escape(budget_note)+'</p><p>'+_escape(cache_note)+'</p>',
             '<h3>실제 모델 생성 기록</h3>'+_table(['모델','정밀도','횟수'],actual_rows),
             '<p class="muted">큰 모델은 GPU 메모리에 따라 NF4 4bit / BF16 compute로 적재할 수 있습니다. 위 실제 원장을 기준으로 확인합니다.</p><details><summary>실행에 기록된 모델 revision</summary>'+_table(['모델','revision'],revisions)+'</details></section>',
             '<section class="panel"><div class="eyebrow">HOW TO READ EVIDENCE</div><h2>박스, 마스크, 이름은 서로 다른 근거입니다.</h2><ul class="bullets">'+''.join(f'<li>{_escape(s)}</li>' for s in semantics)+'</ul><details><summary>수식으로 보는 핵심 비교</summary><pre>'+_escape(formulas)+'</pre><p>유사도와 마스크 안정성은 확률로 보정된 동일성 점수가 아닙니다. 여러 프레임의 근거와 장면 관계를 함께 확인합니다.</p></details></section>',
             '<section class="panel" id="timing"><div class="eyebrow">MEASURED EXECUTION</div><h2>실제 기록으로 보는 처리 시간</h2>'+_table(['범위','시간','기준'],times)+'<p class="note">'+_escape(time_note)+'</p></section>',
             f'<section class="panel" id="quality"><div class="eyebrow">RESULTS & LIMITATIONS</div><h2>이번 결과의 품질 기록</h2><p class="status">실행: <strong>{_escape(run_state)}</strong> · 식별 품질: <strong>{_escape(quality_state)}</strong></p><p>{_escape(quality_note)}</p>'+_table(['항목','실제 기록'],measured),
             '<h3>남아 있는 항목</h3><ul>'+''.join(f'<li>{_escape(item)}</li>' for item in issues or ['issues 목록은 비어 있습니다. 독립 식별 평가 완료를 뜻하지 않습니다.'])+'</ul>']
    for title,data in extra:
        if data:page+=['<details><summary>'+_escape(title)+'</summary><pre>'+_escape(json.dumps(data,ensure_ascii=False,indent=2))+'</pre></details>']
    page += ['</section><section class="panel"><h2>결과를 검토하는 순서</h2><ol><li>전체 박스 영상에서 원본/adaptive 모드의 차이를 확인합니다.</li><li>Entity 갤러리에서 같은 대상의 여러 crop과 등장 샷을 봅니다.</li><li>샷 요약을 눌러 장면으로 이동하고 초기 요약과 보정 요약을 비교합니다.</li><li>품질 기록에서 누락·불확실·fallback 근거를 확인합니다.</li></ol>'+_table(['파일','내용'],files)+'</section>',
             '<section class="panel"><h2>연구에서 가져온 원칙</h2><p>'+_escape(research)+'</p><div class="sources">'+''.join(f'<a href="{url}">{name}</a>' for name,url in sources)+'</div><details><summary>전체 Markdown 설명</summary><pre>'+_escape(markdown)+'</pre></details></section>',
             '<footer>문서의 수치는 선택한 결과 폴더의 실제 파일을 읽은 스냅샷입니다. 파일 생성 여부나 ID 개수만으로 정확도를 판단하지 않습니다.</footer></main></body></html>']
    for name,content in [('REPORT.md',markdown),('REPORT.html',''.join(page))]:
        temporary=output/(name+'.tmp');temporary.write_text(content,encoding='utf-8');temporary.replace(output/name)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    write_report(parser.parse_args().output)
