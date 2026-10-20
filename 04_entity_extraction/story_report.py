"""Generate an evidence-specific Korean report without inference or budget use."""
from __future__ import annotations
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
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
     '사람·동물/생물·물체·불명으로 종류를 다시 평가하고 이름과 외형 설명을 붙입니다. 끊어진 후보, 신체 일부, 붙잡기·운반 관계를 검토합니다. 시간적 대상 교체 후보를 분리하고 같은 local에 묶인 큰 동시 관측은 원본 프레임으로 소유관계를 재검토합니다. 검출과 VLM의 사람↔동물 분류 충돌 및 두 대상 비교의 종류 변경 제안을 후보로 삼고, 대상 하나만 표시한 원본 최대 3프레임으로 종류와 역할을 재확인합니다. 확인한 트랙만 수정하며 같은 그룹의 다른 트랙에는 종류를 전파하지 않습니다. 검출 점수만으로 종류를 바꾸지 않습니다. 미해결은 경고로 남깁니다. VLM가 소지 물체라고 분류해 놓고 사람 ID에 배정한 모순은 물체를 분리하여 보정합니다. 반복된 위치 차이나 분리 마스크는 재검토 후보입니다. 원본에서 다른 개체임을 확인한 시각 근거만 병합금지 제약으로 사용하며, 신체 부분의 소유자는 별도로 검토합니다. 각 트랙의 배정·제외·실패를 기록하고 초기 요약을 작성합니다.',
     '장면에서 보인 검출 누락 후보도 기록합니다. 텍스트만으로 실제 박스를 만들어 존재를 확정하지 않습니다.',
     'local_entities.json · reviewed_tracking.json · local_classification_audit/ · review_traces/'),
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
    ('샷 간 연결','각 Local Entity를 별도 임시 ID로 시작','결정적 초기화','검토된 Local Entity 목록','Local별 독립 임시 ID',
     '샷 내부에서 정리한 각 대상을 별도 ID로 두고, 샷을 가로지르는 연결은 이후의 시각 비교로 추가합니다.',
     '기본 local 초기화는 각 Local Entity 하나를 임시 그룹 하나로 만듭니다. 초기 그룹 생성에는 모델 호출이 없으며, local의 종류·외형과 관측 출처를 보존합니다. 이 초기 목록 자체는 샷 간 동일성이 확인된 결과가 아닙니다.',
     '로컬 트랙의 혼합이 모두 해소되었다는 보장은 없습니다. 미해결 로컬 경고와 후속 비교 근거를 함께 확인합니다.',
     'global_provisional.json · global_entities.json: initialization'),
    ('샷 간 연결','불분명한 연결을 집중 비교하고 검증','GLOBAL_MODEL','임시 목록 + 대상 crop + 원본 장면','근거를 기록한 Global Entity',
     'DINO 외형·설명과 선택적 SFace 얼굴 유사도로 후보를 검색하고, 일반 비교는 두 전역 ID 후보의 여러 샷에서 대표 원본과 crop을 골라 나란히 봅니다.',
     '한 보드에 양쪽의 이미 연결된 서로 다른 샷을 최대 3개씩 표시합니다. 한 샷만 있으면 그 안의 다양한 시점을 씁니다. 설치된 YuNet/SFace는 얼굴 관측 선택·확대와 후보 순위만 보조하며 자동 병합에는 쓰지 않습니다. 모델은 각 열의 view 번호와 시각 근거를 반환하고, 코드가 이를 고정된 Local ID·원본 프레임에 결속합니다. 일반 비교에는 기존 Local 외형 설명을 넣지 않습니다. 샷 경계의 연속성과 소유관계는 전용 경로로 검토합니다. 부분·착용물은 같은 원본 프레임에서 소유자와 직접 이어진 모습이 필요하며 단일 쌍으로 검토합니다. 분리된 전신 관측이 같은 프레임에 있으면 두 박스를 표시한 원본 한 장에서 로컬 Instruct 모델로 신체 수를 먼저 확인합니다. 확인된 동시 개체 및 명시적 로컬 제약을 그룹 전체에 적용하고, 내부 모순이 드러난 그룹은 local 단위로 분리합니다. 샷 간 반증은 재검토 가능한 기록이며 영구 병합금지로 만들지 않습니다. 가린 얼굴이나 복장·조명 차이는 불확실로 남깁니다. 실패·미해결 쌍은 보류하고 같은 생성 상한 안에서 다른 후보를 계속 검토합니다. 실패 이력을 보존하며 외부 소개문은 일반 비교를 포함한 시각 검토의 약한 문맥이며 단독 병합 근거로 사용하지 않습니다.',
     '형식과 근거 위치 검증은 VLM 설명 자체의 진실을 보장하지 않습니다. 유사도나 소개문만으로 동일성을 확정하지 않습니다.',
     'identity_reconciliation/audit.json · global_entities.json · video_context.json'),
    ('샷 간 연결','전역 이름으로 장면 요약을 보정','NARRATIVE_MODEL','Global 목록 + 사건 순서 + 장면','보정된 요약과 사건',
     '같은 대상에게 같은 이름을 쓰면서, 누가 누구에게 무엇을 했는지 장면별로 다시 설명합니다.',
     '전역 ID와 전체 사건 순서, 장면 이미지를 함께 검토합니다. 초기 요약과 보정 요약을 모두 저장합니다. 사건은 주체·행동·상대·시각·근거 프레임을 갖고 관측·추론·불확실을 구분합니다. 최대 10샷 단위로 묶으며 전체 앞뒤 요약은 함께 제공합니다.',
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


def _overview(mask_enabled=True,initialization='local'):
    columns=[('A. 관측을 수집','#eaf1ff','#b9cff3',
              ['원본 → 검출 · 추적','독립 SAM → 마스크 비교','중복 융합 · 시간 구간 후보'] if mask_enabled else
              ['원본 → 검출 · 추적','컷 · 대상 교체 구간 분리','원본 박스 · 장면 보존'],'시각 근거와 후보'),
             ('B. 샷 안을 정리','#e8f6f0','#b6ddcd',
              ['VLM → 대상 · 사건','SAM → 물리 트랙 보정','전경 DINO → 외형 후보'] if mask_enabled else
              ['VLM → 대상 · 사건','원본 crop → DINO 외형','Local 목록 · 초기 요약'],'Local Entity 목록'),
             ('C. 샷 사이를 연결','#fff3e4','#e9cfaa',
              ['2방향 목록 → 집중 비교' if initialization=='whole' else '독립 ID → 후보 집중 비교',
               '근거 인용 · 병합 제약','전역 ID → 요약 · 검증'],'Global Entity + 이야기')]
    parts=['<svg class="overview" viewBox="0 0 1000 306" role="img" aria-labelledby="overview-title"><title id="overview-title">관측 수집, 샷 내부 정리, 샷 간 연결의 세 단계</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#75889e"/></marker></defs>']
    for index,(title,fill,stroke,lines,result) in enumerate(columns):
        x=8+342*index
        parts+=[f'<rect x="{x}" y="12" width="300" height="280" rx="18" fill="{fill}" stroke="{stroke}"/><text x="{x+22}" y="62" font-size="23" font-weight="700" fill="#172d46">{title}</text>']
        for j,line in enumerate(lines):parts+=[f'<text x="{x+22}" y="{108+j*39}" font-size="17" fill="#334b64">{line}</text>']
        parts+=[f'<text x="{x+22}" y="253" font-size="15" fill="#536a82">산출물: {result}</text>']
    return ''.join(parts)+'<path d="M314 155H341M657 155H684" stroke="#75889e" stroke-width="2.5" marker-end="url(#arrow)"/></svg>'


def _json_download(label, data, filename):
    payload=base64.b64encode(json.dumps(data,ensure_ascii=False,indent=2).encode()).decode()
    return f'<a href="data:application/json;base64,{payload}" download="{html.escape(filename,quote=True)}">{_escape(label)}</a>'


def regression_provenance_issues(output, regression, config):
    """Bind an evaluation to current run files, not to surviving old snapshots."""
    if not regression:
        return []
    artifacts=regression.get('artifacts')
    if not isinstance(artifacts,dict):
        return ['평가 입력의 SHA-256 출처 기록이 없습니다.']
    mask_enabled=config.get('mask_refinement') is not False
    bundle_names=(('mask_refinement/bundle.json',) if mask_enabled else ())+('reviewed_tracking.json',)+(
        ('mask_track_graph/bundle.json',) if mask_enabled else ())+('dense_tracking/bundle.json',)
    bundle=next((output/name for name in bundle_names if (output/name).is_file()),None)
    local=output/('mask_refinement/local.json' if bundle==output/'mask_refinement/bundle.json' else 'local_entities.json')
    current=dict(final_catalog=output/'global_entities.json',local_entities=local,bundle=bundle,
                 exported_observations=output/'observations.json',
                 provisional_catalog=output/'global_provisional.json',run_config=output/'run_config.json')
    required={'final_catalog','local_entities','bundle','labels','baseline'}
    if regression.get('provisional'):required.add('provisional_catalog')
    if current['run_config'].is_file():required.add('run_config')
    issues=[]
    for name in sorted(required|set(artifacts)):
        record=artifacts.get(name)
        if not isinstance(record,dict) or not isinstance(record.get('path'),str) or not record['path']:
            issues.append(f'{name}: 평가 파일 출처가 없습니다.');continue
        digest=record.get('sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
            issues.append(f'{name}: SHA-256 기록이 유효하지 않습니다.');continue
        # Run outputs are compared with the active run, even if the evaluator
        # originally used --final-catalog or --observations in another directory.
        path=current[name] if name in current else Path(record['path'])
        if path is None:
            issues.append(f'{name}: 현재 실행 파일이 없습니다.');continue
        try:
            with path.open('rb') as source:actual=hashlib.file_digest(source,'sha256').hexdigest()
        except OSError:
            issues.append(f'{name}: 평가 입력 파일을 확인할 수 없습니다.');continue
        if actual!=digest.lower():issues.append(f'{name}: 현재 파일과 평가 시점의 SHA-256이 다릅니다.')
    return issues


def _review_rows(audit, regression):
    audit_rows=[]
    if audit:
        audit_rows=[['집중 검토 상태',audit.get('status','미기록')],
                    ['전역 목록 크기',f"{audit.get('initial_entities','?')} → {audit.get('final_entities','?')}"],
                    ['추가 병합',sum(len(r.get('joins',[])) for r in audit.get('reviews',[])) if audit.get('strategy')=='compact' else len(audit.get('joins') or [])],
                    ['집중 검토 생성',f"{audit.get('prior_generations',0)+audit.get('generations',0)} / {audit.get('total_generation_limit',audit.get('max_generations','?'))}"],
                    ['미해결 / 미검토 / 오류',f"{len(audit.get('unresolved') or [])} / {len(audit.get('unreviewed_local_ids') or audit.get('unreviewed_candidates') or [])} / {len(audit.get('errors') or [])}"]]
    regression_rows=[]
    def annotation_count(value):
        # Current evaluator emits counts; older saved reports used ID lists.
        return value if isinstance(value,int) else len(value or [])
    for key,label in [('provisional','집중 검토 전'),('final','최종 목록')]:
        result=regression.get(key) or {}
        if not result:continue
        gate=result.get('gate') or {};evaluation=result.get('evaluation') or {}
        current=evaluation.get('current') or {}
        outcome='통과' if gate.get('passed') is True else '실패' if gate.get('passed') is False else '미기록'
        regression_rows.append([label,outcome,
            f"{annotation_count(evaluation.get('mapped_annotation_ids'))} / {annotation_count(evaluation.get('annotation_ids'))}",
            f"{current.get('correct_same_id_pairs','?')} / {current.get('same_id_pairs','?')}",
            current.get('false_merge_pairs','미기록')])
    return audit_rows,regression_rows


def write_report(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);read=lambda name:_read(output,name)
    status=read('status.json');quality=read('quality.json');tracking=read('tracking_summary.json')
    local=read('local_entities.json');glob=read('global_entities.json');config=read('run_config.json')
    graph=read('mask_track_graph/summary.json');raw=read('raw_track_masks/summary.json');mask=read('mask_refinement/summary.json')
    raw_metrics=read('raw_track_masks/run_metrics.json');history=read('execution_history.json');ledger=read('qwen_budget.json')
    validation=read('story_validation.json');verification=read('verification_summary.json') or read('verification.json')
    audit=glob.get('identity_reconciliation') or quality.get('identity_reconciliation') or read('identity_reconciliation/audit.json')
    regression=read('regression_evaluation.json')
    provenance_issues=regression_provenance_issues(output,regression,config)
    audit_rows,regression_rows=_review_rows(audit,{} if provenance_issues else regression)
    regression_issues=((regression.get('final') or {}).get('gate') or {}).get('issues') or []
    regression_note='고정된 부분 정답의 관측만 평가합니다. 통과해도 영상 전체의 검출·식별 정확도를 의미하지 않습니다.'
    work=read('work_timing.json');settings=config.get('settings') or {};stats=tracking.get('stats') or {}
    mask_enabled=config.get('mask_refinement',True) is not False
    initialization=glob.get('initialization',config.get('global_initialization','local'))
    initial_flow='전체 목록의 정방향·역방향 검토' if initialization=='whole' else 'Local별 독립 임시 ID'
    models={'LOCAL_MODEL':settings.get('model_id','Qwen3-VL · 설정 미기록'),
            'GLOBAL_MODEL':config.get('global_model_id','Qwen3-VL · 설정 미기록'),
            'NARRATIVE_MODEL':config.get('narrative_model_id',settings.get('model_id','Qwen3-VL · 설정 미기록'))}
    steps=[(*s[:2],models.get(s[2],s[2]),*s[3:]) for s in STEPS]
    steps[0]=(*steps[0][:2],config.get('detector','검출 모델 미기록')+' · BoT-SORT · DINOv2',*steps[0][3:])
    if initialization=='whole':
        steps[6]=('샷 간 연결','전체 목록을 정방향·역방향으로 검토',models['GLOBAL_MODEL'],
            'Local 목록 + 외형 + 샷 요약','임시 Global Entity 목록',
            '이 실행은 명시적인 whole 초기화로 전체 목록을 먼저 검토했습니다.',
            '정방향 목록을 역방향으로 재검토한 뒤, 같은 집중 비교와 병합 제약 검사를 적용합니다. 초기 그룹의 동일성은 후속 검토 대상이며, 생성과 재시도는 공통 예산에 포함됩니다.',
            '전체 목록 검토는 선택적 실험 경로입니다. 기본값은 각 Local Entity를 별도 ID로 시작합니다.',
            'global_provisional.json · review_traces/ · qwen_budget.json')
    if not mask_enabled:
        steps=[step for index,step in enumerate(steps) if index not in (1,2,4)]
        index=next(i for i,step in enumerate(steps) if step[2]=='DINOv2-small')
        steps[index]=('샷 내부 정리','원본 crop의 외형으로 연결 후보 검색','DINOv2-small',
            '원본 대상 crop','샷 사이 외형 유사도 후보','대상의 여러 모습에서 외형 특징을 비교합니다.',
            '이 실행은 SAM 단계를 생략하여 원본 검출 crop을 사용합니다. DINO 유사도와 설명은 집중 비교의 후보를 고르며 병합을 확정하지 않습니다.',
            '원본 crop의 배경이나 이웃 대상이 유사도에 영향을 줄 수 있습니다. 최종 연결은 실제 장면과 대상 근거를 검토합니다.',
            'dense_tracking/bundle.json · identity_reconciliation/audit.json')
    sampled_only=config.get('analysis_frames')=='sampled'
    if sampled_only:
        steps[0]=('관측 수집','Adaptive 선택 프레임만 검출·연결',config.get('detector','YOLO11x')+' · DINOv2 · 시간 간격 기반 연결',
            'samples.json의 선택 이미지','샘플 시점의 관측·임시 트랙',
            '선택된 프레임만 분석하고 원본 프레임 번호와 시간을 유지합니다.',
            '원본 동영상 디코딩 없이 선택 JPEG만 사용합니다. 외형·박스 겹침으로 일대일 연결하고 실제 경과 시간으로 트랙을 만료합니다. 내부 컷·VLM 대표 이미지·동일성 재검토도 선택 프레임에 한정됩니다.',
            '샘플 사이의 움직임·짧은 출현은 알 수 없습니다. 가림으로 끊어진 트랙은 후속 시각 검토 대상입니다.',
            'dense_tracking/bundle.json (호환 경로명) · tracking_summary.json')
    compact=config.get('global_strategy')=='compact'
    if config.get('global_strategy')=='linked':
        steps.append(('오류 보정','기존 연결을 보존하고 충돌 프레임만 수정',models['LOCAL_MODEL']+' → '+models['GLOBAL_MODEL'],
            '기존 Global 연결 · 선택 프레임 · 외형 검토 근거','보정된 ID · 검증된 동일 몸의 단일 박스',
            '기존 연결을 모두 해체하지 않습니다. 확인된 외형 모순과 같은 프레임의 물리적 충돌만 집중 검토합니다.',
            'Instruct로 물리적 몸을 구분하고 ID 변경이나 낮은 겹침의 몸 병합은 Thinking으로 재확인합니다. 이름은 외형 근거로 별도 결정합니다. 같은 몸이라는 확인을 받은 프레임만 외곽 박스로 합치며 원본 관측은 보존합니다.',
            '기존의 충돌 없는 오연결·미연결을 모두 찾아내는 검토는 아닙니다. 복구 시간은 새 영상 전체 처리 시간이 아닙니다.',
            'catalog_repair/audit.json · catalog_repair/*/decision.json · duplicate_observations.json'))
    if compact:
        steps=[step for step in steps if step[2]!='DINOv2-small']
        i=next(i for i,step in enumerate(steps) if step[1]=='불분명한 연결을 집중 비교하고 검증')
        steps[i]=('샷 간 연결','카탈로그 일괄 비교와 강제 충돌 검사',models['GLOBAL_MODEL'],
            '기존 이름을 가린 후보별 대표 crop','외형 프로필 · Global ID · 검토 이력',
            '한 요청에 최대 18개 후보(후속 묶음은 새 후보 14개와 이전 목록의 유사 후보 최대 4개)를 보고 외형과 병합 후보를 작성합니다. 호출 한도에는 재시도도 포함합니다.',
            'DINO 외형이 급변하면 박스 위치가 겹쳐도 시간 구간을 나누고, 먼저 동일 프레임에서 떨어진 박스를 같은 Local ID에 묶은 경우를 분리합니다. 서로 다른 Local ID의 동시 등장과 확인된 머리·수염·외형 모순은 그룹 전체의 병합 금지 조건입니다. 설명 길이로 이름을 고르지 않습니다. 누락된 병합 제안은 독립 ID로 유지합니다.',
            '가려진 외형은 unknown입니다. 제한된 호출로 모든 후보 쌍을 검토할 수 없으며 잘못된 병합을 줄이는 대신 분리가 남을 수 있습니다.',
            'compact_identity/audit.json · unique_local_audit.json · global_entities.json')
    if compact:
        i=next(i for i,step in enumerate(steps) if step[1]=='카탈로그 일괄 비교와 강제 충돌 검사')
        steps.insert(i,('샷 내부 정리','추적 전환 후보의 외형을 따로 확인',models['LOCAL_MODEL'],
            '동일 샷에서 나뉜 추적 구간의 확대 이미지','근거 프레임에 결속한 외형 속성',
            '애매한 구간만 작은 비교판으로 확인합니다. 전역 이름을 정하는 단계와 분리하며 실제 생성 상한은 4회입니다.',
            '머리카락 유무·수염·얼굴 표현을 확인하고 보이지 않는 특징은 unknown으로 남깁니다. 여러 사람이 섞인 관측은 확정하지 않습니다. 검증된 속성은 전역 비교의 모순 검사에 사용합니다.',
            '명확히 확인하지 못한 시간 구간은 미확정 ID로 남습니다.',
            'temporal_profile_review/audit.json · compact_temporal_audit.json'))
        steps.append(('출력 검증','같은 프레임의 중복 얼굴 확인','YuNet · 좌표/랜드마크 검사',
            '외형이 확인된 인물의 중첩 관측','동일 얼굴 관측에 근거한 중복 연결',
            '같은 원본 프레임에서 같은 얼굴 위치와 랜드마크를 가리킬 때만 중복 검출을 연결합니다.',
            '샷 간 얼굴 인식이 아닙니다. 그룹 전체의 동시 등장·외형 모순 제약을 다시 확인하며, 최종 출력에 동일 ID의 서로 다른 박스가 남으면 실패합니다.',
            '확인되지 않은 신체 부분이나 추적 전환 후보를 억지로 합치지 않습니다.',
            'compact_identity/final_validation.json · story_validation.json'))
    shots=len(tracking.get('shots') or local.get('shots') or [])
    entities=len(glob['entities']) if 'entities' in glob else '미생성'
    calls=ledger.get('generations',status.get('qwen_generations','미기록'))
    limit=ledger.get('limit',config.get('qwen_max_calls',status.get('qwen_limit','미기록')))
    focus_limit=config.get('global_review_max_calls',audit.get('max_generations','미기록'))
    run_state=status.get('status','기록 없음');quality_state=quality.get('status','미평가')
    issues=list(quality.get('issues') or [])
    if status.get('error'):issues.append(status['error'])
    if not quality:issues.append('최종 품질 평가 파일이 아직 없습니다.')
    measured=[['샷',shots or '미기록'],['원본 처리 프레임',stats.get('analyzed_frames',stats.get('dense_frames','미기록'))],
              ['원시 추적 조각',stats.get('tracks','미기록')],['그래프 전 후보',graph.get('input_tracks','미기록')],
              ['그래프 후 후보',graph.get('output_tracks','미기록')],['독립 SAM 마스크 관측',raw.get('mask_frames','미기록')],
              ['독립 SAM 검출 fallback',raw.get('detector_fallback_frames','미기록')],
              ['검토된 local 대상',sum(len(s.get('entities',[])) for s in local.get('shots',[])) if local else '미생성'],
              ['최종 SAM 마스크 관측',mask.get('mask_frames','미기록')],['전역 Entity',entities]]
    if not mask_enabled:
        measured=[row for row in measured if 'SAM' not in row[0] and '그래프' not in row[0]]
        measured.insert(3,['SAM 단계','생략 (실행 설정)'])
    model_rows=[[config.get('detector','검출 모델 미기록'),'원본/타일 박스 검출','연속 프레임'],
                ['BoT-SORT','운동·외형에 따라 샷 내부 연결','GMC 기본값: 없음'],
                ['SAM 2.1 hiera-small','원시 마스크·local 대상 마스크','의미 동일성 판단 없음'],
                ['DINOv2-small','일반 외형 특징·전경 후보 검색','사람·비인간 공통'],
                [models['LOCAL_MODEL'],'샷 내부 종류·동일성·사건','샷별 검토'],
                [models['GLOBAL_MODEL'],'샷 간 동일성',initial_flow+' → 고정 LEFT/RIGHT 한 쌍 비교'],
                [models['NARRATIVE_MODEL'],'전체 문맥을 반영한 요약','장면 이미지 재확인']]
    if not mask_enabled:
        model_rows=[row for row in model_rows if not row[0].startswith('SAM')]
        model_rows[2]=['DINOv2-small','원본 crop 외형 특징·후보 검색','사람·비인간 공통']
    if sampled_only:
        measured[1][0]='선택된 분석 프레임'
        model_rows[0][1:] = ['선택 이미지/타일 검출','Adaptive 선택 프레임']
        model_rows[1] = ['시간 간격 기반 연결','DINO 외형·박스 일대일 연결','원본 시간 간격으로 만료']
    if compact:
        model_rows=[(name,role,'최대 18개 후보 일괄 비교 · 동시 등장/외형 충돌 제약' if role=='샷 간 동일성' else detail) for name,role,detail in model_rows]
    events=ledger.get('events') or []
    actual_models=Counter((e.get('model_id','과거 이벤트: 모델 미기록'),e.get('precision','미기록')) for e in events)
    actual_rows=[(name,precision,count) for (name,precision),count in actual_models.items()] or [('미기록','미기록','미기록')]
    phase_counts=Counter(e.get('phase','미기록').split(':')[0] for e in events)
    revisions=[['Local Qwen',settings.get('model_revision','설정 기록 없음')],
               ['Global Qwen',config.get('global_model_revision','설정 기록 없음')],
               ['SAM',((mask or raw).get('model_identity') or {}).get('revision','이 실행 기록 없음')],
               ['DINO',(read('mask_refinement/appearance.json') or read('mask_track_graph/appearance.json')).get(
                   'revision',(tracking.get('config') or {}).get('appearance_revision','이 실행 기록 없음'))]]
    if not mask_enabled:revisions=[row for row in revisions if row[0]!='SAM']
    finished=[r for r in history.get('invocations',[]) if isinstance(r.get('elapsed_sec'),(int,float))]
    times=[['현재/마지막 파이프라인 호출',_duration(status.get('elapsed_sec')),'status.json; 한 번의 프로세스 실행'],
           ['기록된 종료 호출 누적',_duration(sum(r['elapsed_sec'] for r in finished)) if finished else '기록 없음',f'execution_history.json; {len(finished)}개 종료 기록'],
           ['별도 원시 SAM 전체',_duration(raw_metrics.get('total_wall_sec_including_resumed_attempt')),'중단·재개 포함 별도 실행'],
           ['원시 SAM 마지막 생성 실행',_duration(raw_metrics.get('elapsed_sec',raw.get('elapsed_sec'))),'캐시 재사용 시간과 다름']]
    if not mask_enabled:times=[row for row in times if 'SAM' not in row[0]]
    if work:times.append(['개발·다운로드·검증 전체',_duration(work.get('elapsed_sec')),'work_timing.json; 분석 실행과 별도'])
    time_note='분석 프로세스, 별도 SAM 작업, 개발·다운로드·검증 전체 시간은 서로 다른 범위입니다. 기록이 겹칠 수 있어 자동 합산하지 않습니다. 진행 중 호출은 종료 시간이 아직 없을 수 있습니다.'
    if not mask_enabled:time_note='분석 프로세스와 개발·다운로드·검증 전체 시간은 서로 다른 범위입니다. 기록이 겹칠 수 있어 자동 합산하지 않습니다.'
    quality_note='실행 상태와 식별 품질은 별도입니다. 자동 구조 검사 통과만으로 같은 인물 연결이나 사건 이해가 정확하다고 판단하지 않습니다.'
    if read('reuse_provenance.json'):time_note+=' 이 결과는 기존 샘플·추적·샷 내부 추출을 재사용한 후속 검증이며, 새 영상의 전체 1회 처리 시간이 아닙니다.'
    budget_note=f'이 실행의 전체 생성 상한은 {limit}회, 집중 동일성 검토 상한은 {focus_limit}회입니다. 샷별 검토 → {initial_flow} → 고정 LEFT/RIGHT 한 쌍 비교 → 최대 10샷 단위 요약 보정 순서입니다. 실패 응답의 재시도와 다른 후보 검토도 같은 생성 상한을 따릅니다. 모든 단계의 실패·형식 재시도는 원장에 남습니다.'
    cache_note='호출 직전에 영속 원장을 기록합니다. 실패·재시도·중단도 사용량에 남고 재실행 시 같은 원장을 이어 씁니다. 캐시 적중은 새 생성이 아닙니다. 미검토 샷을 완료로 승격하지 않습니다. Hub 모델은 commit revision을 고정하며 SAM 로컬/HF 경로는 설정·가중치의 content hash를 검증한 경우에만 캐시를 공유합니다.'
    semantics=['검출 관측: 실제 YOLO 박스가 남아 있는 프레임입니다.',
               'SAM 관측: 검사를 통과한 양의 전경 픽셀에서 박스를 계산합니다. 모델 추정이며 수동 정답 마스크는 아닙니다.',
               '검출 fallback: SAM 근거가 실패하여 원래 검출 박스를 보존했습니다. 없는 전경 마스크를 표시하지 않습니다.',
               '관측 공백: 유효 마스크와 검출이 모두 없는 시점입니다. 이전 박스를 임의로 늘려 존재를 확정하지 않습니다.',
               'Adaptive 영상: 선택 프레임과 박스를 기존 유지 구간 동안 표시합니다. 원본 모드에서 연속 추적을 확인합니다.',
               'Global ID: 현재 모델이 같은 대상으로 연결한 판단입니다. 독립 평가와 관측 crop으로 확인해야 합니다.']
    formulas='마스크 IoU = |A ∩ B| / |A ∪ B|\n외형 코사인 유사도 = (a · b) / (‖a‖ ‖b‖)\n전체 시각 = 샷 시작 시각 + 원본 프레임 번호 / FPS'
    if not mask_enabled:
        semantics=[row for row in semantics if not row.startswith(('SAM 관측:', '검출 fallback:', '관측 공백:'))]
        formulas='\n'.join(formulas.splitlines()[1:])
        cache_note=cache_note.split(' Hub 모델은')[0]+' Hub 모델은 commit revision을 고정합니다.'
    flow_lines=['A. 관측 수집: 원본 → 검출/BoT-SORT'+(' → 독립 SAM → 마스크 그래프' if mask_enabled else ''),
                'B. 샷 내부: VLM → Local Entity/초기 요약'+(' → SAM 보정 → 전경 DINO' if mask_enabled else ' → 원본 crop DINO'),
                f'C. 샷 사이: {initial_flow} → 후보 집중 비교/제약 검증 → Global Entity → 요약/내보내기']
    files=[['entities.json / global_human_entities.json / global_other_entities.json','전역 목록·종류·대표 crop'],
           ['shot_entities.json / entity_shot_presence.csv','샷별 대상과 등장 여부 0/1'],
           ['story.json','초기/보정 요약·사건·전역 ID'],
           ['adaptive_entities_full.mp4 / original_entities_full.mp4','adaptive 유지 영상 / 원본 연속 추적 영상'],
           ['quality.json / story_validation.json','검토 실패·누락·참조/시간/중복 검사'],
           ['identity_reconciliation/audit.json','집중 비교의 프레임 근거·병합·제약·미해결 후보'],
           ['regression_evaluation.json','선택적 고정 부분 정답 회귀 평가'],
           ['qwen_budget.json / execution_history.json','실제 생성 예산·모델·실행 시간'],
           ['REPORT.html / REPORT.md','이 실행의 동작 설명과 수치']]
    research='시간에 따른 여러 모습·시각적 연결·장면 문맥을 함께 사용하는 CAST/PPCC/MuHPC의 문제 설정을 참고했습니다. 논문 모델 전체의 재현은 아닙니다. CAL의 옷 변화 재학습, 음성 화자 연결, YuNet/SFace/OSNet은 이 기본 파이프라인에 포함하지 않습니다.'
    if compact:research='Qwen의 시각 판단에 동시 등장 제약과 관측 출처 검증을 결합합니다. YuNet은 같은 프레임의 동일 얼굴 좌표 확인에만 쓰며, SFace/OSNet으로 샷 사이 인물을 자동 확정하지 않습니다.'
    sources=[('SAM 2','https://github.com/facebookresearch/sam2'),('BoT-SORT','https://github.com/NirAharon/BoT-SORT'),
             ('CAST','https://arxiv.org/abs/2201.07619'),('PPCC','https://movienet.github.io/projects/eccv18person.html'),
             ('MuHPC','https://www.robots.ox.ac.uk/~vgg/publications/2021/Brown21c/brown21c.pdf')]
    generated=datetime.now(timezone.utc).isoformat(timespec='seconds')
    md=['# 샷 추적 · 전역 Entity · 이야기 리포트','',f'- 결과 폴더: `{output.name}`',f'- 문서 생성 시각: {generated}','',
        '## 1. 먼저 이해할 컨셉','',
        '한 프레임의 박스를 바로 인물 ID로 만들지 않습니다. **관측을 시간으로 연결 → 샷 안에서 실제 대상을 정리 → 전체 샷에서 같은 대상을 연결 → 이름이 일관된 이야기 작성** 순서입니다.','',
        '03 단계의 adaptive 입력 메타데이터에서 원본 샷을 찾아 추적합니다. 원본 연속 프레임으로 추적하고, adaptive 출력에는 기존 선택 프레임과 유지 시간을 사용합니다.','',
        '```text',*flow_lines,'```','',
        '## 2. 이번 실행의 실제 기록','',f'- 실행 상태: **{run_state}**',f'- 품질 상태: **{quality_state}**',
        f'- 실제 Qwen 생성: **{calls} / {limit}회**. 실패·재시도·중단 이력 포함.',quality_note,'']+_md_table(['항목','실제 기록'],measured)+['','## 3. 단계별 입력과 출력','']
    for index,(group,title,model,before,after,concept,detail,caution,evidence) in enumerate(steps,1):
        md += [f'### {index:02d}. {title}','',concept,'',f'- 입력: {before}',f'- 모델/처리: {model}',f'- 출력: {after}','',detail,'',f'해석 범위: {caution}','',f'근거 파일: `{evidence}`','']
    md+=['## 4. 어떤 DL 모델을 언제 사용하는가','']+_md_table(['모델','역할','범위'],model_rows)+['',
        'GPU 메모리 추정에 따라 큰 Qwen은 NF4 4bit / BF16 compute로 적재할 수 있습니다. 실제 정밀도는 아래 호출 원장 기록을 따릅니다.','']+_md_table(['실제 생성 모델','정밀도','횟수'],actual_rows)+['','### 고정 revision','']+_md_table(['모델','기록된 revision'],revisions)+['',
        '## 5. Agent 구조와 호출 상한','',
        'Python이 역할별 순서를 제어하는 유한 파이프라인입니다. 모델·도구를 자유롭게 반복 소환하는 Agent 루프는 없습니다.','',budget_note,'',cache_note,'',f'실제 단계별 생성: {_text(dict(phase_counts))}','',
        '## 6. 관측과 수식을 읽는 방법','']+[f'- {s}' for s in semantics]+['','```text',formulas,'```','',
        '외형 코사인 유사도는 후보 검색 수치이며 동일 개체 확률이 아닙니다. 특정 작품 이름·샷 ID·대상 수를 정답 규칙으로 사용하지 않습니다. 사용자 부분 정답은 독립 평가에만 사용합니다.','',
        '## 7. 실제 소요 시간','']+_md_table(['범위','시간','기준'],times)+['',time_note,'','## 8. 품질과 남은 항목','',quality_note,'']
    if audit_rows:
        md+=['### 집중 동일성 검토','']+_md_table(['항목','기록'],audit_rows)+['',
             '[전체 검토 근거](identity_reconciliation/audit.json)','']
    if regression_rows:
        md+=['### 고정 부분 정답 회귀','',regression_note,'']+_md_table(
            ['목록','회귀 판정','주석 ID 관측 연결','동일 ID 쌍','오병합 쌍'],regression_rows)+['',
            '[부분 회귀 평가 상세](regression_evaluation.json)','']
        md+=[f'- {_text(item)}' for item in regression_issues]
    elif regression:
        md+=['### 고정 부분 정답 회귀 — 재검증 필요','',
             '현재 결과와 평가 입력의 일치를 확인하지 못해 이전 판정과 수치를 표시하지 않습니다.','']
        md+=[f'- {_text(item)}' for item in provenance_issues]
        md+=['','[이전 회귀 기록 · 현재 결과에 적용하지 않음](regression_evaluation.json)','']
    md += [f'- {_text(item)}' for item in issues] or ['- issues 목록은 비어 있습니다. 독립 식별 평가 완료를 뜻하지 않습니다.']
    extra=[('요청 문제의 부분 검수',read('identity_spot_check.json')),('구조 검사',validation),('독립 부분 정답 평가',quality.get('manual_evaluation')),('별도 검증 기록',verification),('작업 시간과 중단 이력',work)]
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
          '<section class="panel" id="overview"><div class="eyebrow">THE BIG PICTURE</div><h2>전체를 세 단계로 보면 쉽습니다.</h2><p>원본에서 시각 근거를 모으고, 샷 안의 대상을 정리한 뒤, 샷 사이에서 같은 대상을 연결합니다.</p><div class="diagram-scroll">'+_overview(mask_enabled,initialization)+'</div><p class="muted">추적에는 원본 연속 프레임을 사용합니다. Adaptive 메타데이터에서 원본 샷을 찾고 결과 영상에서는 기존 선택 프레임과 유지 시간을 보존합니다.</p></section>']
    for index,(group,title,model,before,after,concept,detail,caution,evidence) in enumerate(steps,1):
        page += [f'<article class="step" id="step-{index:02d}"><div class="eyebrow">{_escape(group)}</div><div class="step-head"><span class="number">{index:02d}</span><h2>{_escape(title)}</h2></div><p class="concept">{_escape(concept)}</p><div class="flow"><div><small>INPUT</small>{_escape(before)}</div><span>→</span><div class="model"><small>PROCESS</small>{_escape(model)}</div><span>→</span><div><small>OUTPUT</small>{_escape(after)}</div></div><p>{_escape(detail)}</p><p class="note">{_escape(caution)}</p><p class="provenance">{_escape(evidence)}</p></article>']
    page += ['<section class="panel" id="models"><div class="eyebrow">MODELS & BOUNDED ORCHESTRATION</div><h2>어떤 일을 어떤 모델에 맡기는가</h2>'+_table(['모델','역할','적용 범위'],model_rows),
             '<h3>호출을 제한한 역할별 파이프라인</h3><p>Python이 순서를 제어합니다. 자유롭게 반복하는 Agent 루프는 없습니다.</p><p>'+_escape(budget_note)+'</p><p>'+_escape(cache_note)+'</p>',
             '<h3>실제 모델 생성 기록</h3>'+_table(['모델','정밀도','횟수'],actual_rows),
             '<p class="muted">큰 모델은 GPU 메모리에 따라 NF4 4bit / BF16 compute로 적재할 수 있습니다. 위 실제 원장을 기준으로 확인합니다.</p><details><summary>실행에 기록된 모델 revision</summary>'+_table(['모델','revision'],revisions)+'</details></section>',
             '<section class="panel"><div class="eyebrow">HOW TO READ EVIDENCE</div><h2>박스, 마스크, 이름은 서로 다른 근거입니다.</h2><ul class="bullets">'+''.join(f'<li>{_escape(s)}</li>' for s in semantics)+'</ul><details><summary>수식으로 보는 핵심 비교</summary><pre>'+_escape(formulas)+'</pre><p>유사도와 마스크 안정성은 확률로 보정된 동일성 점수가 아닙니다. 여러 프레임의 근거와 장면 관계를 함께 확인합니다.</p></details></section>',
             '<section class="panel" id="timing"><div class="eyebrow">MEASURED EXECUTION</div><h2>실제 기록으로 보는 처리 시간</h2>'+_table(['범위','시간','기준'],times)+'<p class="note">'+_escape(time_note)+'</p></section>',
             f'<section class="panel" id="quality"><div class="eyebrow">RESULTS & LIMITATIONS</div><h2>이번 결과의 품질 기록</h2><p class="status">실행: <strong>{_escape(run_state)}</strong> · 식별 품질: <strong>{_escape(quality_state)}</strong></p><p>{_escape(quality_note)}</p>'+_table(['항목','실제 기록'],measured)]
    if audit_rows:
        page+=['<h3>집중 동일성 검토</h3>'+_table(['항목','기록'],audit_rows)+
               '<p>'+_json_download('전체 검토 근거 · identity_reconciliation/audit.json',audit,'identity_audit.json')+'</p>']
    if regression_rows:
        page+=['<h3>고정 부분 정답 회귀</h3><p class="muted">'+_escape(regression_note)+'</p>'+_table(
               ['목록','회귀 판정','주석 ID 관측 연결','동일 ID 쌍','오병합 쌍'],regression_rows)+
               '<p>'+_json_download('부분 회귀 평가 상세 · regression_evaluation.json',regression,'regression_evaluation.json')+'</p>']
        if regression_issues:page+=['<ul>'+''.join(f'<li>{_escape(item)}</li>' for item in regression_issues)+'</ul>']
    elif regression:
        page+=['<h3>고정 부분 정답 회귀 — 재검증 필요</h3><p class="note">현재 결과와 평가 입력의 일치를 확인하지 못해 이전 판정과 수치를 표시하지 않습니다.</p><ul>'+
               ''.join(f'<li>{_escape(item)}</li>' for item in provenance_issues)+'</ul><p>'+
               _json_download('이전 회귀 기록 · 현재 결과에 적용하지 않음',regression,'unverified_regression_evaluation.json')+'</p>']
    page+=['<h3>남아 있는 항목</h3><ul>'+''.join(f'<li>{_escape(item)}</li>' for item in issues or ['issues 목록은 비어 있습니다. 독립 식별 평가 완료를 뜻하지 않습니다.'])+'</ul>']
    for title,data in extra:
        if data:page+=['<details><summary>'+_escape(title)+'</summary><pre>'+_escape(json.dumps(data,ensure_ascii=False,indent=2))+'</pre></details>']
    page += ['</section><section class="panel"><h2>결과를 검토하는 순서</h2><ol><li>전체 박스 영상에서 원본/adaptive 모드의 차이를 확인합니다.</li><li>Entity 갤러리에서 같은 대상의 여러 crop과 등장 샷을 봅니다.</li><li>샷 요약을 눌러 장면으로 이동하고 초기 요약과 보정 요약을 비교합니다.</li><li>품질 기록에서 누락·불확실·fallback 근거를 확인합니다.</li></ol>'+_table(['파일','내용'],files)+'</section>',
             '<section class="panel"><h2>연구에서 가져온 원칙</h2><p>'+_escape(research)+'</p><div class="sources">'+''.join(f'<a href="{url}">{name}</a>' for name,url in sources)+'</div><details><summary>전체 Markdown 설명</summary><pre>'+_escape(markdown)+'</pre></details></section>',
             '<footer>문서의 수치는 선택한 결과 폴더의 실제 파일을 읽은 스냅샷입니다. 파일 생성 여부나 ID 개수만으로 정확도를 판단하지 않습니다.</footer></main></body></html>']
    for name,content in [('REPORT.md',markdown),('REPORT.html',''.join(page))]:
        if sampled_only:
            content=content.replace('03 단계의 adaptive 입력 메타데이터에서 원본 샷을 찾아 추적합니다. 원본 연속 프레임으로 추적하고, adaptive 출력에는 기존 선택 프레임과 유지 시간을 사용합니다.',
                '03에서 선택한 JPEG만 검출·외형 추출·VLM 검토에 사용합니다. 원본 영상은 읽지 않으며 선택 프레임의 원본 시각과 유지 시간을 보존합니다.')
            content=content.replace('추적에는 원본 연속 프레임을 사용합니다. Adaptive 메타데이터에서 원본 샷을 찾고 결과 영상에서는 기존 선택 프레임과 유지 시간을 보존합니다.',
                '분석은 선택 프레임에 한정됩니다. 출력은 각 관측 이미지를 다음 샘플 시각까지 유지하며, 사이 구간을 실제 관측으로 주장하지 않습니다.')
            content=content.replace('원본 → 검출/BoT-SORT','선택 프레임 → 검출/시간 간격 기반 연결').replace('원본 → 검출 · 추적','선택 프레임 → 검출 · 연결')
            content=content.replace('원본 추적 박스 영상과 adaptive 선택 프레임의 유지 구간에 맞춘 영상을 저장합니다.','선택 프레임의 유지 구간에 맞춘 Adaptive 박스 영상만 저장합니다.')
            content=content.replace('adaptive_entities_full.mp4 / original_entities_full.mp4','adaptive_entities_full.mp4').replace('adaptive 유지 영상 / 원본 연속 추적 영상','선택 관측 유지 영상 (원본 전체 분석 없음)')
            content=content.replace('전체 박스 영상에서 원본/adaptive 모드의 차이를 확인합니다.','Adaptive 영상에서 선택 관측과 유지 표시를 확인합니다.')
        if compact:
            content=content.replace('identity_reconciliation/audit.json','compact_identity/audit.json')
        temporary=output/(name+'.tmp');temporary.write_text(content,encoding='utf-8');temporary.replace(output/name)
    if compact:
        from identity_history import write_history
        write_history(output)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    write_report(parser.parse_args().output)
