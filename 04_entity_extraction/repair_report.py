"""Short, artifact-bound explanation of catalog repair and its partial evaluation."""
import html,json
from pathlib import Path
from collections import defaultdict


def write_repair_report(output):
    output=Path(output)
    def read(name,default):
        p=output/name
        return json.loads(p.read_text()) if p.is_file() else default
    evaluation=read('cross_shot_audit.json',{});audit=read('catalog_repair/audit.json',{})
    ledger=read('qwen_budget.json',{'events':[]});catalog=read('global_entities.json',{'entities':[]})
    validation=read('story_validation.json',{});history=read('repair_execution_history.json',[])
    elapsed=sum(row.get('elapsed_sec',0) for row in history)
    generation_time=sum(e.get('elapsed_sec',0) for e in ledger['events'])
    benchmark_path=output.parent/'benchmark_timing.json'
    benchmark=json.loads(benchmark_path.read_text()) if benchmark_path.is_file() else {}
    invocations=read('execution_history.json',{}).get('invocations',[])
    fresh=bool(benchmark and not history and not benchmark.get('resumed') and len(invocations)<=1)
    refined=bool(benchmark.get('refinement_reason') and benchmark.get('original_cold_run_elapsed_sec'))
    physical_check=read('physical_spot_check.json',{})
    title='분석 캐시 없이 만든 인물 목록 검수' if fresh else '기존 인물 연결은 보존하고, 오류가 있는 관측만 보정'
    flow_start='샷 분할 → 샘플링 → 샷별 분석 → 전역 연결' if fresh else '기존 연결 재사용'
    method=('이 결과는 빈 출력 폴더에서 샷 분할·샘플링·검출·인물 연결을 새로 수행했습니다. 기존 모델 가중치만 재사용했습니다. '
            '전역 비교는 먼저 짧게 판단하고, 미확정인 샷 간 비교만 동일 이미지로 긴 추론을 추가합니다. '
            if fresh else '이 결과는 기존 샷 분할·샘플링·검출·샷 간 연결·외형 프로필을 재사용했습니다. ')
    timing=(f"새 분석 전체: {benchmark.get('elapsed_sec',0)/60:.2f}분 (02 → 03 → 04, 모델 적재·출력 포함). "
            f"실행 상태: {html.escape(benchmark.get('status','unknown'))}. 다운로드·설치는 제외합니다."
            if fresh else f'개발 중 재시도·실패·중단·요약 실험까지 포함한 기록된 복구 실행 누적: {elapsed/60:.2f}분. '
            '초기 임시 스크립트 실행은 원장의 생성 시간에 포함되지만 이 실행 누적에는 포함되지 않습니다. '
            '새 영상 1회 처리 시간 또는 최적화된 단일 실행 시간으로 해석하면 안 됩니다.')
    if refined:
        title='빈 분석 캐시에서 시작한 인물 연결·물리 관측 검수'
        flow_start='샷 분할 → 샘플링 → 샷별 분석 → 전역 연결'
        method='빈 결과 폴더에서 전체 분석을 수행한 뒤, 수동 검수에서 발견한 그림자·중복 몸 오류를 추가 보정했습니다. 과거 실행의 인물 목록을 정답처럼 가져오지 않습니다. 사람 후보에 미확정 인물이 생기면 참조 인물 사진 없이 주변 영역을 확대해 실제 몸의 수를 독립적으로 확인합니다. '
        original=benchmark['original_cold_run_elapsed_sec']
        timing=(f"최초 무캐시 전체 실행 {original/60:.2f}분 + 마지막 보완 재실행 {max(0,benchmark.get('elapsed_sec',0)-original)/60:.2f}분. "
                '후자는 이번 분석에서 생성한 캐시를 재사용합니다. 최신 코드의 무중단 1회 처리 시간으로 합산·해석하면 안 됩니다. '
                '모델 가중치는 재사용했고 다운로드·설치는 제외했습니다.')
    cells=[]
    titles={'baseline':'이전 92회 결과','comparison':'5회 실험','repaired':'연결 보존 + 오류 보정'}
    for name in ['baseline','comparison','repaired']:
        if name not in evaluation.get('summary',{}):continue
        v=evaluation['summary'][name]
        cells.append('<tr><td>'+titles[name]+'</td>'+''.join('<td>'+str(v.get(k,0))+'</td>' for k in ('correct_same','fragmented','false_merge'))+'</tr>')
    phases=defaultdict(lambda:[0,0])
    for e in ledger['events']:
        key=e['phase'].split(':')[0];phases[key][0]+=1;phases[key][1]+=e.get('elapsed_sec',0)
    phase_rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v[0]}</td><td>{v[1]:.1f}초</td></tr>' for k,v in phases.items())
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>인물 연결 복구 검수</title><style>
body{{font:17px/1.7 sans-serif;background:#f4f6fa;color:#243449;max-width:1040px;margin:32px auto;padding:24px}}h1{{font-size:30px}}h2{{font-size:21px;margin-top:30px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}.flow{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.box{{padding:18px;background:#e5eefb;border:1px solid #8ea8ce;border-radius:8px}}small{{color:#53647a}}code{{background:#e5eaf1}}details{{margin-top:24px}}</style>
<h1>{title}</h1>
<p>자세 변화만으로 모든 연결을 해체하던 5회 전략을 기본에서 제외했습니다. 레이블 결정·샷 사이 연결·한 프레임의 몸 구분을 분리합니다.</p>
<div class="flow"><div class="box">{flow_start}</div>→<div class="box">확인된 외형 모순만 분리</div>→<div class="box">충돌 프레임 검토</div>→<div class="box">변경 판단만 추가 확인</div>→<div class="box">몸당 박스 하나</div></div>
<h2>검수 결과</h2><p>연결 보존·오병합 방지 기준: {'통과' if evaluation.get('gate',{}).get('passed') else '미통과 또는 미검증'}. 사전에 고정한 {evaluation.get('annotation_count','?')}개 관측, {evaluation.get('annotation_shots','?')}개 샷의 부분 평가입니다. 아래는 관측 쌍의 수이며 영상 전체 정확도가 아닙니다.</p>
<table><tr><th>결과</th><th>맞게 연결한 동일인 쌍</th><th>연결되지 않은 동일인 쌍</th><th>다른 사람의 오병합 쌍</th></tr>{''.join(cells)}</table>
<p>최종 entity {len(catalog['entities'])}개. 프레임별 동일 ID 충돌 {validation.get('simultaneous_identity_conflicts','?')}건. 이전부터 남아 있던 미연결과 미확정 대상은 보존하며 완전한 식별로 표시하지 않습니다.</p>
<p>별도 물리 관측 회귀 검사(그림자의 독립 인물 ID 방지·동일 몸의 중복 crop 통합): {'통과' if physical_check.get('passed') else '미검증 또는 미통과'}. 그림자 관측의 처리 방식은 physical_spot_check.json에 별도로 기록합니다. 원시 검출은 보존합니다.</p>
<h2>연산을 줄인 방법</h2><p>{method}연결 후 보정은 {audit.get('requested_frames','?')}개 충돌/혼합 후보 프레임을 확인했습니다. Instruct가 먼저 판단하고, 기존 ID 변경이나 겹침이 약한 몸 병합만 Thinking이 재확인합니다. 이 보정의 추론은 1,024 토큰과 최종 응답 길이를 함께 제한합니다.</p>
<p>같은 몸으로 확인된 검출만 원래 박스들을 감싸는 하나의 박스로 표현합니다. 원래 관측과 프레임별 검토 근거는 <code>duplicate_observations.json</code>, <code>catalog_repair/</code>에 모두 남깁니다.</p>
<h2>시간의 의미</h2><p>{timing} 모델 생성 누적: {generation_time/60:.2f}분.</p>
<p><code>analyze_sintel.sh</code>/<code>analyze_frozen.sh</code> 기본은 동일한 <code>linked</code> 로직이며 후보 비교 최대 96회(추가 검토·재시도 포함)와 필요한 물리적 충돌 보정으로 동작합니다. 영상별 처리 시간과 정확도는 달라지며, 이 부분 검수는 다른 영상의 정확도를 보장하지 않습니다.</p>
<details><summary>개발 실행의 전체 호출 원장 요약</summary><p>폐기한 중간 방식의 호출도 삭제하지 않았습니다. catalog_repair_check는 개발 중 시험한 Instruct 재확인으로 최종 방식에서는 사용하지 않습니다.</p><table><tr><th>단계</th><th>실제 호출</th><th>생성 시간</th></tr>{phase_rows}</table></details></html>'''
    (output/'REPAIR_REPORT.html').write_text(page)
