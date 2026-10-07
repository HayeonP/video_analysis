"""Automatically check exported mappings and write a readable run summary."""
from collections import Counter, defaultdict
import csv
from pathlib import Path

from low_freq_event.storage import atomic_json, read_json


def write_mode_report(output):
    output=Path(output)
    catalog=read_json(output/'entities.json')['entities']
    observations=read_json(output/'observations.json')['observations']
    frames=read_json(output/'samples.json')
    timeline=read_json(output/'timeline.json')
    annotated=read_json(output/'frame_annotations.json')
    shots=[s['id'] for s in timeline['shots']]
    ids={e['entity_id'] for e in catalog}
    frame_keys={f['frame_key'] for f in frames}
    seen=defaultdict(set);observed_shots=defaultdict(set)
    for o in observations:
        if o['entity_id'] not in ids or o['frame_key'] not in frame_keys:
            raise ValueError('Unknown exported entity/frame reference.')
        if o['entity_id'] in seen[o['frame_key']]:
            raise ValueError('Two boxes in one frame have the same final ID.')
        seen[o['frame_key']].add(o['entity_id'])
        observed_shots[o['entity_id']].add(o['shot_id'])
    if len(ids)!=len(catalog) or [f['frame_key'] for f in annotated]!=[f['frame_key'] for f in frames]:
        raise ValueError('Duplicate entities or incomplete exported frame timeline.')
    with (output/'entity_shot_presence.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=len(ids) or {r['entity_id'] for r in rows}!=ids:
        raise ValueError('Presence matrix rows do not match final entity catalog.')
    for row in rows:
        for sid in shots:
            if row[sid]!=str(int(sid in observed_shots[row['entity_id']])):
                raise ValueError('Presence matrix disagrees with observed final IDs.')
    for e in catalog:
        if set(e['shots'])!=observed_shots[e['entity_id']] or not (output/e['representative']).is_file():
            raise ValueError('Entity shots or representative image missing.')
    checks=dict(passed=True,entities=len(ids),shots=len(shots),observations=len(observations),
        adaptive_frames=len(frames),binary_matrix_matches_observations=True,
        all_adaptive_frames_exported=True,no_co_visible_identity_collisions=True,
        representative_images_exist=True)
    atomic_json(output/'result_validation.json',checks)
    text=['# 자동 실행 결과', '',
        f'- 최종 entity: {len(ids)}개', f'- 샷: {len(shots)}개',
        f'- 분석한 adaptive 프레임: {len(frames)}개', f'- 박스 관측: {len(observations)}개',
        f'- 영상 길이: {timeline["duration_sec"]:.6f}초',
        '- 등장 행렬·관측·대표 이미지·프레임 목록의 일관성 검사: 통과', '',
        '[전체 대표 이미지](entity_gallery.jpg) · [등장 행렬](entity_shot_presence.csv) · [박스 영상](adaptive_entities_full.mp4)', '']
    quality_path=output/'anchored/quality.json'
    if quality_path.is_file():
        quality=read_json(quality_path)
        text+=['## 동일성 검토 상태','',
            f'- 상태: {quality["status"]}',
            f'- 동일성 미확정 관측 묶음: {quality["unresolved_identity_groups"]}개',
            '- 위 일관성 검사는 파일과 시간축 검사이며, 인물 식별 정확도 검사가 아닙니다.','']
        text.extend(f'- {warning}' for warning in quality.get('warnings',[]))
        text+=['','[검토 상태와 실패 내역](anchored/quality.json)','']
    sequence=output/'sequential'
    if sequence.exists():
        mapping=read_json(sequence/'final_mapping.json')
        local=read_json(sequence/'local_entities.json')
        forward=read_json(sequence/'forward.json')['assignments']
        reverse=read_json(sequence/'reverse.json')['assignments']
        if not set(mapping)==set(local)==set(forward)==set(reverse):
            raise ValueError('Forward/reverse/final local entity coverage differs.')
        by_detection={o['detection_id']:o for o in observations}
        for gid,e in local.items():
            for o in e['observations']:
                if by_detection[o['detection_id']]['entity_id']!=mapping[gid]['entity_id']:
                    raise ValueError('Final observation ID disagrees with reconciled mapping.')
        comparison=read_json(sequence/'direction_comparison.json')
        problems=[];added=0;rejected=0
        for phase in ('forward','reverse'):
            for file in sorted((sequence/phase).glob('*/result.json')):
                result=read_json(file)
                if result.get('status')!='complete':
                    problems.append(dict(phase=phase,shot_id=file.parent.name,error=result.get('error','uncertain structured output')))
                added+=len(result.get('additions',[]))
                rejected+=len(result.get('rejected_ids',[]))
        checks.update(both_directions_cover_all_local_entities=True,final_mapping_matches_observations=True,
            uncertain_shot_reviews=problems)
        atomic_json(output/'result_validation.json',checks)
        text+=['## 정방향·역방향·최종 검토', '',
            f'- 정방향 임시 ID: {comparison["forward_entities"]}개',
            f'- 역방향 임시 ID: {comparison["reverse_entities"]}개',
            f'- 양쪽 모두 동일 개체로 판단한 쌍: {len(comparison["same_in_both"])}개',
            f'- 정방향에서만 동일하다고 판단한 쌍: {len(comparison["forward_only"])}개',
            f'- 역방향에서만 동일하다고 판단한 쌍: {len(comparison["reverse_only"])}개',
            f'- 동일 개체 쌍 Jaccard 일치율: {comparison["pair_jaccard"]:.1%}',
            f'- 최종 개별 검토의 불확실 판정: {comparison["uncertain_pair_reviews"]}건',
            f'- 샷 기억 검토에서 추가한 박스: {added}개 / 명시적으로 제외한 임시 ID: {rejected}개',
            f'- 구조화 응답 실패로 보수적 결과를 유지한 샷 검토: {len(problems)}건', '',
            '일치율은 정답 정확도가 아니다. 두 방향은 같은 검출과 샷 내부 묶음을 사용하고, 샷 간 매칭 기억은 독립적으로 구성한다.',
            '0은 분석 샘플에서 확인하지 못했다는 뜻이다. 자동 매칭 오류와 동일 대상의 ID 분할이 남을 수 있다.', '',
            '[방향별 차이](sequential/direction_comparison.json) · [최종 ID 매핑](sequential/final_mapping.json) · [최종 판정 근거](sequential/final/decisions.json)', '']
        if problems:
            text+=['### 재검토가 불확실했던 샷', '']
            text += [f'- {p["phase"]} / {p["shot_id"]}: 기존 관측을 보존하고 불확실 처리' for p in problems]
    (output/'RESULT_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def write_run_reports(root, mode):
    root=Path(root)
    folders=[root/'with_qwen',root/'without_qwen'] if mode=='both' else [root]
    for folder in folders:
        write_mode_report(folder)
