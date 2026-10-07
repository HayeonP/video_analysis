# 문맥 기반 entity 매칭 수정 및 테스트

작성: 2026-10-06 (한국 시간). 적용 버전: `sequential_identity_version=2`.

## 무엇을 바꿨나

같은 인물을 영상 문맥에서 연결한 뒤 어두운 crop 때문에 다시 분리하던 경로를 제거했다.
샷 요약·인물별 관찰 이력·앞뒤 장면·대표 crop를 함께 보고 동일성을 판단한다.
특정 영화, 사람 이름, ID에 대한 수동 병합 규칙은 실제 분석 코드에 넣지 않았다.

1. **정방향 기억:** 모든 이전 샷의 요약 및 등장 대상, entity별 외형 변화·문맥·매칭 이유를 다음 샷에 전달한다.
2. **역방향 기억:** 별도의 요약과 매칭 이력을 만든다. 정방향의 전역 ID와 매칭 이유는 입력하지 않는다.
3. **최종 문맥 판정:** 대상 샷·인접 샷의 장면 프레임과 갤러리, 전체 샷 요약을 함께 제공한다. crop 단독 판정에 거부권을 주던 단계는 제거했다.
4. **불확실성과 차이 구분:** 얼굴이 안 보임, 실루엣, 옷·조명·자세 변화만으로 `different`를 주지 않도록 지시한다. `different`에는 실제 양쪽 대상 프레임 ID와 서로 모순되는 관찰 근거가 필요하다. `same`에는 긍정 근거가 필요하다.
5. **확대 재검토:** 불확실하거나 정·역방향의 기존 연결과 다른 결론이면 더 넓은 주변 샷과 더 많은 프레임으로 다시 본다. 끝까지 불확실하면 강제로 합치지 않는다.
6. **로그:** YOLO 캐시 재사용/신규 추론, Qwen review 캐시 재사용/신규 추론을 구분해 출력한다. OFF의 후보 형식 변환은 추론이 없다고 표시한다.

## 테스트 결과

### 자동 테스트: 17개 통과

- 없는 특징을 다른 사람의 증거로 확정하지 않는 처리.
- 근거 없는 `same`, 유효하지 않은 프레임 ID의 `different`, 서로 모순된 결론 처리.
- 대상 관찰 프레임과 앞뒤 샷 선택, 불확실할 때 확대 재검토.
- 정방향 샷 요약 전달, entity 이력 갱신, 독립 역방향 기억, 최종 병합까지 모의 모델로 통합 검사.
- 동시 등장 대상의 병합 차단 및 다른/불확실 대상 분리 유지.
- 캐시 삭제 범위, 최종 결과 보존, 실행 중 삭제 방지, 심볼릭 링크 외부 파일 보존.

```bash
../01_low_freq_event/.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

`run.sh --help`와 수정한 Python 모듈의 컴파일 검사도 통과했다.

### 실제 Qwen 테스트: 4개 통과

Qwen3-VL-4B-Instruct, RTX 4090, BF16로 실제 추론했다.
기존 정방향의 대상 박스와 샷 요약을 고정하여 **최종 문맥 판정 단계**를 검사했다.
프레임은 adaptive 입력에서 다시 디코딩했고 279장 모두 기존 JPEG와 바이트 단위로 일치했다.
추론 응답은 production 캐시와 분리된 빈 임시 폴더에서 새로 생성했다.

| 사례 | 기대 | 실제 |
|---|---|---|
| 샷 3의 짧은 머리 인물 ↔ 샷 5의 빨간 머리 여성 | 같은 인물 | same |
| 샷 5의 여성 ↔ 샷 11의 여성 | 같은 인물 | same |
| 샷 5의 여성 ↔ 샷 15의 여성 | 같은 인물 | same |
| 샷 4의 수염 있는 남성 ↔ 샷 5의 여성 | 다른 인물 | different |

실제 판정: [qwen_results.json](tests/evidence/context_v2/qwen_results.json).
기존 실패 기록: [v1_failed_pair.json](tests/evidence/context_v2/v1_failed_pair.json).
기존에는 첫 번째 쌍을 영상 판정에서 `same`, crop 판정에서 `different`로 뒤집어 분리했다.

재현 가능한 테스트 입력은 `tests/fixtures/sintel_context`에 보관했다. 모델 응답 캐시가 아니라
검사할 대상의 박스·crop·샷 요약이다. 일반 분석은 이 폴더를 읽지 않는다.

```bash
# 04_entity_extraction에서 실행. 매번 비어 있는 출력 경로를 지정해야 신규 추론 검사다.
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
../01_low_freq_event/.venv/bin/python tests/check_context_pairs.py \
  --output /tmp/entity-context-new-check
```

### 확인 범위

이번에 전체 17개 샷의 검출부터 영상 출력까지 새로 분석하지 않았다. 전체 실행은 사용자가 한다.
따라서 모든 여성 ID가 최종적으로 하나가 된다는 결과나 전체 정확도 향상을 확정한 상태는 아니다.
Qwen이 설명에 부정확한 시각적 근거를 적을 수 있으며, 구조 검사는 그 의미의 진실성까지 보장하지 않는다.
기존 중복 검출, 잘못된 샷 내부 묶음, 애매한 생물 crop가 최종 연결에 영향을 줄 수 있다.

## 캐시 초기화

04의 `outputs/sintel_sequential`과 `outputs/sintel_temporal` 양쪽 ON/OFF에서 추론·중간 캐시를 삭제했다.
YOLO 후보, Qwen 응답, 프레임, 검출, 특징, crop, 정·역방향/최종 검토 중간 자료 및 기존 실행 설정을 포함한다.
삭제 목록: [CACHE_CLEAR_REPORT.json](CACHE_CLEAR_REPORT.json).
입력 영상, 다운로드한 모델 가중치, 최종 카탈로그·대표 이미지·시각화 영상은 유지했다.
테스트 임시 추론 캐시도 정리했다. 테스트 판정 JSON은 검증 기록으로만 보존한다.

```bash
./run.sh
```

위 명령으로 ON의 YOLO부터 새로 계산한다. 같은 실행의 OFF는 ON에서 **새로 계산한 YOLO 후보**를 공유하며,
Qwen 응답·라벨·특징을 공유하지 않는다. 기존 잘못된 응답을 다시 읽는 동작이 아니다.

웹 뷰어에 남아 있는 결과는 이전 분석이다. 새 분석이 끝난 뒤 뷰어를 재시작한다.

```bash
./visualize.sh
```
