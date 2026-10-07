# 04 · 샷 추적, 전역 Entity, 장면 요약

원본 영상에서 대상을 추적하고, **샷 내부 목록 → 전체 영상의 공통 목록 → 공통 ID를 쓰는 장면 요약**을 만듭니다. 실제 분석 결과의 품질 판정은 각 실행의 `quality.json`과 `REPORT.html`에서 확인합니다. 실행 완료와 정확도 검증은 구분합니다.

## 실행과 시각화

```bash
cd ~/HayeonP/video_analysis/04_entity_extraction
./run.sh
./visualize.sh
```

VS Code의 **Ports**에서 **5004**를 전달하고 `http://localhost:5004`를 엽니다. `visualize.sh`는 같은 포트의 기존 서버를 종료하고 다시 시작합니다. 결과 페이지에서 원본/Adaptive 박스 영상, 사람/기타 Entity, 샷별 등장표, 보정 요약, 동작 리포트를 볼 수 있습니다.

```bash
./run.sh --input inputs/your_video --output outputs/your_result
./visualize.sh outputs/your_result
```

- 입력은 03 단계의 `summary.json`, 샘플 기록, Adaptive 조각 영상을 가진 폴더입니다. 매니페스트가 가리키는 원본 샷도 필요합니다.
- 입력을 생략하면 `inputs`에서 적합한 폴더가 하나일 때 자동 선택합니다.
- 출력 생략 시 같은 입력의 최근 `shot_story` 실행을 재개합니다. 없으면 `outputs/<입력 이름>_story`를 사용합니다.
- `outputs/latest_run.json`은 내보내기를 마친 결과를 가리킵니다. 실패한 부분 실행을 완료본으로 등록하지 않습니다.
- 기존 실험 방식은 `./run.sh --efficient` 또는 `--legacy`로 실행할 수 있습니다. 이전 설명은 [README_ANCHORED.md](README_ANCHORED.md)에 있습니다.

## 전체 구조

1. **원본 관측 수집:** YOLO11x + 타일 탐색, BoT-SORT + DINOv2로 연속 프레임의 박스를 연결합니다.
2. **전경 추적:** SAM 2.1이 원시 트랙마다 독립적으로 정방향/역방향 마스크를 구합니다.
3. **후보 정리:** 조각 내부의 명확한 컷/검은 전환을 넘은 근거 없는 마스크 전파를 제외합니다. 동일 마스크의 중복을 융합하고 혼합 가능성이 있는 시간 구간을 나눕니다. 원본 프레임은 삭제하지 않습니다.
4. **샷 내부 Entity:** VLM이 장면, 시간순 원본 crop, 동시 관측 비교 이미지를 보고 오검출·신체 부분·소품·독립 개체를 구별합니다. 샷별 목록과 초기 요약을 저장합니다.
5. **물리 트랙과 외형:** Local Entity별 SAM 추적을 정리하고, 전경 crop의 DINO 특징으로 샷 사이 비교 후보를 구합니다.
6. **전역 연결:** VLM의 정방향 검토와 역방향 검토로 공통 Entity를 만들고, 명백한 배경 오검출은 근거를 남겨 제외합니다.
7. **요약 보정:** 전체 목록과 앞뒤 요약, 실제 장면 이미지를 함께 보고 공통 ID로 요약을 보정합니다. 최대 10샷씩 처리합니다.
8. **저장·검증:** 대표 이미지, 등장표, 시간축이 보존된 영상, 초기/보정 요약, 오류·시간·호출 기록을 저장합니다.

마스크가 떨어져 있다는 사실만으로 다른 인물이라고 확정하지 않습니다. 얼굴과 팔은 같은 사람의 부분일 수 있습니다. 잡고 있는 도구나 붙잡힌 동물은 별도 개체입니다.

## 모델과 실행 자원

- 검출: `yolo11x.pt`
- 일반 외형: `facebook/dinov2-small` (고정 revision)
- 픽셀 영역 추적: `facebook/sam2.1-hiera-small` (고정 revision)
- 의미 검토 기본값: `Qwen/Qwen3-VL-32B-Instruct` (고정 revision, NF4)
- 이 PC에서는 24GB GPU 두 장에 큰 모델을 분산합니다. `run.sh`가 20GB 이상 GPU를 최대 두 장 선택합니다. 검출/추적은 첫 번째 GPU를 사용합니다.
- 명시한 `CUDA_VISIBLE_DEVICES` 또는 `ENTITY_CUDA_DEVICE`는 존중합니다. 첫 실행은 모델 다운로드와 초기화 시간이 추가됩니다.

```bash
ENTITY_CUDA_DEVICE=1,3 ./run.sh
```

## 유한 호출 예산과 캐시

기본 예산은 **실행 폴더당 실제 생성 24회**입니다. 실패, 재시도, 중단 후 재개도 같은 `qwen_budget.json` 원장에 포함합니다. 캐시 적중은 새 호출이 아닙니다. 예산을 소진하면 중단하며 자동으로 늘리거나 새 폴더로 도피하지 않습니다.

17개 샷은 샷별 검토 17회, 전역 검토 2회, 요약 보정 2회가 기본 구조입니다. 남은 예산은 재시도에 사용합니다. 더 많은 샷에는 처음 실행할 때 충분한 `--qwen-max-calls`를 지정해야 합니다. 진행 중인 실행의 예산은 바꿀 수 없습니다.

검출, 원본 추적, 마스크, crop 특징, VLM 응답을 단계별로 저장합니다. 입력·설정·crop·모델 revision·프롬프트가 달라지면 해당 결과를 새로 계산합니다. SAM의 로컬/HF 경로는 실제 체크포인트 내용이 같다고 검증한 경우만 같은 캐시로 취급합니다.

## 단계별 재개와 기록

전체 작업은 `run.sh` 한 번으로 수행합니다. 아래 옵션은 개발·점검을 위한 선택 사항입니다.

```bash
./run.sh --output outputs/your_result --stage tracking
./run.sh --output outputs/your_result --stage geometry
./run.sh --output outputs/your_result --stage local
./run.sh --output outputs/your_result --stage mask
./run.sh --output outputs/your_result --stage global
./run.sh --output outputs/your_result --stage narrative
./run.sh --output outputs/your_result --stage export
```

`run.log`, `status.json`, `execution_history.json`, `stage_timing.json`, `qwen_budget.json`에 진행 상태와 시간이 남습니다. 리포트는 모델 실행 시간, 분석 프로세스 시간, 개발/다운로드를 포함한 전체 경과 시간을 구분합니다.

## 독립 평가와 한계

영화 이름, 특정 등장인물 수, 사용자가 적어준 ID 병합 규칙은 추론에 사용하지 않습니다. 수동 부분 정답은 선택적 독립 평가에만 사용합니다.

```bash
./run.sh --input inputs/your_video --output outputs/your_result \
  --evaluate-against outputs/previous_result \
  --evaluation-labels tests/fixtures/manual_identity_groups.json
```

모든 대상에 정답이 있는 평가는 아닙니다. 아주 작은 대상의 동일성, 가림, 닮은 동물, 완전히 검게 되지 않는 디졸브는 불확실할 수 있습니다. 관측이 없는 등장표 0은 실제 부재를 보장하지 않습니다. 최종 결과의 `quality.json`과 시각적 검증 기록을 함께 확인해야 합니다.
