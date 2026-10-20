# 04 · 선택 프레임 분석, 전역 Entity, 장면 요약

**현재 기본값은 `--analysis-frames sampled`입니다.** 03의 `samples.json`에 나열된 이미지에만 검출·DINO 특징 추출·VLM 검토를 수행합니다. 원본 전체 영상은 디코딩하거나 추론하지 않습니다. 내부 컷 검사, 대표 이미지, 샷 간 비교도 선택된 이미지로 제한합니다. 선택 JPEG가 없으면 원본으로 우회하지 않고 실패합니다.

처음 구조를 이해하려면 **[단계별 쉬운 설명](MODEL_FLOW_SIMPLE.html)** 또는 **[2페이지 PDF](MODEL_FLOW_SIMPLE.pdf)**부터 읽으세요. 현재 sampled 경로 기준입니다.


## 현재 분석 상세 설명과 비교 뷰어

현재 기준 결과의 샘플링 이후 처리 과정을 [세 수준의 상세 보고서](CURRENT_ANALYSIS_DETAILED_REPORT.html)와 [PDF](CURRENT_ANALYSIS_DETAILED_REPORT.pdf)로 설명합니다. 중학생 비유 → 고등학생 원리 → 컴공 학부생 구현 상세 순서이며, 실제 모델·병합 조건·호출 수·시간·한계를 포함합니다.

```bash
./visualize_sintel.sh       # 고정된 기존 결과, 5004
./analyze_new_sintel.sh     # 별도 폴더의 batched 실험; 기존 결과를 덮어쓰지 않음
./visualize_new_sintel.sh   # 새 실험 결과만, 5006
```

기존 선택은 `outputs/baseline_sintel.json`, 새 선택은 `outputs/new_sintel.json`으로 분리합니다. 새 결과가 없으면 기존 결과로 대체하지 않습니다. 새 버전의 10회 상한은 샷별 추출 이후 전역 통합·외형·몸 보정의 실제 생성 합계이며, 로컬 검토와 요약 호출은 별도입니다. 이는 실험 경로이며 기존 정확도와의 동등성은 독립 검수 결과로 판단해야 합니다.

현재 비교용 실험(`sintel_batch_trial_13`)은 통합·외형·몸 보정 9회, 371.35초입니다. 이전 검출·샷별 추출을 재사용했으므로 신규 영상 전체 처리 시간이 아닙니다. 23개 관측의 부분 정답에서 오병합 14쌍이 남아 **기존 정확도 복구는 미완료**입니다. 기존 기준 결과는 변경하지 않았으며, 새 뷰어의 「인물 연결 복구 검수」에 수치와 범위를 표시합니다. 새 분석 스크립트는 이 실험 전략으로 처음부터 실행하고 진행률·전체 시간을 기록합니다.


## 영상별 원클릭 실행

```bash
cd /home/hayeon/3r_video/04_entity_extraction
./analyze_sintel.sh     # Sintel: 02 → 03 → sampled 04, 진행률·시간 기록
./visualize_sintel.sh   # Sintel 고정 비교 기준 결과: http://localhost:5004
./analyze_frozen.sh     # Frozen: 02 → 03 → sampled 04, 진행률·시간 기록
./visualize_frozen.sh   # Frozen 최근 완료 결과: http://localhost:5005
```

요청한 철자 `./anallyze_sintel.sh`도 동일하게 동작합니다. 분석은 매번 `outputs/<영상>_sampled_<시각>/`에 새로 저장합니다. Sintel 시각화는 고정된 기준 결과를 열고, Frozen은 최근 완료 결과를 엽니다. 새 Sintel 실험은 visualize_new_sintel.sh로 엽니다. 진행 중이거나 실패한 결과는 선택하지 않습니다. 두 뷰어는 서로 다른 포트를 사용합니다. 원격 VS Code에서는 해당 포트를 전달하세요. 분석 둘을 동시에 실행하면 GPU 경쟁으로 시간 측정이 왜곡될 수 있으므로 하나씩 실행하세요.

`--check`로 분석 없이 사전 검사할 수 있고, 분석 스크립트에 `--output /path/to/new_run`과 예산 옵션을 전달할 수 있습니다. 완료 후 `benchmark_summary.txt`에 전체/단계별 시간이, `entities/timing_summary.txt`에 04 내부 시간이 남습니다. 로그·JSON 상세 기록도 같은 결과 폴더에 저장됩니다.

## 중단된 분석 이어하기

```bash
./analyze_sintel.sh --resume   # Sintel의 최근 실패/중단 실행
./analyze_frozen.sh --resume   # Frozen의 최근 실패/중단 실행
# 특정 실행을 선택하려면
./analyze_sintel.sh --resume --output outputs/<기존_실행_폴더>
```

완료된 02/03 단계는 건너뛰고, 04의 유효한 분석 캐시는 재사용합니다. 입력 영상 체크섬·분석 모드·호출 예산이 다르면 거절합니다. 삭제된 중간 결과가 필요한 실행은 이어갈 수 없습니다. `--resume` 없이 실행하면 새 측정입니다.

모델이 응답 형식을 틀리면 **이전 응답과 구체적인 오류·허용값을 함께 전달하여 교정**합니다. 역할(`body_part`)을 개체 ID로 잘못 쓰는 경우도 처리합니다. 의미를 추측하여 강제로 병합하거나 검증 실패를 성공으로 바꾸지는 않습니다. 재시도 후에도 유효하지 않은 응답, GPU/파일 오류, 예산 부족은 원인을 기록하고 중단합니다.

이어하기의 `benchmark_summary.txt` 전체 시간은 **이전 실행 + 재개 실행의 실제 실행 시간 합계**이며, 중단 후 기다린 시간은 제외합니다. 새 영상의 중단 없는 1회 처리 시간과 구분하세요. 이전 요약은 `benchmark_attempts/`, 원래 로그와 재개 로그는 별도 파일로 보존됩니다.

## 전체 영상 한 번 실행하며 시간 측정

원본 영상을 직접 지정해 02 샷 분할 → 03 샘플링 → 04 sampled 분석을 실행할 수 있습니다. 가중치·환경 사전 검사가 끝난 뒤 진행하세요.

```bash
cd /home/hayeon/3r_video/04_entity_extraction
./benchmark_frozen.sh --video /absolute/path/to/video.mp4 \
  --output outputs/my_video_sampled
```

`benchmark_frozen.sh`는 호환용 이름이며 지정한 영상으로 실행합니다. 입력 파일은 저장소에 둘 필요가 없습니다. 완료 시 결과 폴더에 단계별 로그와 `benchmark_timing.json`, `benchmark_summary.txt`가 저장됩니다. 측정은 다운로드·설치를 제외하고 모델 적재·분석·출력 검증을 포함합니다.

```bash
./visualize.sh outputs/my_video_sampled/entities
```

## 이미 샘플링한 영상에 04만 실행

```bash
./run.sh --input /path/to/adaptive_output --output outputs/my_sampled_run \
  --analysis-frames sampled
# 다른 영상 입력으로 재실행할 때도 반드시 새 결과 폴더 사용
./run.sh --input inputs/sintel_adaptive --output outputs/sintel_sampled_new
```

04만 실행해도 진행률·시간 화면 출력과 파일 저장은 동일합니다. 04 시간은 02/03 및 영상 다운로드를 포함하지 않습니다. 입력·결과·모델·예산 기록이 바뀌면 캐시를 검증하고, 이전 dense 결과 폴더를 sampled 분석으로 덮어쓸 수 없습니다. 기본 결과 폴더 이름에도 `_sampled`가 붙습니다.

## 현재 sampled 구조

```text
03 선택 이미지 + 원본 프레임 번호·유지 시간
  → YOLO11x (+ 선택 이미지 내 작은 대상 타일)
  → DINOv2 외형·박스 겹침 + 실제 시간 간격을 사용하는 일대일 연결
  → 선택 이미지 내 컷·대상 교체 후보 보정
  → Qwen3-VL-8B-Instruct: 샷 내부 Local Entity·분류·소유관계
  → Qwen3-VL-8B-Thinking + 동시 등장·외형 모순 제약: Global 연결
  → Instruct: Global ID로 요약 보정
  → 선택 관측 유지 영상·Entity 목록·등장표·검증·뷰어
```

연속 프레임을 가정한 BoT-SORT 대신, 실제 경과 시간으로 만료되는 보수적 샘플 간 연결을 사용합니다. 외형·위치의 일대일 대응은 임시 트랙이며 동일성 확정은 뒤의 시각 검토가 담당합니다. 끊어진 연결은 Local/Global 단계에서 재검토할 수 있습니다. 샘플 사이의 등장·움직임은 알 수 없으므로 전체 정확도 보장은 하지 않습니다.

SAM video 전파는 전체 프레임 접근이 필요해 sampled 모드에서 허용하지 않습니다. `--mask-refinement`를 함께 주면 오류로 안내합니다. 선택 샘플의 원본 프레임 번호와 시간은 모든 관측에 유지하며, 샘플 바깥 근거가 들어오면 검증에서 거절합니다.

**출력은 Adaptive 영상만 생성합니다.** 관측 이미지와 박스를 다음 선택 시각까지 유지하고 `SAMPLED observation held`를 표시합니다. 원본 전체 프레임에 박스가 관측된 것처럼 채우지 않습니다. 원본 전체 박스 영상은 생성하지 않습니다. `tracking_summary.json`의 `analyzed_frames`가 실제 선택 프레임 수와 일치하는지 확인할 수 있습니다. `dense_tracking/`이라는 저장 경로명은 이전 코드 호환을 위한 이름이며 실제 처리 범위는 `analysis_frames`에 기록됩니다.

## 기존 보고서와 결과

이전에 검수한 `sintel_global_v2` 및 **[9페이지 학습 보고서](PIPELINE_STUDY_REPORT.html)** · **[PDF](PIPELINE_STUDY_REPORT.pdf)** · **[1페이지 보고서](GLOBAL_IDENTITY_REPORT.html)**는 **원본 전체 분석(dense) 결과**입니다. 새 sampled 방식의 품질·소요 시간은 직접 실행한 뒤 확인해야 합니다. 기존 결과와 입력은 보존합니다.

뷰어는 `./visualize.sh`로 최근 완료 결과를 엽니다. 기본 주소는 `http://localhost:5004`이고 원격 VS Code에서는 포트 5004를 전달합니다.

## 이전 dense 경로의 구조 (명시적으로 `--analysis-frames dense`를 지정할 때만)

1. **원본 관측 수집:** YOLO11x + 타일 탐색, BoT-SORT + DINOv2로 연속 프레임의 박스를 연결합니다.
2. **전경 추적:** SAM 2.1이 원시 트랙마다 독립적으로 정방향/역방향 마스크를 구합니다.
3. **시간 구간 정리:** 마스크 사용 여부와 무관하게 조각 내부의 컷·검은 전환과 지속적인 외형/박스 변화를 검사합니다. 대상이 바뀌었을 가능성이 있는 트랙을 구간 후보로 나누되 같은 인물로 재결합할 수 있습니다. 검출 관측은 보존하며, 마스크가 있으면 근거 없는 전파와 중복도 정리합니다.
4. **샷 내부 Entity:** VLM이 장면, 시간순 원본 crop, 동시 관측 비교 이미지를 보고 오검출·신체 부분·소품·독립 개체를 구별합니다. 검출과 VLM의 사람↔동물 분류 충돌, 두 대상 비교에서 제안된 종류 변경은 재검토 후보로만 씁니다. 대상 하나만 표시한 원본 최대 3프레임으로 확인한 뒤 해당 트랙만 수정합니다. 같은 화면의 위치 차이도 재검토 후보이며, 실제로 다른 신체라는 시각 근거가 있어야 병합금지 제약으로 사용합니다. 샷별 목록과 초기 요약을 저장하고 미해결 경고를 남깁니다.
5. **물리 트랙과 외형:** Local Entity별 SAM 추적을 정리하고, 전경 crop의 DINO 특징으로 샷 사이 비교 후보를 구합니다.
6. **전역 연결 (`linked`, 기본):** 여러 샷의 후보를 기존 쌍별 시각 근거 방식으로 연결합니다. Thinking은 먼저 1,024 토큰으로 비교하고, 검증 후 미확정인 샷 간 비교만 동일한 원본 이미지로 긴 추론을 한 번 더 수행합니다(응답 전체 최대 8,192 토큰). 두 단계의 재시도를 합쳐 실제 전역 생성은 최대 96회입니다. 후처리 `catalog_repair.py`는 기존 연결을 유지하며 확인된 외형 모순과 실제 충돌 프레임만 수정합니다. Instruct가 같은 몸의 부분/중복과 별도 사람을 구분하고, ID 변경·낮은 겹침의 몸 병합은 Thinking으로 추가 확인합니다. 두 모델을 한 번씩 모아서 실행하므로 프레임마다 모델을 바꾸지 않습니다. 확인된 같은 몸의 박스만 하나로 통합하며 원본 관측을 보존합니다. 사람 후보에 새 미확정 ID가 생기면 인물 참조 사진 없이 해당 영역을 확대해 몸의 수를 다시 확인합니다. 같은 몸의 확대 박스는 통합하고, 확인된 그림자·반사·배경 오검출은 이유를 기록해 제외하되 원시 관측은 남깁니다. 동물 관측에는 이 사람 전용 보정을 적용하지 않습니다. 이름은 연결과 별도로 정하고 설명 길이로 선택하지 않습니다. 기존 5회 `compact`는 분리 과다 문제가 있어 기본에서 제외했습니다.
7. **요약 보정:** 전체 목록과 앞뒤 요약, 실제 장면 이미지를 함께 보고 공통 ID로 요약을 보정합니다. 최대 10샷씩 처리합니다. 미확인 부분 ID를 언급한 자동 서술은 원문을 보존하고 관측 목록으로 대체해, 별도 인물을 만들어 내지 않게 합니다.
8. **저장·검증:** 대표 이미지, 등장표, 시간축이 보존된 영상, 초기/보정 요약, 오류·시간·호출 기록을 저장합니다.

마스크가 떨어져 있다는 사실만으로 다른 인물이라고 확정하지 않습니다. 얼굴과 팔은 같은 사람의 부분일 수 있습니다. 잡고 있는 도구나 붙잡힌 동물은 별도 개체입니다.
소유자가 확인되지 않은 신체 부분·착용물은 관측을 유지하되 목록에서 별도로 표시합니다. 소유관계는 같은 프레임의 관측된 몸 영역과 겹치는 경우에만 시각 검토하며, 겹침 자체를 동일성 증거로 쓰지는 않습니다. 원래 관측 이름은 `observation_label`에 보존하고, `identity_scope=part_owner_unresolved`와 미확정 상태를 기록합니다.

## 모델과 실행 자원

- 검출: `yolo11x.pt`
- 일반 외형: `facebook/dinov2-small` (고정 revision)
- 선택적 얼굴 보조: `01_low_freq_event/models/yunet.onnx`, `sface.onnx`. 설치된 경우 얼굴 관측 선택·확대와 비교 후보 순위에 사용합니다. 얼굴 점수로 자동 병합하지 않으며, 없으면 일반 외형 후보와 원본 crop으로 동작합니다.
- 픽셀 영역 추적: `facebook/sam2.1-hiera-small` (고정 revision)
- 로컬 Entity·요약: `Qwen/Qwen3-VL-8B-Instruct` (`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, 메모리에 따라 NF4)
- 새 실행의 전역 동일성 비교: `Qwen/Qwen3-VL-8B-Thinking` (고정 revision `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`). Linked 후보 비교의 첫 판단과 충돌 재확인은 추론 과정 1,024 토큰 이후 답변으로 전환합니다. 후보 비교가 미확정이면 같은 시각 근거로 긴 추론을 한 번 추가하며, 모든 응답은 동일한 근거 검증을 통과해야 합니다. Compact 실험 옵션은 2,048 토큰입니다. 이 제한은 별도 모델 호출을 추가하지 않습니다. 설치된 NF4 변환 가중치를 사용합니다.
- 단일 RTX 5080 16GB에서 실행할 수 있도록 전체 이미지 입력량을 제한합니다. 큰 모델은 명시적으로 지정할 수 있습니다. 검출/추적은 첫 번째 GPU를 사용합니다.
- 표준 Hugging Face 캐시를 공유하며, 지정한 `HF_HOME`을 존중합니다. RTX 50 시리즈에는 해당 GPU를 지원하는 PyTorch CUDA 빌드가 필요합니다(이 환경: PyTorch 2.7.1+cu128).
- 명시한 `CUDA_VISIBLE_DEVICES` 또는 `ENTITY_CUDA_DEVICE`는 존중합니다. 첫 실행은 모델 다운로드와 초기화 시간이 추가됩니다.

기존 실행은 저장된 모델을 유지합니다. `--global-model-id`와 `--global-model-revision`으로 전역 비교 모델만 교체할 수 있습니다. `--model-id`를 명시하면 별도 전역 모델을 지정하지 않는 한 해당 모델을 두 단계에 사용합니다.

```bash
ENTITY_CUDA_DEVICE=1,3 ./run.sh
```

## 유한 호출 예산과 캐시

기본 linked 예산은 **실행 폴더당 실제 생성 320회**입니다. 실패, 재시도, 중단 후 재개도 같은 `qwen_budget.json` 원장에 포함합니다. 캐시 적중은 새 호출이 아닙니다. 예산을 소진하면 중단하며 자동으로 늘리거나 새 폴더로 도피하지 않습니다.

17개 샷은 샷별 검토 17회, 샷 내부 소유관계 검토 최대 4회, 검출·의미 분류 충돌 재검토 최대 4회, 추적 전환 후보의 외형 확인 최대 4회, 전역 후보 비교 최대 96회 및 필요한 충돌 프레임 보정, 요약 보정 5회(4샷 이하 묶음)가 기본 구조입니다. 전역 검토는 `--global-review-max-calls`로 제한하며 재시도·재개 전 사용량도 포함합니다. 더 많은 샷에는 처음 실행할 때 충분한 `--qwen-max-calls`를 지정합니다. 일반 재개로는 예산을 변경할 수 없습니다. 개발 중 추가 검수가 필요하면 `--extend-qwen-budget-to 120 --budget-extension-reason '추가 검수 사유'`로 명시적으로 늘릴 수 있으며, 이전 한도·누적 사용량·변경 사유를 같은 원장에 보존합니다.

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

특정 등장인물 수나 수동 ID 병합 규칙은 추론에 사용하지 않습니다. 선택적 `--context`는 영상 제목·매체 유형·출처가 있는 짧은 소개만 받습니다. 소개는 일반 LEFT/RIGHT 비교를 포함한 시각 검토의 약한 문맥으로만 사용합니다. 시각적 근거를 대신하거나 등장하지 않은 대상을 생성할 수 없습니다. 수동 부분 정답은 독립 평가에만 사용합니다.

```bash
# 공개 소개문을 수집한 뒤, 다른 영상에도 같은 형식으로 사용합니다.
python video_context.py --title '영상 제목' --media-type 'film trailer' \
  --url 'https://example.org/film' --output inputs/your_video/video_context.json
./run.sh --input inputs/your_video --context inputs/your_video/video_context.json

# 다른 PC의 절대 경로를 포함한 입력: 원본 프레임 구간으로 샷을 복원합니다.
./run.sh --input inputs/your_video --original-video /path/to/original.mp4
```

`--skip-mask-refinement`는 SAM 마스크 단계 없이 검출/추적 관측으로 실행합니다. 시간 구간 보정과 동시 대상 분리 검사는 이 경로에서도 적용됩니다. 기존 출력 재개 시 모델·검출기·원본 영상·문맥·마스크 사용 설정을 유지합니다. 마스크 사용을 바꿀 때는 `--stage all`로 의존 결과를 다시 만들며, 켜는 옵션은 `--mask-refinement`입니다. Local 내용·tracking·문맥이 달라지면 narrative/export 전에 `--stage global`을 다시 실행해야 합니다. 시간 구간 변경은 `raw_temporal_repair/summary.json`, 새 전역 검토 이력은 `compact_identity/audit.json` (이전 pairwise 실행은 `identity_reconciliation/audit.json`), 실제 비교 이미지는 같은 폴더의 배치별 하위 폴더에 저장됩니다. 미검토/불확실 후보는 성공으로 숨기지 않고 `quality.json`에 표시합니다.

```bash
./run.sh --input inputs/your_video --output outputs/your_result \
  --evaluate-against outputs/previous_result \
  --evaluation-labels tests/fixtures/manual_identity_groups.json
```

모든 대상에 정답이 있는 평가는 아닙니다. 아주 작은 대상의 동일성, 가림, 닮은 동물, 완전히 검게 되지 않는 디졸브는 불확실할 수 있습니다. 관측이 없는 등장표 0은 실제 부재를 보장하지 않습니다. 최종 결과의 `quality.json`과 시각적 검증 기록을 함께 확인해야 합니다.

기존 결과의 샷 간 연결을 유지하면서 보정할 때는 아래 명령을 사용합니다. 입력 결과는 변경하지 않습니다. `--profile-cache`로 같은 영상의 기존 외형 검토 결과를 지정하면 해당 추론도 재사용하며 인용 이미지 바이트를 검증합니다. 없으면 의심 추적 구간의 외형을 최대 4회 확인합니다. 진행률과 `repair_execution_history.json`에 시간·재시도를 기록합니다.

```bash
./repair_entities.sh --source outputs/기존결과/entities --output outputs/복구결과/entities
```

복구는 이전 연결의 모든 오류를 찾아내는 전면 재분석이 아닙니다. 충돌이 없는 잘못된 연결이나 기존 미연결이 남을 수 있으며, 새 영상 전체 시간과 복구 시간은 구분해야 합니다. 검토에 실패하면 결과를 완료로 표시하지 않습니다.


### 새 결과의 맥락 보정 (2026-10-14)

새 `batched` 분석은 선명한 복장·얼굴 참조, 반복 등장 빈도, 실제 샷 경계의 연속성을 함께 사용합니다. 작은 동물은 기존 DINO 특징으로 서로 가장 닮은 관측과 시점을 찾고, VLM으로 해부학적 충돌을 확인한 뒤 해당 관측만 재배정합니다. 전체 샷 요약은 한국어 여부를 검사하여 내보냅니다.

- [중학생 → 고등학생 → 컴공 학부생 설명](CONTEXT_ANALYSIS_REPORT.html) · [PDF](CONTEXT_ANALYSIS_REPORT.pdf)
- `./analyze_new_sintel.sh`: 빈 결과 폴더에서 02→03→04 실행. 영상 종류는 `inputs/trailer_context.json`에서 읽습니다.
- `./visualize_new_sintel.sh`: 새 비교 결과, 포트 5006. 기존 `./visualize_sintel.sh` 결과는 5004에 유지합니다.
- 전역 통합·외형·몸 보정·문맥 검토의 실제 생성은 기본 최대 10회입니다. 샷별 추출을 포함한 **전체 Qwen 호출 수는 별도**이며, 재시도도 예산에 포함됩니다.
- 실행 시간은 `benchmark_timing.json`, 단계 시간은 `entities/run_timing.json`, 모델별 생성 시간은 `entities/qwen_budget.json`에 남습니다.
- `context_reassignment/person.json`, `animal.json`, `shot_memory/memory.json`에서 입력 근거와 결정을 확인할 수 있습니다. 동물 모델에는 잘못된 종 이름의 영향을 피하려고 전체 요약 대신 한국어 전환 기억을 전달합니다.

다른 영상에도 같은 로직을 적용할 수 있습니다. 영화명이나 특정 샷의 정답을 추론에 넣지 않습니다. 많은 개체·긴 영상은 기본 호출 상한을 초과할 수 있으며, 모든 영상에서 같은 정확도를 보장하지 않습니다.

```bash
../01_low_freq_event/.venv/bin/python benchmark_video.py \
  --video /absolute/path/video.mp4 --global-strategy batched
```
