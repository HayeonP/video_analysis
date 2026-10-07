# 04_entity_extraction

과정·실제 결과·새 agent 설계 제안을 PPT 형태로 보려면 [HTML 검토 슬라이드](ENTITY_EXTRACTION_SLIDES.html)를 여세요.
23장으로 구성되어 있으며 이미지와 폰트를 포함합니다. [슬라이드 사용법](presentation/README.md)

03 단계의 adaptive sampling 결과에서 인물·동물/캐릭터·주요 물체를 찾고,
동일한 대상으로 판단되는 관찰을 전역 entity ID로 묶습니다.
개체별 대표 이미지와 레이블을 저장하고, 모든 샷을 이어 붙인 adaptive 영상에 bounding box를 표시합니다.

현재 기본 방식(v2)은 **샷 요약·entity 이력 누적 → 독립 역방향 매칭 → 문맥을 함께 보는 최종 재검토**입니다.
변경점과 회귀 테스트 결과는 [문맥 매칭 수정 보고서](CONTEXT_FIX_REPORT.md)를 참고하세요.
기존 HTML 슬라이드는 v1 실행과 설계 제안의 기록입니다.
사용법과 현재 알고리즘은 [자동 실행 및 양방향 매칭 설명](SEQUENTIAL_MATCHING.md)을 참고하세요.
이전 temporal 방식의 세부 동작·수식·실행 시간 기록은 [이전 리포트](ENTITY_EXTRACTION_REPORT.md)에 보존했습니다.

## 실행

### 웹에서 결과 보기 (VS Code Remote)

```bash
cd ~/HayeonP/video_analysis/04_entity_extraction
./visualize.sh
```

VS Code 하단 **포트(Ports)** 탭에서 **5004**를 전달하고, 브라우저에서
`http://localhost:5004`를 여세요. 전달된 로컬 포트가 다르면 Ports 탭의 **브라우저에서 열기**를 사용하세요.
첫 화면 **분석 과정 이해**에서는 현재 sequential v2의 전체 흐름 → Agent 구조 → DL 모델의 역할 →
샷 요약·기억 → 최종 문맥 판정 → 산출물·캐시를 위에서부터 설명합니다.
ON/OFF 흐름 선택, 단계별 입출력, 샷별 기억 갱신 예시, 검토 범위 전환을 눌러 볼 수 있습니다.
**영상 · Entity 결과** 탭에서는 저장된 결과를 봅니다. 결과가 없는 상태에서도 과정 설명을 열 수 있습니다.
상단 **이전 v1 PPT 기록**은 과거 버전의 실행·설계 기록입니다.

기본 결과는 `outputs/sintel_sequential`이며, Qwen ON/OFF 선택, 전체 박스 영상,
비교 영상, entity 대표 이미지·설명, 샷 이동, 등장 행렬과 다운로드를 제공합니다.
현재 구현 설명은 `viewer_guide.html`이며 웹 경로 `/guide`로도 직접 열 수 있습니다.

```bash
# 다른 결과 또는 다른 포트
./visualize.sh outputs/sintel_temporal --port 5005
```

Python 표준 라이브러리만 사용하며 모델 추론 없이 저장된 결과를 표시합니다.
기본 주소는 `127.0.0.1`입니다. 지정한 결과의 공개용 파일만 제공하며 디렉터리 목록,
입력 영상 폴더, 추론 캐시와 소스 파일은 웹에 노출하지 않습니다.
종료는 `Ctrl+C`, 분석을 다시 실행한 뒤에는 뷰어를 재시작하세요.
`./visualize.sh`는 지정한 포트(기본 5004)를 점유한 기존 프로세스를 자동 종료하고 새로 시작합니다.
기존 뷰어 이외의 프로그램도 해당 포트를 점유하면 종료 대상입니다. 종료 요청 후 1초 내에
포트를 놓지 않으면 강제 종료합니다. 운영체제의 프로세스 조회·종료 권한이 없으면 오류를 표시합니다.

### 분석 실행

```bash
cd ~/HayeonP/video_analysis/04_entity_extraction
./run.sh
```

기본 실행은 `--mode both`입니다. **Qwen 사용/미사용 결과를 별도 폴더에 저장하고 나란히 보기 영상도 생성**합니다.

기본 경로:

- 입력: `inputs/sintel_adaptive/`
- 출력: `outputs/sintel_sequential/`

```text
outputs/sintel_sequential/
  without_qwen/                 YOLO + 외형 매칭만 사용
  with_qwen/                    검출 보완 + 누적 기억 + 역방향 매칭 + 최종 검토
  qwen_comparison_full.mp4       왼쪽 OFF / 오른쪽 ON
  comparison_summary.json        동일 입력 여부와 모드별 수치
```

각 하위 폴더에 `entities.json`, `entity_gallery.jpg`, `adaptive_entities_full.mp4`,
`entity_shot_presence.csv`(entity × shot의 0/1 행렬), `shot_entities.json`과 관찰 기록을 저장합니다.
Qwen ON의 `sequential/direction_comparison.json`에는 두 방향의 일치·불일치 쌍을 저장합니다.
**ID는 모드마다 독립적으로 배정하므로, 같은 ID 문자열이나 색상이 모드 사이의 동일 개체를 뜻하지 않습니다.**

한 가지 모드만 실행하거나 이전 Qwen 분석을 재사용할 수도 있습니다.

```bash
./run.sh --mode without_qwen --output outputs/yolo_only
./run.sh --mode with_qwen --output outputs/qwen_enabled

# 기존 Qwen 결과와 동일한 입력·설정이면, 시간이 걸리는 추론 캐시 재사용
./run.sh --mode both --reuse-qwen outputs/sintel_temporal/with_qwen \
  --output outputs/sintel_rerun

# 이전 crop 갤러리만 이용하는 방식을 재현하려면
./run.sh --qwen-identity gallery --output outputs/gallery_only

# 이전 연속 3개 샷 방식
./run.sh --qwen-identity temporal --output outputs/temporal_rerun
```

`--reuse-qwen`은 입력 해시, 타임라인, 설정, 검출 프롬프트와 배치 크기가 일치할 때만 허용합니다.
미사용 경로에 공유하는 데이터는 Qwen을 거치기 전의 YOLO 후보뿐입니다.
미사용 경로의 박스·분류 레이블은 YOLO에서 직접 가져오고, 외형 특징과 ID는 별도로 계산합니다.
Qwen을 로드하거나 Qwen의 검출 결과·설명·병합 결과를 읽어 사용하지 않습니다.

### 이전 Sintel 비교 결과 기록

당시 결과 폴더 `outputs/sintel_comparison`은 출력 정리 과정에서 삭제했습니다. 아래는 당시 수치 기록입니다.

두 결과 모두 같은 17개 샷, 279개 선택 프레임, 52.208333초 타임라인을 사용합니다.

| 항목 | Qwen OFF | Qwen ON |
|---|---:|---:|
| 프레임별 검출 관찰 합계 | 144 | 169 |
| 검출이 있는 프레임 수 | 127 | 145 |
| Qwen ID 검토 전 ID 수 | 50 | 69 |
| Qwen 추가 ID 병합 수 | 0 | 26 |
| 최종 자동 ID 수 | 50 | 43 |

ON의 ID 병합은 **69 → 43**입니다. OFF의 50개 ID에서 7개만 줄인 결과가 아닙니다.
첫 검출 단계부터 달라지기 때문에 두 경로의 초기 ID도 다릅니다.
최종 ID 수는 실제 정답 개체 수나 정확도가 아닙니다. 오검출 및 동일 개체가 여러 ID로 나뉘는 경우가 남을 수 있습니다.
Qwen ON에는 `dragon` 분류가 생기지만, OFF는 YOLO의 `bird`, `airplane` 같은 고정 클래스만 사용합니다.
분류가 더 구체적이라고 박스나 동일성 매핑까지 정확하다는 뜻은 아닙니다.

이번 ON 결과는 기존 추론 캐시를 재사용했습니다. 두 실행의 소요 시간을 추론 성능 차이로 비교하면 안 됩니다.

### 시간순 매칭 적용 결과: `outputs/sintel_temporal`

- Qwen OFF: 50개 ID / 144개 관찰.
- Qwen ON + 시간순 검토: 36개 ID / 169개 관찰.
- ON은 샷별 초기 ID 72개에서 샷 내부 crop 검토·시간순 검토·추가 후보 재판정을 수행했습니다.
- 두 출력 모두 279개 샘플, 52.208333초이며 샘플 PTS와 전체 유지 구간을 보존했습니다.

작은 용의 확인한 예시:

| 원본 시각 | 모습 | 이전 ID | 새 ID |
|---|---|---|---|
| 26.375초 / shot 0010 | 초기 비행 | object_0007 | object_0007 — 아직 분리됨 |
| 30.208초 / shot 0012 | 정면에서 보이는 작은 용 | object_0027 | object_0025 |
| 34.375초 / shot 0013 | 붙잡힌 작은 용 | object_0032 | object_0025 |
| 35.208초 / shot 0014 | 다음 샷의 작은 용 | object_0033 | object_0025 |
| 27.250초 / shot 0010 | 큰 생물 | object_0013 | object_0013 — 작은 용과 별도 |

`with_qwen/matching_examples.jpg`와 `matching_examples.json`에 이 예시를 저장했습니다.
**초기 비행까지 모든 모습을 하나의 ID로 통합한 결과는 아닙니다.**
4B 모델의 관점 변화 판정, 겹친 검출 박스 및 오분류 때문에 분리된 ID가 남습니다.
위 예시 확인은 전체 데이터의 정답률 평가가 아닙니다. ID 수가 줄어든 것 자체를 정확도 향상으로 해석하면 안 됩니다.

다른 입력이나 새 결과 폴더:

```bash
./run.sh --input inputs/sintel_adaptive --output outputs/sintel_entities_v2
```

같은 설정과 같은 출력 폴더로 실행하면 저장된 검출·특징·모델 응답을 재사용합니다.
중단 후에도 같은 명령으로 이어서 실행할 수 있습니다. 입력 영상/메타데이터나 설정이 달라지면 새 결과 폴더를 사용하세요.
한 출력 폴더에 동시에 두 프로세스가 쓰는 것은 차단합니다.

### 실행 시간

다음 실행부터 각 단계, ON/OFF 모드, 비교 영상 생성 및 **전체 작업 시간**을 터미널에 표시하고,
출력 루트의 `timing.json`에 저장합니다. 입력 디코딩·캐시 준비도 포함하며,
전체 측정은 무거운 Python 라이브러리 로딩 전부터 시작합니다.
같은 폴더로 재실행하면 시간 기록은 최신 실행 값으로 갱신됩니다.
모드별 시간은 하위 단계 시간을 포함하므로 중복 합산하지 마세요.

현재 보존된 마지막 실행은 파일 시각 기준 **약 3분 11초**입니다.
기존 타이머의 ON 처리 134.73초와 OFF 처리 30.66초는 입력 준비·캐시 준비·나란히 비교 영상 생성을 제외한 값입니다.
캐시를 재사용한 실행이며, 전체 개발·반복 실험 시간과는 다릅니다.
기존 시간의 근거는 `outputs/sintel_temporal/previous_run_timing.json`에 저장했습니다.

## 환경

기존 `01_low_freq_event/.venv`의 Python과 모델 코드를 사용합니다.
`01_low_freq_event`와 `03_adaptive_sampling` 폴더가 현재 위치에 있어야 합니다.
01의 코드를 수정하지 않고 import하여 재사용합니다.

- GPU: 기본 물리 GPU 1번(현재 PC의 RTX 4090).
- 변경: `ENTITY_CUDA_DEVICE=3 ./run.sh`.
- 기존 `CUDA_VISIBLE_DEVICES`가 설정되어 있으면 그 설정을 우선합니다.
- 다른 Python 환경: `ENTITY_PYTHON=/path/to/python ./run.sh`.
- Qwen과 DINOv2 가중치: 이 프로젝트의 `.cache/huggingface/`.
- YOLO·OSNet·YuNet·SFace 가중치: `01_low_freq_event/models/`.
- 한국어 폰트: `assets/NanumGothic-Regular.ttf`, 라이선스 `assets/OFL.txt`.

Qwen3-VL은 CUDA GPU가 필요합니다. 최초 실행 시 없는 모델 가중치를 다운로드합니다.

## 처리 방식

아래는 `with_qwen`의 전체 흐름입니다. `without_qwen`은 Qwen 검출 보완과 Qwen ID 검토를 건너뜁니다.
두 모드의 입력 샘플, YOLO 후보, 검출 임계치와 외형 매칭 설정은 동일합니다.

### Qwen을 적용하는 정확한 위치

```text
공통: adaptive 프레임 → YOLO11s 후보

OFF: YOLO 후보 확정
       → SFace / OSNet-AIN / DINOv2 특징
       → 외형 기반 ID 연결 → 출력

ON:  Qwen이 프레임 + YOLO 후보를 보고 최종 검출 목록 작성
       → SFace / OSNet-AIN / DINOv2 특징
       → 외형 기반 ID 연결
       → 샷 경계에서 ID 분리 (샷별 초기 ID)
       → Qwen이 같은 샷의 crop 갤러리로 ID 병합 검토
       → Qwen이 시간순 전체 장면으로 샷 간 ID 재검토 → 출력
```

**첫 번째 Qwen 호출: 프레임별 검출 보완**

- 모델: `Qwen/Qwen3-VL-4B-Instruct`, revision은 `run_config.json`에 저장합니다.
- 입력: 선택된 전체 화면 이미지 최대 4장과 각 이미지의 YOLO 클래스·박스·confidence.
- 이미지 처리는 장당 최대 196,608픽셀을 사용합니다. 영상 전체를 한 번에 전달하지 않습니다.
- 기본 NF4 4bit로 실행하며, 생성은 `do_sample=False`입니다.
- 반환: 각 프레임의 `kind`, `category`, 한국어 `label`, `appearance`, `bbox`, `uncertain`.
- 추가 검출만 반환하는 방식이 아닙니다. **최종 전체 검출 목록을 반환하므로 YOLO 후보를 추가·수정·제거할 수 있습니다.**
- `bbox`는 0~1000 정규화 좌표입니다. 같은 인물/비인물 구분에 속한 YOLO 후보와 IoU 0.55 이상이면
  실제 좌표는 YOLO의 좌표를 사용합니다. 종류 설명은 여전히 Qwen이 수정할 수 있습니다.
- 대응 YOLO 후보가 없으면 Qwen 좌표를 사용하고, 검출 confidence는 임의로 만들지 않고 null로 기록합니다.
- 같은 프레임의 중복 박스를 제거한 뒤, 이 최종 crop으로 기존 외형 특징을 계산합니다.

프롬프트는 실제 보이는 독립적인 인물·동물·주요 물체만 찾도록 지시하며,
장면 배경·텍스트·옷·신체 일부를 별도 대상으로 만들지 않도록 합니다. 이 지시가 모델의 오검출을 보장해서 막는 것은 아닙니다.
허용된 프레임 키, 종류와 박스 범위는 구조 검사를 거치며, 잘못된 JSON은 기본 최대 2회 재시도합니다.

**두 번째 Qwen 호출: 전역 ID 병합 검토**

아래는 `--qwen-identity gallery` 및 이전 결과의 방식입니다.
새 기본 `temporal`에서는 샷 내부 crop 검토만 사용하고, 샷 사이 crop만으로 하는 병합은 실행하지 않습니다.

- 외형 매칭이 만든 ID마다 최대 2개의 crop을 모아 ID가 적힌 격자 이미지를 만듭니다.
- 격자 이미지, ID 목록·종류·레이블, **병합 금지 쌍**을 전달합니다.
- 먼저 샷 내부의 ID를 비교하고, 그다음 줄어든 후보들을 샷 사이에서 비교합니다.
- 요청 단위로 최대 약 10개의 ID를 비교합니다. 격자 이미지의 최대 처리 크기는 786,432픽셀입니다.
- 반환 형식은 `{"groups":[{"ids":["person_0001","person_0002"],"reason":"시각적 근거"}]}`입니다.
- 동시에 검출된 두 ID 또는 서로 다른 종류의 ID를 묶는 응답은 거부합니다.
- 허용된 그룹에 대해서만 하나의 전역 ID로 통합합니다. 이 단계는 새 박스를 만들거나 박스를 수정하지 않습니다.
- 16GB 이상 GPU에서는 동일 모델을 BF16/FP16으로 사용하고, 작은 GPU에서는 NF4를 사용합니다.

따라서 ON/OFF 차이는 단순한 레이블 차이가 아닙니다. Qwen이 박스를 바꾸면
외형 특징을 추출하는 crop 자체가 달라지고, 첫 ID 연결과 후단 병합 결과에도 영향을 줍니다.
기존 `sintel_comparison`은 위 두 Qwen 개입을 함께 켜고 끈 비교입니다.
새 기본 `sintel_temporal`은 아래 시간순 장면 검토까지 포함합니다.

### 시간순 장면을 이용한 샷 간 매칭

**이전에도 샷 사이 매칭은 했습니다.** 하지만 외형 특징과 대표 crop만 비교했으므로,
작은 생물이 날다가 다른 샷에서 붙잡혀 뒤집히는 사건의 연결은 활용하지 못했습니다.
또 같은 생물이 `person`, `bird`, `dragon`으로 잘못 분류되면 종류 제약 때문에 병합이 막혔습니다.

추가로 이전 `object_0005`에는 shot 0010의 큰 생물과 shot 0012의 작은 생물이 이미 섞여 있었습니다.
이런 ID는 병합만으로 고칠 수 없습니다. 따라서 새 기본 방식은 외형 연결 결과를 **(ID, 샷)** 단위로
먼저 나누고, 샷 내부의 연결만 초기값으로 남깁니다. 기존 ID를 그대로 믿고 다른 ID를 덧붙이지 않습니다.
초기 분리 기록은 `shot_identity_seeds.json`에 저장합니다. 샷 간 병합은 아래 시간순 검토에서 수행합니다.

새 단계의 컨셉은 **앞뒤 장면을 함께 보고, 외형과 행동 관계가 이어지는 관찰을 같은 ID로 연결**하는 것입니다.
예를 들어 잡는 큰 생물과 잡히는 작은 생물을 구별한 상태에서,
날고 있는 작은 생물과 다음 샷에서 붙잡힌 작은 생물의 외형·상황을 함께 비교합니다.

1. 입력 순서대로 연속 3개 샷을 묶습니다. 다음 요청은 한 샷 이동하므로 2개 샷이 겹칩니다.
2. 각 후보 ID의 큰 관찰 crop에 해당하는 **전체 화면**을 반드시 포함하고,
   각 샷의 시작·중간·끝 샘플도 넣습니다. 시간순으로 정렬하고 상자·ID·원본 시각을 표시합니다.
3. 이를 Qwen의 **video 입력**으로 전달하고, 같은 ID의 확대 crop 갤러리도 함께 줍니다.
   작은 대상의 외형은 crop으로, 주변 사건은 video로 비교하도록 합니다.
   불규칙한 샘플 시각을 유지하며 모델 내부 재샘플링을 끕니다.
   문맥 프레임은 960×540으로 작성되고 모델 processor가 설정된 픽셀 예산에 맞춰 처리합니다.
   이 문맥용 프레임 선택은 출력 영상의 279개 샘플을 제거하거나 변경하지 않습니다.
4. Qwen은 얼굴·머리·뿔·날개·의상 같은 외형과, 이동·접촉·행동의 순서를 함께 비교합니다.
   이전 검출 레이블을 이 요청에 전달하지 않으며, ID 접두사도 정답으로 취급하지 않습니다.
5. 먼저 연결 가능성이 있는 ID 쌍과 간단한 근거를 제안받습니다.
   존재하지 않는 ID가 포함된 제안은 제외합니다. 큰 그룹을 제안한 경우 두 ID씩 분해합니다.
   아직 병합하지 않고, 시간적으로 가까운 관찰 쌍부터 상세 검토합니다.
6. **전체 영상에서 동시에 관찰된 서로 다른 ID는 병합할 수 없습니다.**
   이 제약은 이미 합쳐진 ID의 모든 관찰까지 포함합니다. 종류 불일치 자체는 이 단계에서 차단 사유가 아닙니다.
7. 후보를 **두 대상씩 다시 검토**합니다.
   앞 요청의 결론·레이블은 알려주지 않고, A/B 확대 crop과 두 대상의 위치가 표시된 짧은 시간순 장면을 줍니다.
   `same`, `different`, `uncertain` 중 `same`인 쌍만 합칩니다. 다른 샷/시점의 모습도 가능한 경우 각각 2개까지 보여줍니다.
   이 응답에서 최종 종류·레이블, 구체적 외형 근거와 시간적 근거를 받습니다.
   근거 프레임 키는 모델에게 긴 문자열을 재작성시키지 않고 실제 A/B 관찰이 표시된 입력 기록에서 저장합니다.
   병합마다 누적된 전체 관찰의 동시 등장 제약을 다시 확인합니다.
   어둡거나 가려진 이미지와, 큰 생물·작은 생물이 겹친 crop을 잘못 합치는 경우를 줄이기 위한 단계입니다.
8. 확정한 별칭을 다음 요청에 반영합니다. 겹치는 샷의 ID가 다음 문맥의 연결점이 됩니다.
   마지막에 모든 관찰과 출력 영상에 최종 ID·레이블을 적용합니다.

**후보 누락 보완:** Qwen이 처음부터 연결 후보를 제안하지 않는 경우도 있으므로, 이후 DINOv2 특징이 있는
동물·물체 관찰에서 인접 1~2개 샷의 후보를 추가 검색합니다. 다른 샷마다 유사도가 높은 서로 다른 ID 2개까지
선정하며, ID 쌍의 점수는 해당 관찰 쌍들의 정규화 특징 내적 중 최댓값입니다.
0.25는 후보 검색의 느슨한 하한일 뿐 동일 개체 판정 임계치가 아닙니다.
추가 후보도 같은 A/B 확대 이미지 + 시간순 장면 검토에서 `same` 판정을 받아야 병합됩니다.
추가 후보의 특징 유사도가 0.7 미만이면, 시간적 문맥 없이 확대 이미지의 **주 대상 외형만** 다시 비교합니다.
이 판정도 `same`이어야 병합합니다. 큰 생물과 작은 생물이 겹친 crop의 다른 대상을 골라
잘못 연결하는 것을 줄이기 위한 조건이며, `different` 또는 `ambiguous`이면 보류합니다.
외형 재검토는 `temporal_retrieval/ID__ID_appearance.json`에 저장합니다.
동시 등장 제약은 이 단계에서도 누적 관찰 전체에 적용합니다.
처음부터 인물로 오분류되어 DINO 특징이 없는 관찰은 이 추가 검색 대상에서 빠지지만,
앞 단계의 Qwen 후보 제안에서는 다른 종류 ID와도 연결할 수 있습니다.

이 단계는 검출 박스를 추가하거나 위치를 바꾸지 않습니다. 따라서 검출 자체가 빠진 프레임이나
잘못된 박스는 남을 수 있습니다. 샷 순서는 보조 근거이며, 같은 역할/줄거리만으로 병합하지 않도록 지시합니다.
검증 코드는 ID·동시 등장·근거 프레임의 구조적 일관성을 확인하지만, 모델이 기술한 외형 근거의 사실성까지 보장하지 않습니다.
3개 샷 문맥 밖의 장거리 연결은 겹치는 문맥에서 이미 이어진 ID를 통해서만 전파됩니다.
오랫동안 등장하지 않다가 다시 나타난 개체는 별도 ID로 남을 수 있습니다.
또한 샷 내부에서 이미 잘못 섞인 초기 ID를 개별 관찰 단위로 분리하는 기능은 포함하지 않습니다.
새 ON/OFF 비교에는 Qwen 호출뿐 아니라 ON의 샷별 초기 ID 분리도 포함되므로 순수한 모델 단독 효과 실험은 아닙니다.

근거는 `with_qwen/temporal_review/`에 저장합니다.

- `review.json`: 모든 시간순 검토 및 병합 내역.
- `shot_XXXX_to_shot_YYYY/input.json`: 프롬프트, 시간순 프레임/ID, 병합 금지 쌍.
- 같은 폴더의 `storyboard.jpg`, 개별 JPG: Qwen에 전달한 장면의 확인용 이미지.
- 같은 폴더의 `reference_crops.jpg`: 같은 요청에 함께 전달한 ID별 확대 이미지.
- 같은 폴더의 `result.json`: 장면 요약과 병합 근거.
- 같은 폴더의 `ID__ID.jpg`, `ID__ID_input.json`, `ID__ID_result.json`: 두 대상 재검토 입력과 판정.
- `observations.json`의 `pre_temporal_label`, `pre_temporal_entity_kind`: 수정 전 분류 보존.
- `temporal_retrieval/candidates.json`: DINO 추가 후보와 유사도.
- `temporal_retrieval/review.json`, `ID__ID_*`: 추가 후보의 Qwen 판정 및 실제 적용 병합.

`result.json`의 그룹은 제안이고, 실제 적용된 병합은 `review.json`의 `joins`입니다.
재시도 후에도 응답이 잘못된 구간은 `status: rejected`로 기록하고 기존 ID를 유지합니다.
두 대상 재검토도 같은 모델의 자동 판단이므로 정답 보장은 아닙니다.

공통 매칭에는 01의 `associate_frame()`과 `reconcile_tracklets()`를 사용합니다.
01의 별도 고정 FPS 샘플링 및 BoT-SORT 전체 추적 실행을 그대로 재현한 것은 아니며,
이번 비교에서는 03의 adaptive 샘플만 사용하는 04의 공통 연결 방식을 유지합니다.

### 1. 선택된 프레임과 원본 시각 읽기

`summary.json`의 영상 순서를 따릅니다. 각 `sampled.mp4`를 디코딩하여
`samples.json`의 선택 시각 및 유지 구간과 일치하는지 확인합니다.
영상이 VFR이라는 점을 반영하며, 일정 FPS로 다시 샘플링하지 않습니다.

복사해 온 입력 폴더 안의 상대 경로를 사용합니다.
`samples.json`에 기록된 과거 `source` 절대 경로에 있는 원본 영상을 다시 분석하지 않습니다.

### 2. YOLO 검출과 Qwen 보완

01의 YOLO11s로 검출 후보를 얻습니다. 기본 신뢰도 기준은 인물 0.60, 물체 0.70입니다.

이후 Qwen3-VL-4B-Instruct가 선택 이미지와 YOLO 후보를 함께 보고 최종 대상을 정합니다.
일반 검출기의 클래스에 없는 용 같은 판타지 생물도 보완하고, 오검출로 판단한 후보를 제거할 수 있습니다.
한 요청에 기본 4개 이미지를 전달하지만 각각의 검출 결과는 별도로 반환합니다.

범위는 보이는 인물, 동물/판타지 생물, 주요 독립 물체입니다.
배경·하늘·지형·자막·로고 및 인물의 신체 일부/옷은 별도 개체로 만들지 않습니다.
이야기 지식으로 이름이나 의도를 추정하지 않고, 관찰한 외형으로 한국어 레이블을 작성합니다.

Qwen 박스가 같은 종류의 YOLO 후보와 IoU 0.55 이상으로 대응하면 YOLO의 박스 좌표를 사용합니다.
그 밖의 추가 검출은 Qwen 좌표를 사용합니다.
Qwen 박스에는 YOLO confidence를 임의로 만들어 넣지 않으며 `detection_confidence`를 null로 기록합니다.

### 3. 외형 특징과 동일 개체 연결

01의 인코더와 연결 함수를 재사용합니다.

- 인물: YuNet/SFace의 얼굴 증거와 OSNet-AIN의 몸 외형 특징.
- 물체·동물: DINOv2-small의 crop 특징.
- 온라인 연결: 외형 유사도와 경쟁 후보 간 차이를 사용한 일대일 배정.
- 같은 샷 내부: 카메라 움직임을 반영한 연속성 정보를 보조적으로 사용.
- 샷 사이 초기 연결: 위치의 연속성을 적용하지 않고 외형으로 연결.
- 후처리: 강한 외형 증거가 있는 tracklet을 통합하고, Qwen이 여러 대표 crop을 비교해 추가 병합을 제안.

Qwen crop 검토는 샷 내부의 중복을 정리합니다. `gallery` 모드에서만 그 뒤 갤러리로 샷 사이도 비교합니다.
검토에 사용한 이미지와 병합 근거는 `identity_galleries/`, `identity_review_progress.json`에 저장합니다.
검출 단계는 01의 NF4 4bit 설정을 사용합니다. 갤러리 검토는 16GB 이상 GPU에서
동일 Qwen 모델의 BF16/FP16 가중치를 사용하며, 그보다 작은 GPU에서는 NF4를 사용합니다.

**같은 종류라는 이유만으로 같은 개체로 묶지 않습니다.**
동일 프레임에 함께 검출된 두 개체는 같은 ID로 합칠 수 없습니다.
crop 검토에는 동시 등장 금지 및 종류 호환성 제약을 적용합니다.
시간순 장면 검토는 동시 등장 금지를 유지하면서 시각 근거에 따른 종류 수정을 허용합니다.

모든 판단은 자동 추정입니다. 의심스러운 검출과 매핑 후보는 메타데이터에 남습니다.
동일 개체의 ID가 여러 개로 나뉘거나 서로 닮은 개체가 잘못 연결될 가능성이 있으므로,
대표 이미지와 시각화로 결과를 확인할 수 있도록 구성했습니다.

### 4. 대표 이미지와 레이블

각 최종 ID의 관찰 중 crop 크기, 선명도, 밝기, 불확실성 등을 고려해 대표 이미지를 고릅니다.
기본적으로 대표 이미지의 레이블을 사용하며, 시간순 검토에서 수정한 경우 최종 레이블을 사용합니다.
수정 전 관찰 레이블은 관찰 기록에 보존합니다.
최대 6개 샷의 추가 모습을 `view_*.jpg`로 저장합니다.
레이블이 비슷해도 서로 다른 entity ID라면 별도 개체입니다.

### 5. 전체 bounding box 영상

`adaptive_entities_full.mp4`는 모든 입력 샷을 입력 순서대로 연결한 **adaptive 영상 한 화면**입니다.
원본과 나란히 보이는 comparison 영상의 오른쪽 화면에 해당합니다.

- 선택된 이미지 위에 최종 전역 entity ID, 한국어 레이블, 개체별 고정 색상의 박스를 표시합니다.
- 이미지와 박스를 다음 선택 시점까지 함께 유지합니다.
- 샷마다 첫 이미지의 검출 결과로 갱신하므로 이전 샷의 박스를 이어서 그리지 않습니다.
- VFR로 선택 프레임 수와 재생 길이를 유지합니다.
- 상단 시각은 화면에 유지되고 있는 샘플의 원본 시각과 연결 영상 내 시각입니다.
- 비교용 결과와 마찬가지로 무음입니다.

유지 구간의 박스는 그 구간에 표시되는 대표 이미지의 박스입니다.
생략된 원본 프레임에서 물체가 계속 같은 위치에 있었다는 뜻은 아닙니다.

## 출력 파일

```text
outputs/sintel_entities/
  adaptive_entities_full.mp4   전체 adaptive 영상 + entity 박스
  entity_gallery.jpg          모든 entity의 대표 이미지와 레이블
  entities.json               최종 개체 목록
  entities/
    person_0001/
      representative.jpg      개체 대표 crop
      view_01.jpg ...         다른 샷에서 보이는 모습
      entity.json             레이블·등장 기록·병합된 ID
    object_0001/ ...           물체와 동물도 object 계열 ID 사용; kind로 구분
  observations.json           관찰별 bbox·검출 근거·전역 ID·원본 시각
  frame_annotations.json      샘플별 박스 목록; 빈 프레임도 포함
  samples.json                04에서 사용한 모든 샘플과 연결 시간
  timeline.json               샷 순서·길이·입력 해시
  identity_review.json        Qwen의 병합 근거와 ID 별칭
  identity_state.json         최종 ID 배정 상태
  identity_before_review.json Qwen 검토 전 ID 배정
  video_validation.json      출력 시각·길이 확인 결과
  run_config.json             설정과 모델 revision
  status.json                 진행 상태 또는 완료 요약
  frames/                     입력 영상에서 디코딩한 선택 이미지
  crops/                      검출된 모든 crop
  grounding/                  프레임별 Qwen 검출 결과
  features/                   외형 특징, 재실행 캐시
  traces/                     모델 요청·응답 기록
  review_traces/              갤러리 검토 요청·응답 기록
  identity_galleries/         샷 내부 및 샷 사이 비교에 사용한 이미지
  request_cache/              모델 응답 재사용 캐시
```

위 구조는 각 모드 폴더 내부에 해당합니다. `without_qwen`에는 Qwen 요청·검토 갤러리가 없고,
`identity_review.json`의 `qwen_enabled`가 false입니다. `grounding/`에는 YOLO 후보를 그대로 기록합니다.

`bbox`는 `[x1, y1, x2, y2]`, 범위 0~1000의 정규화 좌표입니다.
픽셀 좌표는 x에 영상 폭/1000, y에 영상 높이/1000을 곱해 얻습니다.

- `source_frame`, `source_time_sec`: 각 샷 안의 좌표.
- `original_frame`, `original_time_sec`: 전체 원본 영상 기준 좌표.
- `full_frame`, `full_time_sec`: 입력 샷들을 연결한 출력 영상 기준 좌표.
- `hold_frames`: 대표 이미지를 유지하는 원본 프레임 수.
- `display_intervals`: 출력에 해당 개체의 박스가 표시되는 구간. 끝은 포함하지 않습니다.

전체 샷이 원본 순서대로 빠짐없이 입력된 Sintel 결과에서는 원본과 연결 영상의 시간축이 일치합니다.
임의의 일부 샷만 입력하거나 순서를 바꾸면 두 시간축은 달라지며, 각각의 값을 따로 저장합니다.
