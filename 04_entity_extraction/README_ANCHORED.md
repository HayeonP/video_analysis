# 04 · Entity extraction

현재 기본 실행은 **검출·중복 박스 융합 → 샷 사이 외형 연결 → Qwen 레이블·배정 → 최종 카탈로그 관계 검토**입니다.
Qwen 실제 생성은 **작업당 최대 8회**이며 실패·재시도·중단 후 재개도 같은 예산에 포함합니다.
입력 영상에서 대상 목록을 추론하며, 영화 이름·특정 샷 번호·정해진 인물 수를 매칭 규칙으로 사용하지 않습니다.
이번 수정은 실제 분석·테스트를 추가 실행하지 않은 상태입니다. 앞서 보고한 79개 검사는 이전 코드 기준이며, 이번 수정의 정확도를 증명하지 않습니다.

### 이번에 확인한 실패

이전 anchored 실행에서는 Qwen이 허용 개수보다 많은 참조 트랙을 반환해 전체 목록 응답의 JSON 검증이 실패했습니다.
그 결과 외형 연결을 활용하지 못하고 **56개 항목이 모두 미확정 ID**로 남았습니다.
이번에는 외형 연결 결과를 먼저 만들고, Qwen 응답 실패 시에도 보존하도록 변경합니다.
동일 대상의 중복 박스도 전역 ID를 만들기 전에 융합합니다. 새 추론 결과로 정확도가 개선되었는지는 아직 확인하지 않았습니다.
[실패 원인·저장된 근거·이번 수정 범위](ANCHORED_FAILURE_REPORT.md)에 자세히 정리했습니다.

## 실행

```bash
cd ~/HayeonP/video_analysis/04_entity_extraction
./run.sh --input inputs/your_video
./visualize.sh
```

VS Code **Ports**에서 5004를 전달하고 `http://localhost:5004`를 엽니다.
뷰어는 **분석 과정 이해** 탭의 논문 발표용 PPT 형식 방법 슬라이드로 시작합니다.
여기서 현재 4단계 구조와 모델·호출 예산을,
**영상 · Entity 결과** 탭에서 저장된 결과를 봅니다.
같은 포트의 기존 프로세스는 시각화 실행 시 종료합니다.

### 방법 슬라이드 보기

- 현재 방법을 **13장 슬라이드**로 설명하며, 도식은 확대해도 선명한 **벡터 SVG**로 구성했습니다.
- 상단 **방법 슬라이드 ↗** 또는 `/slides`에서 슬라이드만 크게 엽니다. 기본 탭의 `/guide`도 같은 내용을 표시합니다.
- `←` / `→` 키로 이동하고, `N` 키로 발표자 노트를 표시합니다. 탭 안에서는 슬라이드를 한 번 클릭한 뒤 키를 누르세요.
- 상단 **PDF 다운로드** 또는 `/slides.pdf`에서 [발표용 PDF](presentation/output/pdf/METHOD_SLIDES.pdf)를 내려받습니다. 브라우저 인쇄의 **PDF로 저장**도 사용할 수 있습니다.
- 상단 **HTML 다운로드**로 [METHOD_SLIDES.html](METHOD_SLIDES.html)을 저장해 로컬 브라우저에서도 엽니다. 슬라이드는 단일 HTML 파일입니다.
- 이전 상세 텍스트 안내는 `/guide-text`, 이전 버전 PPT 기록은 `/slides-legacy`에서 볼 수 있습니다.

### 입력과 실행 기본값

- 입력: **03 단계의 출력 폴더** (summary.json·samples.json·adaptive 조각 영상). 임의 MP4 폴더를 직접 받는 형식은 아닙니다.
- `--input` 생략 시: `inputs` 아래에서 03 출력 폴더가 정확히 하나일 때 자동 선택합니다. 여러 개이면 직접 지정합니다.
- `--output` 생략 시: `outputs/<입력폴더이름>_entities`에 저장합니다.
- `./visualize.sh`는 `outputs/latest_run.json`에 기록된 최근 완료 결과를 엽니다. 완료 기록이 없으면 처리 설명을 표시합니다.
- 모델: YOLO11x, YuNet/SFace, OSNet-AIN, DINOv2-small, Qwen3-VL-4B-Instruct

전체 비교 영상을 만들려면 입력 샷의 원본 FPS와 해상도가 같아야 합니다.

```bash
./run.sh --input inputs/your_video --mode with_qwen
./run.sh --input inputs/your_video --mode without_qwen --output outputs/your_video_off
./run.sh --input /path/to/adaptive --output outputs/your_result
./run.sh --input inputs/your_video --qwen-max-calls 4 --output outputs/your_video_budget4
./visualize.sh outputs/your_result  # 지정한 결과 보기
```

## 설정으로 바꾸는 항목

기본값은 `config.default.json`에서 읽습니다. 다른 영상에 맞게 설정을 복사해 수정할 수 있습니다.

```bash
cp config.default.json config.my_video.json
./run.sh --input inputs/your_video --config config.my_video.json
```

| 설정 구역 | 조정하는 내용 |
| --- | --- |
| `detector` | 검출 모델·입력 크기·신뢰도·타일 크기와 수·확대할 클래스·작은 대상의 화면 대비 높이 |
| `matching` | 장면·참고 이미지 수, 배치·최종 검토 트랙 수, 후보 수, 동일성 외형 임계값 |
| `tracking` | 샷 내부 연결의 최대 샘플·시간 간격, 할당 점수·경쟁 후보 간격 |
| `fusion` | 같은 프레임과 같은 샷에서 중복 검출을 융합하는 위치·외형·반복 관측 조건 |
| `visual_association` | 샷 사이 외형 연결의 얼굴·몸·DINO 기준, 후보 수, 연결 묶음의 일관성 조건 |
| `filtering` | 단일 관측의 약한 근거 표시와 제외 여부 |
| `appearance` | Qwen OFF의 외형 연결 점수와 경쟁 후보 간격 |
| `class_kinds` | 검출 클래스 이름을 person / animal / object로 분류하는 표 |

수치 기준은 일반적인 초기 휴리스틱이며, 여러 영상에서 정확도가 보정된 확률값은 아닙니다.
설정은 실행 기록에 저장하며, 입력이나 설정을 바꾸면 새 `--output`을 사용합니다.
영상별 ID나 시간 구간 예외는 사용하지 않습니다.
커스텀 검출 모델은 실제 클래스명을 `detector.tile_categories`와 `class_kinds`에 지정하세요.
`class_kinds`는 기본 COCO 분류표를 덮어쓰며 지정하지 않은 클래스는 `object`입니다.
GPU 선택은 `ENTITY_CUDA_DEVICE` 또는 `CUDA_VISIBLE_DEVICES`로 바꿀 수 있습니다.

## 컨셉: 외형으로 연결해 둔 목록을 Qwen이 검토한다

먼저 중복 박스를 줄이고, 얼굴·몸·대상의 시각적 특징으로 샷 사이의 동일 대상 후보를 연결합니다.
이 결과가 기본 목록입니다. Qwen은 장면과 참고 이미지를 보고 짧은 이름을 붙이고, 배정과 목록 사이의 관계를 검토합니다.
Qwen이 잘못된 형식의 답을 내도 기본 연결을 버리지 않습니다.

참고 트랙은 실제 외형 연결에서 얻은 항목으로 제한합니다. Qwen이 없는 트랙을 만들거나,
종류·이름이 같다는 이유만으로 모두 같은 대상으로 합칠 수 없습니다.
추가 통합에는 외형의 긍정적 근거가 필요하며, 같은 프레임의 서로 다른 대상은 분리합니다.
근거가 부족한 단독 트랙은 **동일성 미확정**입니다. 이것은 실제로 새 개체가 나타났다는 뜻이 아닙니다.

## 처리 단계

1. **YOLO11x + 작은 사람 타일 검출**: 전체 화면 1280 검출 후 필요한 프레임에 최대 6개 타일을 적용합니다.
   기본 확대 대상은 사람이며, 타일 추가 박스의 높이는 화면 높이의 9% 이하입니다. 전체 화면 검출에는 이 높이 제한이 없습니다.
   `detector.tile_categories`로 확대할 클래스 목록 또는 `all`을 선택할 수 있습니다.
   얼굴·몸·범용 DINO 특징을 뽑고, 외형과 박스 위치·속도로 **임시 샷 내부 트랙**을 만듭니다.
   먼저 여러 프레임에서 반복되는 중복 트랙을 융합합니다. 이어 프레임별 중복 관측을 정리하고 깨끗한 트랙을 다시 구성해 전역 매칭으로 보냅니다.
   겹쳤다는 이유만으로 합치지 않으며, 서로 다른 대상을 안거나 붙잡은 경우를 구분하기 위해 크기·위치·외형 근거를 함께 확인합니다.
   한 번만 관측되고 신뢰도가 0.65 미만인 트랙은 약한 근거로 표시하고 `weak_tracklets.json`에 기록합니다.
   **기본값은 보존**입니다. `filtering.drop_weak_singletons`를 `true`로 바꾸면 제외합니다.
2. **샷 사이 외형 연결, Qwen 0회**: 품질 조건을 통과한 얼굴, 얼굴과 몸·DINO의 결합 근거,
   강한 몸·DINO 근거로 연결 후보를 만듭니다. 사람이 아닌 대상은 DINO 외형을 사용합니다.
   트랙당 기본 최대 5개 후보를 강한 근거부터 연결하며, 묶음 전체의 동시 등장·얼굴 충돌과 대표 모습의 일관성을 확인합니다.
   결과는 `V…` 기본 ID와 실제 참고 트랙 목록입니다. 단독 트랙은 미확정으로 남깁니다.
3. **Qwen 전체 이해 1회 + 배정 최대 5회**: 이미 존재하는 기본 ID에 짧은 이름·종류·외형 설명을 붙입니다.
   참고 트랙의 소속은 외형 연결에서 가져오며 Qwen이 새로 작성하지 않습니다.
   이후 시간순 배치에서 트랙을 기준 대상 / 미확정 / 명백한 오검출로 판단합니다.
   외형 후보와 장면을 함께 제공하고, 잘못된 YOLO 종류는 명시적으로 교정합니다.
4. **최종 카탈로그 관계 검토 1회**: 미해결 배정과 함께, 분리된 ID가 동일 대상인지 검토합니다.
   통합 제안은 실제 참고 관측의 외형 근거와 동시 등장 제약을 통과해야 적용합니다.
   모든 ID 쌍을 반복 호출하거나 앞 단계로 되돌아가는 루프는 없습니다.
   응답 실패·검토 누락은 `anchored/quality.json`과 결과 상태에 경고로 남기며, 검토가 성공한 것처럼 표시하지 않습니다.

긴 입력에서 요청 크기가 계속 커지지 않도록 기본 장면 프레임 수는 최대 40개,
배치별 제공 트랙은 최대 32개로 제한합니다. 한도를 넘겨 검토하지 못한 트랙에는 Qwen 확인을 붙이지 않고 기존 외형 추정을 보존합니다.
이 수량은 `matching` 설정으로 조정할 수 있습니다. **8회 예산으로 임의 길이 영상의 모든 대상을 검토한다는 보장은 없습니다.**

독립 agent가 자율적으로 계획하는 구조는 아닙니다. Python이 고정된 처리 순서를 제어하고,
같은 Qwen 모델을 목록 설명·트랙 배정·마지막 관계 검토 역할로 호출합니다.

## Qwen 예산과 캐시

- 기본 계획: 전체 목록 설명 1회 + 최대 5개 배치 + 마지막 관계 검토 1회 = **계획상 최대 7회**. 빈 묶음은 생략합니다.
- **실제 생성의 절대 상한은 8회**. 요청 실패·재시도·GPU 메모리 오류도 1회로 계산합니다.
- 최종 검토용 1회는 예약합니다. 예산 부족이나 응답 실패 시 외형 추정을 보존하고 검토 미완료를 기록합니다.
- `with_qwen/qwen_budget.json`에 실제 횟수와 단계별 호출 이력을 기록합니다.
- 같은 output으로 재실행하면 남은 예산과 요청 캐시를 이어 씁니다. 캐시 적중은 새 생성에 포함하지 않습니다.
- 새 분석 예산을 시작하려면 새 `--output`을 지정하거나 `python3 clear_cache.py --apply`로 초기화합니다.
- 특징 캐시는 검출 fingerprint·박스 파일·특징 설정과 연결되어, 다른 박스의 이전 특징을 재사용하지 않습니다.

ON/OFF는 동일 입력과 YOLO 후보를 사용합니다. OFF는 Qwen 응답·레이블·매칭 결과를 사용하지 않습니다.
두 모드의 ID 문자열은 서로 독립적입니다.

## 산출물

```text
outputs/your_video_entities/
  timing.json
  qwen_comparison_full.mp4
  with_qwen/
    entities.json, entities/*/representative.jpg
    entity_shot_presence.csv
    adaptive_entities_full.mp4
    detector_summary.json
    tracklets.json, weak_tracklets.json
    observation_fusion.json, track_fusion.json
    qwen_budget.json
    duplicate_observations.json
    run_config.json  # 사용한 전체 설정
    anchored/coverage.json  # Qwen에 제시한/제시하지 못한 관측 범위
    anchored/visual_association.json  # 외형 연결·거절 근거
    anchored/quality.json  # 응답 실패·검토 범위·남은 불확실성
    anchored/  # 기준 목록·배치별 입력/응답·최종 검토·배정 감사 기록
  without_qwen/
```

전체 영상의 원본 시각과 adaptive 유지 구간은 보존합니다. 박스는 실제 선택 프레임의 관측이며,
생략된 원본 프레임의 존재 여부까지 증명하지는 않습니다. 결과 영상은 무음입니다.
설정이나 입력이 달라지면 새 output을 사용하세요. 분석 완료 후 뷰어를 재시작하면 결과가 갱신됩니다.

## 이전 방식

```bash
./run.sh --legacy --qwen-identity sequential --output outputs/legacy_run
```

[이전 사용법](README_LEGACY.md), [v2 문맥 매칭 기록](CONTEXT_FIX_REPORT.md),
[이전 PPT](ENTITY_EXTRACTION_SLIDES.html)는 이전 실행과 설계 기록입니다.
[이전 개선 기록](EFFICIENT_UPGRADE_REPORT.md)은 과거 측정치이며,
[현재 실패 분석과 수정 범위](ANCHORED_FAILURE_REPORT.md)는 별도로 정리했습니다.
현재 모델은 판타지 생물 분류, 작은 흐린 대상, 겹친 박스에서 오류가 남을 수 있습니다.
검출 수나 최종 ID 수 자체는 정확도가 아닙니다.
