# 03_adaptive_sampling

개념부터 이해하려면 **[Adaptive Sampling 설명서](ADAPTIVE_SAMPLING_GUIDE.md)**를 읽으세요.
목적과 핵심 아이디어를 먼저 설명한 뒤, 계산식·선택 규칙·시간 보존·설정 조절 방법을 순서대로 다룹니다.

샷 단위 영상을 **optical flow + visual change의 누적 변화량**으로 샘플링합니다.
목표는 VLM에 전달할 프레임 수를 줄이면서 동작과 내용 변화를 따라가는 것입니다.
CPU에서 저해상도로 모든 프레임을 분석하고, 선택된 프레임만 원본 해상도로 출력합니다.

## 제공된 Sintel 결과

`outputs/sintel_adaptive/`에 17개 영상의 처리를 완료했습니다.

- 1,253 → **279프레임**, **77.7% 감소**.
- 합계 **52.208333초** 유지. 17개 모두 영상/컨테이너 길이 차이 0초.
- 원본 오디오 17개 보존, 최대 이미지 유지 시간 1초.
- 전체 원본 타임라인에서 샷 간 간격/중복 0프레임.
- `batch_validation.json`에 이번 실행의 검사 결과를 모았습니다.

액션 장면인 `shot_0009`는 50 → 18프레임, 타이틀 장면인 `shot_0015`는 141 → 9프레임입니다.
프레임 감소율은 실제 VLM 연산량을 측정한 값은 아닙니다.

## 실행

```bash
cd ~/HayeonP/video_analysis/03_adaptive_sampling
./setup.sh
./run.sh --comparison
```

기본 입력은 `inputs/sintel_shots.json`에 나열된 17개 영상입니다.
실제 영상 경로는 이 프로젝트의 `outputs/sintel_trainler_full/shots/`입니다.
기본 결과 폴더는 `outputs/sintel_adaptive/`이며 이번 실행 결과가 이미 있습니다. 기존 결과는 덮어쓰지 않으므로,
다시 실행할 때는 새 결과 폴더를 지정하세요.

```bash
./run.sh --input-list inputs/sintel_shots.json \
  --output outputs/sintel_adaptive_v2 --comparison
```

`--comparison`은 샷별 비교 영상과 **전체 비교 영상 `comparison_full.mp4`**를 함께 만듭니다.
왼쪽은 원본, 오른쪽은 adaptive sampling 결과이며, 입력 목록 순서로 연결합니다.
기존 비교 영상을 재인코딩 없이 연결하므로 화질을 추가로 손실하지 않습니다. 비교 영상은 무음입니다.
전체 비교 영상도 원본 FPS로 재생하는 검토용 영상이며 VLM 입력용으로 사용하지 않습니다.
화면 위에 표시된 `t`와 프레임 번호는 샷 내부 좌표라 샷이 바뀔 때 다시 시작합니다.
`comparison_full.json`에는 각 샷의 전체 비교 영상 내 시작/끝 시간과 원본 좌표를 기록합니다.

이미 샷별 비교 영상이 만들어진 결과 폴더에서는 샘플링을 다시 하지 않고 연결할 수 있습니다.

```bash
.venv/bin/python join_comparisons.py outputs/sintel_adaptive
```

이미 전체 비교 영상이 있으면 덮어쓰지 않습니다. 연결 대상의 해상도/FPS/코덱/time_base는 같아야 하며,
연결 후 모든 프레임 시각과 총 재생시간을 확인합니다.

기존 02 단계의 `shots.json`도 직접 입력할 수 있습니다. 이 경우 전체 원본의 시작 프레임을 자동으로 가져옵니다.

```bash
./run.sh --input-list outputs/sintel_trainler_full/shots.json \
  --output outputs/from_shot_manifest
```

영상 경로들을 바로 나열하는 방법도 있습니다. 이 방법의 시간 기준은 각각의 부분영상 시작점입니다.

```bash
./run.sh /path/to/shot_a.mp4 /path/to/shot_b.mp4 --output outputs/custom
```

필요 환경: Python 3.11 이상, `ffmpeg`, `ffprobe`, Linux의 `flock`.
`setup.sh`가 프로젝트 전용 `.venv`에 NumPy, OpenCV, PyAV를 설치합니다. GPU는 사용하지 않습니다.

## 샘플 선택 방법

1. 매 프레임을 폭 256px로 축소합니다. 원본을 건너뛰며 분석하지 않으므로 샘플 사이의 변화도 점수 계산에 반영합니다.
2. Farneback optical flow를 계산합니다.
3. RANSAC으로 화면 전반에 걸친 카메라의 이동·회전·확대/축소를 추정합니다. 충분한 공간적 지지와 일치점이 있을 때만 보정합니다.
4. 카메라 움직임을 뺀 **residual flow**, 카메라 보정 후 **색상/밝기 변화**, 적은 비중의 **카메라 움직임**을 합칩니다.
5. 점수를 짧게 평활화하여 누적하고, 임계치에 도달하면 프레임을 선택합니다. 선택 후 누적치는 초기화합니다.

기본 점수는 다음과 같습니다. flow 크기는 분석 이미지 대각선 길이로 정규화하며,
visual change는 RGB 절대 차이를 0~1로 정규화합니다.

```text
step = 2.0 × residual_flow + 1.0 × visual_change + 0.15 × camera_motion
increment = 0.5 × step + 0.5 × EMA(step, 0.12초)
누적 increment >= 0.08 → 새 샘플
```

작은 피사체의 움직임이 화면 평균에 묻히지 않도록 4×4 영역의 통계도 반영합니다.
flow는 85백분위수, visual change는 영역 평균을 사용합니다.

추가 규칙:

- 기본적으로 샘플 간격은 최대 약 1초입니다. 정적인 구간도 주기적으로 선택합니다.
- 일반적인 선택 빈도 상한은 10fps입니다. 원본 프레임 격자에 맞추므로 24fps 입력에서 최소 일반 간격은 3프레임입니다.
- 마지막 선택 이미지와 현재 이미지의 차이가 커지면 누적치가 부족해도 선택합니다. 카메라 이동으로 새로 드러난 내용도 반영됩니다.
- 인접 프레임의 변화가 매우 크면 일시적 사건으로 간주해 즉시 선택합니다. 이 규칙과 첫/마지막 프레임 보존은 일반 빈도 상한의 예외입니다.
- 첫 프레임과 마지막 프레임은 항상 선택합니다.

이 점수는 의미 이해 모델이 아닙니다. 작은 표정 변화나 짧은 행동을 모두 보존한다고 보장할 수는 없습니다.
검토용 비교 영상을 보고 분석 목적에 맞춰 샘플 밀도를 조정하세요.

### 구현의 정확한 수식

현재 구현은 `sampling.py`의 `motion_features()`와 `Selector.push()`에 있습니다.
아래에서 `t`는 프레임 번호, `F`는 원본 FPS, `dt = 1/F`, `D`는 분석 이미지의 대각선 길이입니다.
점수와 가중치는 학습한 값이 아닌 초기 휴리스틱 설정입니다.

**1. 카메라 보정과 residual flow**

`u_t(p)`는 이전 프레임의 픽셀 `p`가 현재 프레임에서 이동한 벡터입니다.
전역 변환 `T_t(p)`로 설명되는 이동 `g_t(p) = T_t(p) - p`를 뺍니다.

```text
residual_t(p) = u_t(p) - g_t(p)
M_t = max(0, Q85(||residual_t||) / D - 0.0002)
C_t = P90(||g_t||) / D
```

`Q85`는 유효 영역 전체의 85백분위수와 4×4 각 영역의 85백분위수 중 최댓값입니다.
flow 통계에서는 질감이 있는 픽셀을 사용하고 테두리 4px를 제외합니다.
`C_t`는 화면 전체 카메라 이동 크기의 90백분위수입니다.

카메라 변환은 이동·회전·균일 확대/축소를 포함하는 partial affine입니다.
6px 간격의 질감 있는 점들을 사용하고, RANSAC 오차 허용치는 분석 이미지 기준 1.2px입니다.
최소 12개 일치점, 일치 비율 55% 이상, 화면 폭 45%/높이 35% 이상의 공간적 분포,
프레임 사이 배율 0.90~1.10 및 제한된 이동량을 만족할 때만 보정합니다.
실패하면 항등 변환을 사용하므로 카메라 성분은 0, residual은 보정 전 flow가 됩니다.

**2. Visual change**

이전 이미지를 카메라 변환으로 정렬한 `I_aligned`와 현재 이미지 `I_t`를 비교합니다.

```text
e_t(p) = (|R_t-R_aligned| + |G_t-G_aligned| + |B_t-B_aligned|) / (3 × 255)
V_t = max(0, R(e_t) - 0.002)
```

`R`은 전체 유효 픽셀 평균과 4×4 영역 중 평균 차이가 큰 상위 3개 영역의 평균 중 큰 값입니다.
카메라 정렬 후 겹치는 영역만 사용하고 테두리 3px는 제외합니다.
밝기/색상 변화와 페이드도 이 항에 반영됩니다. SSIM이나 VLM embedding을 사용하지 않습니다.

**3. 결합, 평활화, 누적**

```text
s_t = 2.0 × M_t + 1.0 × V_t + 0.15 × C_t
alpha = 1 - exp(-dt / 0.12)
r_t = (1-alpha) × r_(t-1) + alpha × (s_t / dt)
q_t = 0.5 × s_t + 0.5 × r_t × dt
A_t = A_(t-1) + q_t
```

`r_t`는 초당 변화율의 지수이동평균이며 초기값은 0입니다.
현재 변화와 평활화한 변화를 절반씩 섞어서 순간 반응과 안정성을 함께 반영합니다.
샘플이 선택되면 `A`는 0으로 초기화합니다. 초과한 누적치는 이월하지 않으며 `r`은 유지합니다.
새 샷에서는 모든 상태를 초기화합니다. 별도의 상향/하향 임계치를 사용하는 hysteresis는 없습니다.

**4. 실제 선택 조건: 아래 순서로 적용**

`k`는 마지막으로 선택한 프레임, `elapsed = (t-k)/F`입니다.

| 우선순위 | 조건 | 선택 사유 |
|---|---|---|
| 시작 | 첫 프레임 | `first` |
| 1 | 보정 전 인접 이미지 차이 `R(abs(I_t-I_(t-1))/255) >= 0.25` | `transient` |
| 2 | `elapsed >= 1.0초` | `max_interval` |
| 3 | `elapsed >= 0.1초`이고 마지막 선택 이미지와의 차이 `>= 0.18` | `visual_guard` |
| 4 | `elapsed >= 0.1초`이고 `A_t >= 0.08` | `change_budget` |
| 종료 | 마지막 프레임이 아직 선택되지 않았으면 추가 | `last` |

위 이미지 차이 식의 RGB 차이는 채널 평균을 사용합니다.
`transient`와 `visual_guard`에서는 카메라 정렬과 작은 차이 제거를 적용하지 않습니다.
따라서 카메라가 움직여 새 내용이 보이면 카메라 보정을 했어도 자주 선택할 수 있습니다.
`transient`는 일반 최소 간격 0.1초를 무시하므로 연속 프레임도 선택할 수 있습니다.

예를 들어 24fps에서 누적 증가량 `q_t`가 안정적으로 0.01이면 약 8프레임마다 임계치 0.08에 도달하므로
약 3fps가 됩니다. 0.04라면 2프레임 만에 임계치에 도달하지만, 일반 최소 간격 때문에 3프레임마다 선택하여
약 8fps가 됩니다. 거의 변화가 없으면 24프레임마다 선택합니다. 첫/마지막 프레임과 추가 규칙 때문에
실제 `샘플 수 / 영상 길이`는 이 근사값과 다를 수 있습니다.

### 밀도 조절

```bash
# 더 촘촘하게: 누적 임계치를 낮추고, 최소 빈도를 높임
./run.sh --change-budget 0.04 --min-fps 2 --max-fps 12 \
  --output outputs/sintel_dense --comparison

# 더 성기게
./run.sh --change-budget 0.12 --min-fps 0.5 --max-fps 8 \
  --output outputs/sintel_sparse --comparison
```

| 옵션 | 기본값 | 역할 |
|---|---:|---|
| `--change-budget` | 0.08 | 낮을수록 누적 변화에 빨리 반응 |
| `--min-fps` | 1 | 최대 유지 간격의 역수. 실제 간격은 원본 프레임 단위로 올림됨 |
| `--max-fps` | 10 | 일반 선택 빈도 상한. 순간 변화/끝점은 예외 |
| `--visual-guard` | 0.18 | 마지막 선택 프레임 대비 큰 시각적 변화 감지 |
| `--transient-guard` | 0.25 | 인접 프레임의 큰 순간 변화 감지 |
| `--analysis-width` | 256 | 분석 해상도. 높일수록 작은 변화에 유리하고 분석 비용 증가 |
| `--comparison` | 꺼짐 | 샷별 나란히 보기 영상과 전체 연결 영상 생성 |

## 출력과 시간 보존

```text
outputs/sintel_adaptive/
  summary.json / summary.csv       전체 프레임 감소량과 영상별 요약
  input_list.json                 실제 사용한 입력 목록
  comparison_full.mp4            전체 샷 비교 영상을 입력 순서대로 연결 (--comparison)
  comparison_full.json           연결 구간별 시간 좌표와 전체 길이 확인 결과
  shot_0001/
    sampled.mp4                   VFR 샘플링 영상, 원본 해상도/영상 길이 유지
    frames/frame_00000000.jpg ...  선택한 원본 프레임의 JPEG
    samples.json                  샘플 시각, 전체 원본 시각, 유지 구간, 설정
    motion.csv                    모든 입력 프레임의 점수와 선택 사유
    validation.json               출력 PTS/프레임 수/재생시간 확인 결과
    selected_frames.jpg           선택 프레임을 모은 이미지
    comparison.mp4                --comparison 지정 시 생성, 무음
```

**시간축을 자르거나 영상을 빠르게 재생하지 않습니다.** 예를 들어 원본의 0.0초, 0.4초, 1.1초를 선택했다면,
출력은 첫 이미지를 `[0.0, 0.4)`, 다음 이미지를 `[0.4, 1.1)` 동안 유지합니다.
중간 프레임의 시각적 정보는 샘플링으로 생략되지만, 그 시간 구간은 출력에 남습니다.

`sampled.mp4`는 선택된 프레임만 인코딩한 **VFR(가변 프레임 간격) MP4**입니다.
각 프레임 PTS는 원본 샘플 시점이고, 유지 시간을 명시하여 마지막까지 원본 영상 길이를 보존합니다.
입력의 첫 오디오 트랙은 재인코딩 없이 복사합니다. `comparison.mp4`는 사람이 비교할 수 있도록 원본 FPS로 만든 별도의 검토 영상입니다.

출력 직후 영상을 실제 디코딩하여 선택 프레임 수와 모든 PTS가 일치하는지,
영상 스트림 길이가 정확히 같은지 확인합니다. 컨테이너 길이 차이는 최대 2ms까지 허용하며 결과에 기록합니다.
입력은 현재 **CFR(고정 FPS), 짝수 해상도 영상**을 지원합니다. VFR 입력은 임의 변환하지 않고 오류로 알립니다.

### VLM에 넘길 때

`frames/`의 이미지들을 **`samples.json`의 `samples` 순서대로** 전달하는 것이 가장 명확합니다.
영상 전처리기가 MP4를 다시 일정 FPS로 추출하면 유지 중인 이미지가 중복 입력될 수 있으므로,
선택 프레임 수 감소가 실제 VLM 연산 감소로 이어지는지 전처리 방식도 맞춰야 합니다.
`comparison.mp4`는 VLM 입력으로 사용하지 마세요.

샘플별 좌표는 모두 0부터 시작합니다.

- `source_frame`, `source_time_sec`: 해당 부분영상 안에서의 프레임 번호와 시각.
- `original_frame`, `original_time_sec`: 전체 Sintel 원본 안에서의 프레임 번호와 시각.
- `hold_until_frame`: 이 샘플을 유지하는 구간의 끝 프레임 번호. 끝은 포함하지 않음.
- `hold_duration_sec`: 샘플의 표시 유지 시간.
- `image`: 해당 `samples.json` 폴더 기준 이미지 경로.

정확한 원본 좌표가 필요하면 정수 `original_frame`과 유리수 문자열 `original_fps`를 사용하세요.
초 단위 필드는 편의상 실수로 저장합니다.

## 입력 목록 형식

경로는 **목록 JSON 파일이 있는 폴더 기준**입니다. 간단한 문자열 배열도 허용합니다.

```json
{
  "videos": [
    {
      "id": "shot_0001",
      "path": "../outputs/sintel_trainler_full/shots/shot_0001.mp4",
      "original_start_frame": 0,
      "original_fps": "24",
      "original_video": "/path/to/full_original.mp4"
    }
  ]
}
```

`original_start_frame` 등을 생략하면 해당 부분영상 자체가 시간 기준이 됩니다.
전체 원본 기준 분석을 하려면 제공된 `inputs/sintel_shots.json` 또는 02 단계의 `shots.json`을 사용하세요.
샷과 전체 원본의 FPS는 같아야 합니다.

## 참고

- [OpenCV optical flow 문서](https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html)
- [OpenCV affine transform 추정 문서](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html)
