# Video Analysis Experiments

이 저장소는 영상 분석 실험을 단계별로 정리합니다. **최신 버전은 `04_entity_extraction`**입니다. 입력 영상은 저장소에 포함하지 않습니다. 직접 보유하거나 사용할 권한이 있는 영상을 별도로 준비해 주세요.

| 버전 | 경로 | 기능 |
| --- | --- | --- |
| 00 | `00_prototype/` | Qwen3-VL 이벤트 분석 프로토타입 |
| 01 | `01_low_freq_event/` | 저빈도 이벤트 감지 |
| 02 | `02_shot_boundary_detection/` | 샷 경계 검출 |
| 03 | `03_adaptive_sampling/` | 움직임 기반 프레임 샘플링 |
| **04 최신** | `04_entity_extraction/` | 선택 프레임 기반 Entity 추출·동일성 연결·장면 요약 |

## 04 빠른 실행

04의 기본 분석은 **sampled 모드**입니다. 03이 만든 입력 폴더의 선택 이미지와 `samples.json`만 분석합니다. 먼저 영상 파일을 준비해 02 샷 분할과 03 적응형 샘플링을 실행한 다음, 04에 그 결과 폴더를 전달합니다.

```bash
cd 03_adaptive_sampling
./setup.sh
./run.sh --help
./run.sh /absolute/path/to/video.mp4 --output outputs/my_video

cd ../04_entity_extraction
./run.sh --input ../03_adaptive_sampling/outputs/my_video \
  --output outputs/my_video_sampled --analysis-frames sampled
./visualize.sh outputs/my_video_sampled/entities
```

각 단계의 인자와 모델 준비 방법은 해당 폴더의 README를 참고하세요. 04의 상세 실행·재개·출력 설명은 [`04_entity_extraction/README.md`](04_entity_extraction/README.md)에 있습니다. 모델 가중치와 Python 환경은 첫 실행 전에 준비해야 하며, 첫 실행은 다운로드 시간이 추가됩니다.

## 저장소에 포함하지 않는 파일

영상 샘플과 다운로드한 영상은 Git에 포함하지 않습니다. `.venv/`, 모델 가중치, 캐시, `outputs/`의 분석 결과도 로컬에만 둡니다. 영상 경로를 요구하는 명령에는 각자 준비한 절대 경로를 전달하세요.
