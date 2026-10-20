# 원본 시간축을 보존하는 Shot Boundary Detection

TransNetV2로 하드 컷·페이드·디졸브를 검출하고, **원본의 모든 프레임을 정확히 한 번씩 포함하는 연속 구간**으로 분할합니다. CPU에서 실행하며, PySceneDetect와 FFmpeg로 H.264/AAC MP4를 저장합니다.

## 전체 예고편 결과

예전 Sintel 샘플 실행 결과는 로컬 실험 기록이며 저장소에는 포함하지 않습니다. 실행할 때 본인이 준비한 입력 영상을 지정하세요.

예전에 사용한 `clean_shots` 방식은 전환 구간을 삭제했습니다. 현재 기본 방식은 `contiguous`이며 전환·검은 화면·짧은 조각을 포함한 원본 전체를 보존합니다. 이전 누락본과 검증 보고서는 `outputs/archive/`에 보관합니다.

## 실행

```bash
cd /path/to/3r_video/02_shot_boundary_detection

# 전체 예고편을 새 결과 폴더에 저장
./run.sh /absolute/path/to/video.mp4 --output outputs/full_new

# 다른 영상
./run.sh /path/to/video.mp4 --output outputs/my_video

# 직접 지정한 영상 처리
./run.sh /absolute/path/to/video.mp4 --output outputs/demo_new
```

기존 폴더는 덮어쓰지 않습니다. `run.sh`가 필요하면 환경을 설치하며, 첫 TransNetV2 실행에서 약 30 MB 가중치를 다운로드합니다.

## 경계와 전환 구간

1. TransNetV2 확률과 밝기를 이용해 장면 내부와 전환 구간을 찾습니다.
2. 두 장면 사이 전환 구간의 중간 프레임을 **하나의 분할 경계**로 정합니다. 한 프레임짜리 하드 컷 예측은 이전 장면 끝 프레임으로 배정합니다.
3. 전환 구간의 앞부분은 이전 샷, 뒷부분은 다음 샷에 포함합니다. 시작·끝의 검은 화면도 첫·마지막 샷에 포함합니다.
4. 짧은 잔여 조각은 주변 전환 구간에 합쳐 경계를 정합니다. 해당 프레임 자체는 저장 결과에 남습니다.
5. 저장 전 전체 경계가 `[0, 원본 프레임 수)`를 빈틈없이 덮는지 검사합니다. 저장 후 MP4를 디코딩해 각 파일의 프레임 수와 FPS를 확인하며, 불일치하면 실패 처리합니다.

따라서 각 샷의 시작·끝에 페이드 일부가 보일 수 있습니다. 원본 프레임을 보존하면서 전환 구간을 두 샷에 나눠 담은 것입니다. H.264/AAC 재인코딩을 하므로 압축 픽셀·오디오 데이터가 비트 단위로 동일하다는 의미는 아닙니다.

```bash
# 전환 검출 민감도 조정
./run.sh /absolute/path/to/video.mp4 \
  --threshold 0.4 --min-shot-seconds 0.25 --output outputs/sensitive

# PySceneDetect AdaptiveDetector와 비교
./run.sh /absolute/path/to/video.mp4 \
  --detector adaptive --threshold 3.0 --output outputs/adaptive_compare
```

기본값은 `--detector transnet --threshold 0.5 --min-shot-seconds 0.25`입니다. 더 짧은 영상이나 경계가 없는 영상은 전체를 한 파일로 저장합니다. 검출 정확도는 영상에 따라 달라질 수 있으나, 어떤 경계가 선택되더라도 프레임을 삭제하지 않습니다. 고정 프레임레이트(CFR) 입력을 기준으로 동작합니다.

## 원본 시간축으로 분석하기

`shots.json`/`shots.csv`의 프레임 번호는 **0부터 시작**, 구간은 **[start_frame 포함, end_frame_exclusive 제외)**입니다. `start_sec`, `end_sec`는 원본 영상 기준입니다.

```text
원본 프레임 번호 = 해당 샷의 start_frame + 샷 안의 프레임 번호
원본 시각(초)    = 원본 프레임 번호 / 원본 FPS
원본 시각(초)    = 해당 샷의 start_sec + 샷 안의 비디오 재생 시각(초)
```

분할 MP4의 재생 시각은 각각 0초부터 시작합니다. 정확한 분석에는 `start_frame`과 FPS를 기준으로 변환하세요. 컨테이너 전체 길이는 AAC 오디오 패딩 등으로 비디오 길이와 조금 다를 수 있으므로, 파일 속성의 전체 길이를 누적하는 대신 위 좌표를 사용하세요.

다음 조건을 만족하도록 저장합니다.

```text
첫 샷 start_frame == 0
이전 샷 end_frame_exclusive == 다음 샷 start_frame
마지막 샷 end_frame_exclusive == total_frames
exported_frame_count == total_frames
excluded_frame_count == 0
```

`transition_ranges`는 전환이 검출된 범위와 실제 `cut_frame`을 기록합니다. 이 범위의 프레임도 MP4에 포함됩니다. `excluded_ranges`는 빈 배열입니다.

## 검증

```bash
# 원본과 모든 샷을 실제 디코딩해 프레임 수·순서·원본 프레임 내용 대조
.venv/bin/python verify_timeline.py outputs/sintel_trainler_full

# 페이드·검은 화면·짧은 조각·하드 컷·오디오·실제 MP4 출력 회귀 테스트
.venv/bin/python -m unittest discover -s tests -v
```

`verify_timeline.py`는 원본 체크섬, 프레임 수, 누락·중복 구간, FPS, 원본의 대응 프레임과 재인코딩 결과의 축소 이미지 차이를 검사합니다. `timeline_audit.json`, `timeline_gaps.csv`를 저장하고 실패하면 종료 코드 1을 반환합니다. 누락이 없으면 CSV에는 헤더만 남습니다.

## 구성과 출처

- `gradual.py`: TransNetV2 추론 및 모든 프레임을 보존하는 경계 결정
- `split_shots.py`: 분할·시간축 메타데이터·디코딩 검증
- `download_demo.py`: 전체 예고편 다운로드 및 앞 30초 샘플 준비
- `setup.sh`: 독립 `.venv`, PySceneDetect/OpenCV headless, CPU용 PyTorch 설치
- `vendor/transnetv2/README.md`: 고정된 소스 버전, 변환 가중치 출처와 체크섬

30초 샘플은 체크섬과 FPS가 일치하는 전체 예고편을 추론 문맥으로 참고하고, 저장 범위는 샘플 안으로 제한합니다.

- [Sintel 공식 프로젝트](https://durian.blender.org/download/), [실제 다운로드 미러](https://media.w3.org/2010/05/sintel/trailer_hd.mp4)
- © Blender Foundation | durian.blender.org. [Creative Commons Attribution 3.0 안내](https://durian.blender.org/sharing/)
- [TransNetV2](https://github.com/soCzech/TransNetV2), [PySceneDetect](https://www.scenedetect.com/docs/latest/api.html)
