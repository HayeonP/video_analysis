# 인물별 영상 요약 (4B)

## 프롬프트 수정

- `prompts/person.txt`: 인물별 행동 설명 지시. 이 파일을 주로 수정하세요.
- `prompts/scene.txt`: 배경·구도·카메라 설명 지시.
- 공통 JSON/언어 규칙: `low_freq_event/prompts.py`의 `SYSTEM`.

대상 ID, 추적 관찰 구간, 전체 프레임 bbox와 JSON schema는 코드에서 자동으로 덧붙입니다. `person.txt`에는 행동을 관찰하고 서술하는 방식만 적으면 됩니다. 출력 필드를 바꾸려면 `low_freq_event/segment_events.py`의 스키마도 바꿔야 합니다.

## 실행

```bash
./events_10fps.sh --output outputs/prompt_v1
./visualize.sh outputs/prompt_v1
```

프롬프트를 수정했으면 **새 결과 폴더**를 지정하세요.

```bash
./events_10fps.sh --output outputs/prompt_v2
./visualize.sh outputs/prompt_v2
```

분석이 중단된 경우 프롬프트를 그대로 두고 동일 명령을 실행하면 재개합니다. 변경된 프롬프트로 기존 결과를 재개하는 것은 차단합니다. 실행 시 사용한 프롬프트는 결과 폴더의 `prompts_used.json`에 저장합니다. 실행 중 파일을 수정해도 이미 시작한 분석에는 반영되지 않습니다.

빠른 첫 윈도 확인:

```bash
./events_10fps.sh --output outputs/prompt_test --max-windows 1
./visualize.sh outputs/prompt_test
```

설정만 확인 (모델 실행 없음):

```bash
./events_10fps.sh --plan
```

## 고정된 분석 조건

Qwen3-VL-4B-Instruct, NF4 4bit. 전체 화면 10fps·5초, 오버랩 및 윈도 병합 없음. 현재 영상의 VLM 입력은 320×192입니다. 인물별로 같은 전체 영상을 보고 각각 요청하며 배경·구도는 별도 요청합니다. 크롭하지 않습니다. 시간 범위는 요약 대상 윈도이며 개별 행동의 시작·종료 시각이 아닙니다.

승인된 추적 결과 `outputs/260929_pretty_girl_person_tracking_v4`를 기본 입력으로 재사용합니다. **이 폴더를 삭제하지 마세요.** 다른 10fps 추적 결과는 첫 인자로 전달할 수 있습니다.

## 폴더

- `events_10fps.sh`, `visualize.sh`: 사용자 실행 진입점
- `prompts/`: 직접 수정할 프롬프트
- `runtime/`, `low_freq_event/`: 실행·설치에 필요한 내부 코드
- `models/`, `.venv/`: 추적 모델·실행 환경
- 입력 영상은 저장소 밖에 준비하고 실행 인자로 경로를 지정합니다.
- `outputs/`: 승인된 추적 결과와 새 분석 결과
- `tests/`: 코드 검증
- `docs/memo.md`: 보존한 사용자 메모

내부 CLI가 필요한 경우 `python3 runtime/run.py --help`, 환경 재설치는 `bash runtime/setup.sh`를 사용합니다.
