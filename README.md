# Video Analysis Experiments

| 단계 | 폴더 | 설명 |
| --- | --- | --- |
| 01 | `01_low_freq_event/` | 인물별 이벤트 분석·요약 |
| 02 | `02_shot_boundary_detection/` | 영상 샷(장면 전환) 분할 |
| 03 | `03_adaptive_sampling/` | 분석할 프레임 선택 |
| 04 | `04_entity_extraction/` | 선택 프레임에서 Entity 추출·연결 및 장면 요약 |

## Sintel 분석 및 시각화

04의 새 분석 방식으로 Sintel을 분석하고 결과를 확인합니다.

```bash
cd 04_entity_extraction
./analyze_new_sintel.sh
./visualize_new_sintel.sh
```

분석 결과는 `04_entity_extraction/outputs/`에 저장되며, 시각화 명령은 새 방식의 Sintel 결과를 엽니다.
