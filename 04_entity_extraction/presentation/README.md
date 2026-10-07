# Entity extraction HTML 검토 슬라이드

결과 파일: [ENTITY_EXTRACTION_SLIDES.html](../ENTITY_EXTRACTION_SLIDES.html)

## 열기

HTML 파일을 브라우저에서 연다. 원격 VS Code에서는 파일을 로컬로 다운로드해 열어도
슬라이드, 폰트, 대표 이미지, 샷 로그, 전체 등장 행렬을 볼 수 있다. 외부 CDN이나 서버 요청은 없다.
결과 영상·JSON·CSV를 여는 상대 링크는 원래 프로젝트 폴더 구조에서 열 때 유효하다.

## 조작

- `←` / `→`, `Page Up` / `Page Down`, `Space`: 슬라이드 이동.
- `Home` / `End`: 처음 / 마지막 슬라이드.
- `O`: 목차. `F`: 전체화면. `Esc`: 목차·메모 닫기.
- 슬라이드 8: 샷을 선택해 실제 정방향 누적 로그 보기.
- 슬라이드 15: 45개 entity의 전체 등장 행렬 검색 및 대표 이미지 선택.
- 검토 메모: 슬라이드별 메모를 해당 브라우저에 저장하고 전체 JSON 내려받기.
- 인쇄 / PDF: 브라우저의 인쇄 기능 사용. 스크롤 표와 로그는 화면에 보이는 부분만 인쇄된다.

브라우저 저장소를 사용할 수 없는 경우 메모는 현재 창에서만 유지된다. JSON으로 내려받으면 된다.
메모는 분석 결과나 entity ID를 변경하지 않는다.

## 구성

23장. 현재 처리 과정 → 실제 결과와 한계 → 타깃 재탐색 제안 → 샷 요약·entity 기억 기반 agent 제안.
주황색으로 표시한 제안은 미구현 상태다. 이 자료를 만들면서 분석 파이프라인을 바꾸거나 재실행하지 않았다.
렌더링·브라우저 자동화 테스트는 실행하지 않았다.

## 원본

- `slides_template.html`: 슬라이드 내용, CSS, JavaScript.
- `build_slides.py`: 기록된 실행의 JSON·이미지·폰트를 단일 HTML에 포함하는 생성기.

```bash
python 04_entity_extraction/presentation/build_slides.py
```

설명과 정적 수치는 2026-10-06 `sintel_sequential` 실행의 스냅샷이다.
다른 실행 결과로 자료를 갱신하려면 본문의 수치와 사례 설명도 함께 갱신해야 한다.
