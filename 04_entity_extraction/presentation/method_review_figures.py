"""Vector-only review/output figures for slides 9–13 of the method deck.

All people, identities, matrix entries and frames below are conceptual examples,
not rendered analysis results. Importing this module performs no analysis.
"""
from method_diagrams import *


def _slide(title, subtitle, phase, kicker, body, takeaway, notes):
    return dict(title=title, subtitle=subtitle, phase=phase, kicker=kicker,
                body=body, takeaway=takeaway, notes=notes)


def _catalog_figure():
    out = card(42, 30, 380, 256, '기존 Entity 목록', BLUE)
    out += frame(65, 99, 150, 147, TEAL, label='V A')
    out += frame(245, 99, 150, 147, AMBER, subject='creature', label='V B')
    out += card(42, 325, 380, 189, '선택된 전체 장면', BLUE)
    out += frame(65, 382, 150, 106, TEAL, pose='scarf', boxed=False)
    out += frame(245, 382, 150, 106, AMBER, subject='creature', boxed=False)
    out += arrow(425, 160, 553, 214, BLUE, bend=500)
    out += arrow(425, 421, 553, 295, BLUE, bend=500)
    out += rect(555, 167, 280, 178, PALE_PURPLE, PURPLE, 18)
    out += text(695, 222, 'Qwen', 32, PURPLE, 700, 'middle')
    out += text(695, 267, '전체 목록 검토', 24, INK, 500, 'middle')
    out += text(695, 307, '장면 + 대표 모습', 21, MUTED, 400, 'middle')
    out += arrow(838, 255, 985, 255, PURPLE, dash=True)
    out += text(910, 231, '제안', 21, PURPLE, 500, 'middle')
    out += card(990, 86, 404, 350, '목록의 의미와 관계', PURPLE)
    for y, label, value in ((180, '이름', '짧은 레이블'),
                            (265, '종류', '인물 · 동물 · 물체'),
                            (350, '관계', '같음 · 다름 · 미확정')):
        out += text(1017, y, label, 23, PURPLE, 700)
        out += text(1100, y, value, 22, INK)
        if y < 350:
            out += line(1016, y+34, 1366, y+34)
    out += text(1192, 493, '기존 ID에 근거를 덧붙입니다', 22, MUTED, 400, 'middle')
    return out


def _batch_figure():
    out = card(42, 34, 374, 136, '현재 배치의 관측', BLUE)
    out += frame(64, 85, 95, 65, TEAL, label=None)
    out += frame(173, 85, 95, 65, TEAL, pose='back', label=None)
    out += text(340, 123, 'T …', 22, TEAL, 600, 'middle')
    out += card(42, 208, 374, 136, '기존 ID의 참고 모습', BLUE)
    out += person(96, 267, .40, TEAL)
    out += person(164, 267, .40, TEAL, 'scarf')
    out += creature(273, 295, .52, AMBER)
    out += text(350, 302, 'V …', 22, BLUE, 600, 'middle')
    out += card(42, 382, 374, 136, '앞선 샷의 요약', BLUE)
    out += line(69, 452, 267, 452, MUTED, 3)
    out += line(69, 479, 336, 479, LINE, 3)
    for y in (104, 276, 450):
        out += arrow(420, y, 552, 270, BLUE, bend=490)
    out += rect(555, 154, 292, 232, PALE_PURPLE, PURPLE, 18)
    out += text(701, 205, 'Qwen · 시간순 배정', 25, PURPLE, 700, 'middle')
    out += text(701, 258, ['기존 ID로 연결', '미확정 유지', '명백한 오검출 제외'],
                22, INK, 400, 'middle')
    out += arrow(850, 270, 977, 218, PURPLE, dash=True, bend=920)
    out += card(982, 93, 408, 186, '배정 제안 → 코드 검사', BLUE)
    out += chip(1007, 171, 'Track', TEAL, PALE_TEAL, 105)
    out += arrow(1130, 190, 1210, 190, BLUE)
    out += chip(1230, 171, 'Entity', TEAL, PALE_TEAL, 128)
    out += text(1186, 249, '외형 근거와 충돌 확인', 21, MUTED, 400, 'middle')
    out += arrow(1186, 282, 1186, 334, BLUE)
    out += rect(1023, 341, 326, 73, 'white', LINE)
    out += text(1186, 386, '이번 샷의 요약', 24, INK, 500, 'middle')
    out += arrow(1186, 418, 1186, 467, BLUE)
    out += rect(1084, 476, 204, 61, PALE_BLUE, PALE_BLUE)
    out += text(1186, 515, '다음 배치', 23, BLUE, 600, 'middle')
    return out


def _merge_figure():
    out = card(36, 115, 238, 272, '분리된 두 ID', BLUE)
    out += person(106, 197, .78, TEAL)
    out += person(205, 197, .78, TEAL, 'scarf')
    out += text(106, 351, 'ID A', 22, TEAL, 600, 'middle')
    out += text(205, 351, 'ID B', 22, TEAL, 600, 'middle')
    out += arrow(277, 250, 326, 250, BLUE)
    out += rect(336, 176, 210, 151, PALE_PURPLE, PURPLE)
    out += text(441, 225, 'Qwen', 27, PURPLE, 700, 'middle')
    out += text(441, 274, '같은 대상 제안', 23, INK, 500, 'middle')
    out += arrow(549, 250, 596, 218, PURPLE, dash=True)
    out += card(606, 117, 238, 181, '외형 근거', BLUE)
    out += vector(634, 198, BLUE, 10, 152, 44)
    out += text(725, 278, '얼굴 · 몸 · DINO', 21, MUTED, 400, 'middle')
    out += arrow(847, 214, 881, 214, BLUE)
    out += card(892, 117, 241, 181, '동시 등장 검사', BLUE)
    out += text(1012, 206, ['서로 다른 관측의', '충돌이 없는가?'], 21, INK, 400, 'middle')
    out += arrow(1137, 214, 1182, 214, BLUE)
    out += rect(1192, 96, 211, 230, PALE_TEAL, TEAL)
    out += text(1297, 136, '하나로 연결', 24, TEAL, 700, 'middle')
    out += person(1297, 160, .82, TEAL)
    out += line(725, 301, 725, 460, MUTED, 2)
    out += arrow(728, 460, 964, 460, MUTED)
    out += arrow(1012, 302, 1116, 406, MUTED, bend=1012)
    out += text(751, 430, '근거 부족 / 충돌', 22, MUTED)
    out += rect(974, 414, 428, 93, PAPER, LINE)
    out += text(1188, 452, 'ID A와 ID B를 유지', 24, INK, 600, 'middle')
    out += text(1188, 486, '미확정 관계로 남길 수 있습니다', 20, MUTED, 400, 'middle')
    return out


def _budget_figure():
    out = text(44, 45, '정해진 요청 계획', 25, INK, 700)
    out += rect(44, 85, 253, 151, PALE_PURPLE, PURPLE)
    out += text(170, 132, '전체 목록', 24, PURPLE, 600, 'middle')
    out += text(170, 200, '1', 43, INK, 700, 'middle')
    out += text(329, 173, '+', 35, MUTED, 400, 'middle')
    out += rect(364, 85, 526, 151, 'white', LINE)
    out += text(627, 132, '시간순 배치 · 최대 5', 24, PURPLE, 600, 'middle')
    for i in range(5):
        x = 393+i*95
        out += rect(x, 157, 77, 55, PALE_PURPLE, PALE_PURPLE, 10)
        out += text(x+38.5, 194, str(i+1), 24, PURPLE, 600, 'middle')
    out += text(923, 173, '+', 35, MUTED, 400, 'middle')
    out += rect(957, 85, 240, 151, PALE_PURPLE, PURPLE)
    out += text(1077, 132, '최종 관계', 24, PURPLE, 600, 'middle')
    out += text(1077, 200, '1', 43, INK, 700, 'middle')
    out += text(1295, 126, '계획 최대', 22, MUTED, 500, 'middle')
    out += text(1295, 197, '7회', 43, PURPLE, 700, 'middle')
    out += line(45, 274, 1395, 274)
    out += text(45, 331, '실제 생성의 절대 상한', 25, INK, 700)
    for i in range(8):
        x = 46+i*74
        out += rect(x, 364, 61, 61, PALE_BLUE, BLUE, 10)
        out += text(x+30.5, 404, str(i+1), 24, BLUE, 600, 'middle')
    out += text(338, 465, '실패 · 재시도도 포함', 23, BLUE, 600, 'middle')
    out += text(338, 508, '마지막 검토용 1회 예약', 21, MUTED, 400, 'middle')
    out += rect(744, 354, 229, 135, PAPER, LINE)
    out += text(858, 408, ['응답 오류', '예산 소진'], 23, INK, 500, 'middle')
    out += arrow(978, 421, 1083, 421, BLUE)
    out += rect(1093, 354, 301, 135, PALE_TEAL, TEAL)
    out += text(1243, 407, ['기본 외형 연결 보존', '검토 상태 기록'], 23, TEAL, 600, 'middle')
    return out


def _outputs_figure():
    out = card(35, 63, 427, 454, '대표 이미지 + Entity 목록', TEAL)
    out += frame(61, 131, 153, 143, TEAL, label='A')
    out += text(237, 183, 'Entity A', 25, TEAL, 700)
    out += text(237, 225, '짧은 이름', 22, MUTED)
    out += line(61, 305, 435, 305)
    out += frame(61, 335, 153, 143, AMBER, subject='creature', label='B')
    out += text(237, 387, 'Entity B', 25, AMBER, 700)
    out += text(237, 429, '대표 모습', 22, MUTED)

    out += card(507, 63, 427, 454, 'Entity × Shot 등장 행렬', BLUE)
    for i, label in enumerate(('S1', 'S2', 'S3')):
        out += text(660+i*94, 184, label, 24, MUTED, 600, 'middle')
    for row, (label, color, pale, values) in enumerate((
            ('A', TEAL, PALE_TEAL, (1, 0, 1)),
            ('B', AMBER, PALE_AMBER, (0, 1, 1)))):
        y = 213+row*107
        out += text(560, y+52, label, 26, color, 700)
        for col, value in enumerate(values):
            x = 617+col*94
            out += rect(x, y, 83, 81, pale if value else PAPER, LINE, 10)
            out += text(x+41.5, y+53, str(value), 31, color if value else MUTED,
                        700 if value else 400, 'middle')
    out += text(720, 457, '1: 관측됨    0: 미검출', 22, MUTED, 400, 'middle')

    out += card(979, 63, 427, 454, '전체 시간축의 박스 영상', BLUE)
    out += frame(1005, 135, 375, 213, TEAL, pose='scarf', pos=.40, label='A')
    out += creature(1296, 235, .68, AMBER)
    out += rect(1245, 200, 99, 59, 'none', AMBER, 3, 2.5)
    out += text(1350, 221, 'B', 20, AMBER, 700)
    out += text(1008, 392, '선택 프레임의 유지 구간', 22, MUTED)
    for x, width, color in ((1007, 93, TEAL), (1105, 159, BLUE), (1269, 109, AMBER)):
        out += rect(x, 414, width, 24, color, color, 4)
    out += arrow(1007, 459, 1377, 459, MUTED, 1.8)
    out += text(1192, 491, '원본 시간 정보 보존', 22, INK, 500, 'middle')
    return out


def review_slides():
    """Return the final five slides as metadata plus SVG body fragments."""
    return [
        _slide(
            'Qwen은 기존 목록의 의미를 검토합니다',
            '기존 ID와 선택 장면을 입력하는 개념도입니다. 실제 분석 결과를 표시한 그림은 아닙니다.',
            'review', '09 / CATALOG REVIEW', _catalog_figure(),
            '이름을 붙이는 일과 동일 개체를 연결하는 일을 구분합니다.',
            '<p>먼저 외형 연결로 구성한 기존 Entity ID가 있습니다. Qwen은 선택된 전체 장면과 '
            'ID별 대표 모습을 보고 짧은 한국어 이름, 종류, 안정적인 외형 설명을 제안합니다.</p>'
            '<p>이 단계에서도 두 기존 ID의 같음·다름·미확정 관계를 제안할 수 있습니다. '
            '관계 제안은 코드의 외형 근거와 실제 동시 등장 검사에 통과해야 적용됩니다.</p>'
            '<ul><li>기본 전체 검토: 샷당 장면 프레임 2개, 요청당 최대 40개, 기존 ID 최대 24개.</li>'
            '<li>한 ID의 대표 crop은 최대 2개이며 가능한 경우 서로 다른 샷의 모습을 사용합니다.</li>'
            '<li>새 박스나 존재하지 않는 ID를 생성하지 않습니다. 명확한 오검출 제외에는 두 개의 구체적 근거를 요구합니다.</li></ul>'),
        _slide(
            '시간순으로 관측의 소속을 검토합니다',
            '현재 관측 → 기존 ID 배정 → 다음 배치에 요약 전달을 나타낸 개념도입니다.',
            'review', '10 / SEQUENTIAL ASSIGNMENT', _batch_figure(),
            '외형 후보와 앞선 장면의 요약을 함께 주되, 실제 연결은 코드가 확인합니다.',
            '<p>Qwen은 현재 배치의 트랙 crop, 후보 Entity의 대표 모습, 선택 장면 프레임, '
            '기존 배정, 앞선 샷의 요약을 함께 받습니다. 각 트랙을 기존 ID에 연결하거나 '
            '미확정으로 두거나 명백한 오검출로 제안할 수 있습니다.</p>'
            '<p>기본 배치는 4샷 단위이지만 긴 입력에서는 전체 배치 수가 5개를 넘지 않도록 '
            '한 묶음의 샷 수가 늘어납니다. 배치당 최대 32개 트랙, 샷당 장면 프레임 4개를 '
            '제시하며 요청당 장면 수는 최대 40개입니다.</p>'
            '<ul><li>외형 후보는 기본 3개이며, 작은 목록에서는 전체 기준 ID를 보여줍니다. 기존 배정도 항상 후보 화면에 포함합니다.</li>'
            '<li>같은 Qwen 모델을 Python이 정해진 순서로 호출합니다. 여러 자율 agent가 계획하거나 협의하는 구조는 아닙니다.</li>'
            '<li>박스가 사라졌을 때 Qwen이 다시 검출해 새 박스를 만드는 절차는 현재 없습니다.</li></ul>'),
        _slide(
            '마지막에는 분리된 ID의 관계를 다시 봅니다',
            '같은 인물이 두 ID로 남았을 때의 병합 제안과 적용 조건을 보여주는 개념도입니다.',
            'review', '11 / PROPOSAL AND GATES', _merge_figure(),
            'Qwen의 “같다”는 제안이며, 외형과 실제 관측의 충돌 검사를 통과해야 연결됩니다.',
            '<p>최종 검토는 활성 Entity 목록 중 선택된 ID를 대상으로 한 번 수행합니다. '
            'Qwen은 전체 선택 장면, 대표 모습, 앞선 요약을 보고 두 ID가 같은 개체인지 제안합니다.</p>'
            '<p>추가 병합에는 얼굴·몸·DINO의 긍정적 외형 근거가 필요합니다. 실제로 같은 프레임에 '
            '나타난 서로 다른 대상은 분리합니다. 다만 중복 검출로 확인된 두 박스는 별개의 대상이라는 '
            '충돌 근거로 취급하지 않습니다.</p>'
            '<ul><li>이름·종류·서사상 역할이 같다는 이유만으로 합치지 않습니다.</li>'
            '<li>기본 최종 검토 대상은 최대 24개 ID입니다. 모든 미확정 트랙을 다시 개별 배정하는 단계는 아닙니다.</li>'
            '<li>역방향 순회나 모든 ID 쌍을 반복 호출하는 절차는 없습니다. 얼굴이 가려지거나 외형이 크게 바뀌면 연결이 보류될 수 있습니다.</li></ul>'),
        _slide(
            '계획은 최대 7개 요청, 실제 생성은 최대 8회',
            '현재 설정의 호출 예산과 실패 시 처리 경로를 나타낸 개념도입니다.',
            'review', '12 / BOUNDED COMPUTATION', _budget_figure(),
            '응답 실패로 기본 외형 연결을 버리지 않고, 미완료 검토를 표시합니다.',
            '<p>기본 요청 계획은 전체 목록 검토 1회, 시간순 배치 최대 5회, 마지막 목록 관계 검토 1회입니다. '
            '관측이 없는 묶음은 생략하므로 실제 요청 수는 더 적을 수 있습니다.</p>'
            '<p>새 Qwen 생성의 절대 상한은 실패·재시도·GPU 메모리 오류를 포함해 작업당 8회입니다. '
            '마지막 검토를 위해 1회를 예약하고, 중단 후 재개할 때도 기존 사용량을 이어 씁니다.</p>'
            '<ul><li>각 요청 위치에서 재시도는 최대 한 번이며 공통 8회 상한을 넘지 못합니다.</li>'
            '<li>저장된 요청 캐시를 재사용하면 새 생성 횟수는 늘지 않습니다.</li>'
            '<li>잘못된 응답 항목은 개별적으로 제외합니다. 응답 실패·예산 소진·검토 누락은 기본 연결을 보존하고 품질 기록에 남깁니다.</li>'
            '<li>8회 예산으로 임의 길이 영상의 모든 대상을 검토한다는 의미는 아닙니다.</li></ul>'),
        _slide(
            '목록·등장 행렬·영상으로 결과를 검토합니다',
            '카드, 0·1 행렬, 박스 영상은 산출물 형식을 설명하는 예시이며 실제 결과 수치가 아닙니다.',
            'export', '13 / INSPECTABLE OUTPUTS', _outputs_figure(),
            'Entity의 대표 모습과 등장 근거를 원본 시간축에서 함께 확인합니다.',
            '<p>출력에는 Entity별 대표 이미지와 짧은 레이블, 샷별 등장 행렬, adaptive 전체 영상의 '
            '박스 시각화가 포함됩니다. 비교 모드에서는 ON/OFF 결과와 비교 영상도 저장합니다.</p>'
            '<p>표의 1은 선택 프레임에서 해당 Entity가 관측되었다는 뜻입니다. '
            '0은 이 분석에서 검출되지 않았다는 뜻이며, 생략된 원본 프레임까지 대상이 없었다고 증명하지는 않습니다.</p>'
            '<ul><li>원본 시간 정보와 adaptive 프레임의 유지 구간을 보존합니다. 결과 영상은 무음입니다.</li>'
            '<li>미확정 연결, 검토하지 못한 관측, 호출 예산과 적용·거절 근거도 별도로 기록합니다.</li>'
            '<li>ON/OFF는 같은 입력과 검출 후보를 사용하지만 외형 연결 알고리즘도 달라 Qwen만 분리한 엄밀한 대조 실험은 아닙니다.</li>'
            '<li>현재 수정의 실제 정확도는 새 추론 결과로 아직 확인하지 않았습니다. 개체 수가 줄었다는 사실만으로 정확도가 좋아졌다고 판단하지 않습니다.</li></ul>'),
    ]
