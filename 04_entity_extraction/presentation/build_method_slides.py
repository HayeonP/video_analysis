#!/usr/bin/env python3
"""Create the current method deck from offline vector figures; no ML is run."""
import base64
from html import escape
from pathlib import Path
from method_diagrams import *
from method_review_figures import review_slides

ROOT=Path(__file__).resolve().parents[1]

def item(title,subtitle,phase,kicker,body,takeaway,notes):
    return dict(title=title,subtitle=subtitle,phase=phase,kicker=kicker,body=body,takeaway=takeaway,notes=notes)

def opening_slides():
    slides=[]
    # 01 — One visual grammar and only four conceptual stages.
    b=''
    for x,n,title,color,fill in [(25,'01','관측 정리',TEAL,PALE_TEAL),(385,'02','외형 연결',BLUE,PALE_BLUE),(745,'03','문맥 검토',PURPLE,PALE_PURPLE),(1105,'04','결과 기록',INK,'#eef1f3')]:
        b+=rect(x,60,310,370,fill,fill)+chip(x+20,80,n,color,'white',56)+text(x+25,167,title,31,color,700)
    b+=frame(52,199,116,145,boxed=True)+frame(186,199,116,145,boxed=True,pose='scarf')
    b+=text(180,394,'박스 → 깨끗한 트랙',21,TEAL,600,'middle')
    for x,y,lab in [(427,235,'T1'),(427,335,'T4'),(639,285,'V1')]:
        b+=circle(x,y,27,'white',BLUE)+text(x,y+7,lab,20,BLUE,600,'middle')
    b+=arrow(455,239,606,277,BLUE)+arrow(455,329,606,294,BLUE)
    b+=text(540,394,'샷이 달라도 같은 대상',21,BLUE,600,'middle')
    b+=rect(785,207,230,118,'white','white')+text(900,246,'Qwen',31,PURPLE,700,'middle')+text(900,290,'이미지 + 장면 요약',20,PURPLE,400,'middle')
    b+=text(900,394,'이름 · 배정 · 관계',21,PURPLE,600,'middle')
    for i,col in enumerate([TEAL,AMBER,BLUE]):
        b+=circle(1153,228+i*52,11,col,col)+line(1184,222+i*52,1368,222+i*52,LINE,8)+line(1184,240+i*52,1302,240+i*52,LINE,4)
    b+=text(1260,394,'목록 · 등장 표 · 영상',21,INK,600,'middle')
    for x in [337,697,1057]:b+=arrow(x,247,x+36,247,MUTED)
    b+=text(30,501,'03의 선택 이미지에서 시작',20,MUTED)+text(1410,501,'원본 시각을 유지하며 저장',20,MUTED,400,'end')
    slides.append(item('여러 샷의 관측을, 같은 대상끼리 연결합니다','현재 기본 실행 · 도식은 개념 예시이며 분석 결과를 나타내지 않습니다.','all','01 / THE METHOD',b,'네 단계만 기억하세요: 관측을 정리하고 → 외형으로 연결하고 → 문맥으로 검토하고 → 기록합니다.',
        '<p>현재 run.sh의 Qwen ON 경로를 설명합니다. YOLO 검출·특징 추출, 중복 정리, 외형 기반 기본 목록, 제한된 Qwen 검토, 결과 저장 순서입니다.</p><p>그림의 인물·생물·ID·개수는 개념 예시입니다. 특정 영화나 정해진 등장인물 수를 규칙으로 사용하지 않습니다. 수정 후 실제 동일성 정확도는 아직 확인하지 않았습니다.</p>'))
    # 02 — Observation -> local track -> global entity.
    b=''
    for x,title,eng in [(30,'박스 하나','OBSERVATION'),(510,'한 샷의 연결','TRACK'),(990,'샷을 넘는 연결','ENTITY')]:
        b+=text(x,55,eng,16,MUTED,600)+text(x,99,title,30,INK,700)
    b+=frame(83,148,255,248,TEAL,label='관측 A')+text(210,452,'한 프레임에서 본 대상',23,TEAL,600,'middle')
    for i,x in enumerate([510,651,792]):
        b+=frame(x,195,116,167,TEAL,pos=.4+i*.1)+text(x+58,394,f't{i+1}',19,MUTED,400,'middle')
        if i<2:b+=arrow(x+117,276,x+135,276,TEAL)
    b+=chip(640,133,'T01',TEAL,PALE_TEAL,100)+text(700,452,'시간을 따라 이어진 관측',23,TEAL,600,'middle')
    b+=chip(1162,143,'V01',BLUE,PALE_BLUE,100)
    for y,lab,pose in [(214,'샷 A · T01','scarf'),(314,'샷 C · T08','front')]:
        b+=rect(1020,y,190,73,'white',LINE)+person(1051,y+8,.35,TEAL,pose)+text(1082,y+43,lab,19)
        b+=arrow(1213,y+36,1270,265,BLUE)
    b+=circle(1310,265,40,PALE_BLUE,BLUE)+person(1310,236,.38,TEAL)
    b+=text(1200,452,'여러 샷에 등장한 같은 대상',23,BLUE,600,'middle')
    b+=arrow(368,275,474,275,MUTED)+arrow(890,275,972,275,MUTED)
    slides.append(item('박스 → 트랙 → Entity: 세 가지 단위','같은 대상은 여러 박스와 여러 트랙으로 관측될 수 있습니다.','observe','02 / THREE SCALES',b,'검출 횟수와 실제 대상의 수를 구분합니다.',
        '<p>관측은 선택 프레임 하나의 검출 박스입니다. 트랙은 한 샷 안에서 이어진 관측 묶음이고, Entity는 샷을 넘나드는 동일성 추정입니다.</p><p>T…는 샷 내부 트랙, V…는 외형 연결로 만든 기본 ID이며 출력 시 E… 형식의 최종 ID로 정리됩니다. 번호는 의미나 인물 수를 미리 정하는 규칙이 아닙니다.</p>'))
    # 03 — Preserve temporal spans with fewer actual images.
    b=''
    for x,lab,pose in [(80,'A','front'),(550,'B','scarf'),(1070,'C','front')]:
        b+=frame(x,35,280,187,TEAL,pose=pose,label=lab)
    b+=text(30,293,'원본 시간',19,MUTED)+line(180,287,1400,287,MUTED,2)
    for i in range(45):b+=line(180+i*27.7,278,180+i*27.7,296,LINE,2)
    for x,lab,tim in [(180,'A','1.0 s'),(430,'B','1.4 s'),(1040,'C','2.4 s')]:
        b+=circle(x,287,7,TEAL,TEAL)+text(x,333,tim,19,MUTED,400,'middle')
    b+=arrow(220,232,180,268,TEAL)+arrow(690,232,430,268,TEAL)+arrow(1210,232,1040,268,TEAL)
    b+=text(30,404,'표시 구간',19,MUTED)
    for x,w,lab,fill in [(180,250,'A 유지',PALE_TEAL),(430,610,'B 유지',PALE_BLUE),(1040,360,'C 유지',PALE_PURPLE)]:
        b+=rect(x,363,w-3,72,fill,fill,r=5)+text(x+w/2,407,lab,24,INK,600,'middle')
    b+=text(790,504,'분석은 선택 이미지에만 · 재생 시간은 원래 길이로',24,TEAL,600,'middle')
    slides.append(item('시간은 유지하고, 선택된 이미지만 분석합니다','입력: 03의 adaptive 조각 영상 + 선택 시각과 유지 구간','observe','03 / ADAPTIVE INPUT',b,'04는 03의 샘플 시각을 이어받습니다. 생략된 원본 순간을 다시 분석하지 않습니다.',
        '<p>summary.json과 샷별 samples.json에서 source_frame, hold_until_frame, 원본 시각을 읽습니다. adaptive 영상의 디코딩 프레임과 메타데이터의 시각이 맞는지 확인합니다.</p><p>선택 이미지 A를 다음 선택 이미지 B까지 유지합니다. 표시 시간이 길어도 같은 이미지를 매 원본 프레임마다 다시 분석하지 않습니다. 그림의 시각·간격은 설명용입니다.</p>'))
    # 04 — Whole image and zoom tile join in the detector.
    b=frame(30,82,450,285,TEAL,boxed=False,subject='bird')
    b+=person(408,267,.27,TEAL)+rect(393,260,31,53,'none',TEAL,r=2,sw=2)
    b+=text(32,55,'전체 화면',23,TEAL,700)+text(38,404,'작은 사람은 확대 타일로 보강',22,INK,600)
    b+=rect(275,418,190,118,PALE_TEAL,TEAL,dash=True)+person(370,429,.58,TEAL)
    b+=arrow(421,306,380,410,TEAL,dash=True)
    b+=arrow(492,205,597,205,TEAL)+arrow(472,477,609,307,TEAL)
    b+=rect(610,157,230,181,PALE_TEAL,PALE_TEAL)+text(725,224,'YOLO11x',31,TEAL,700,'middle')+text(725,271,'위치 · 종류 · 신뢰도',20,TEAL,400,'middle')
    b+=arrow(852,248,921,248,TEAL)
    b+=frame(938,82,450,285,TEAL,subject='bird',boxed=True)
    b+=person(1316,267,.27,TEAL)+rect(1301,260,31,53,'none',TEAL,r=2,sw=2)
    b+=text(942,55,'실제 검출 박스',23,TEAL,700)
    b+=chip(1010,414,'전체 화면 + 필요한 확대 영역',TEAL,PALE_TEAL,354)
    slides.append(item('전체 화면을 보고, 작은 대상을 확대합니다','박스 위치는 YOLO에서 얻습니다. 현재 확대 검출의 기본 대상은 사람입니다.','observe','04 / DETECTION',b,'검출 단계의 목표는 “무엇이 어디에 보이는가?”입니다.',
        '<p>기본 모델은 YOLO11x, 입력 크기 1280, 사람·물체 신뢰도 기준 0.25입니다. 작은 사람이 관측된 샷이거나 해당 프레임에서 사람을 찾지 못하면 타일 검출을 시도합니다.</p><p>프레임당 최대 6개 타일을 사용하며, 타일에서 추가하는 대상의 높이는 전체 화면 높이의 9% 이하입니다. 전체 화면 검출에는 이 높이 제한이 없습니다. 검출할 확대 클래스는 설정으로 변경할 수 있습니다.</p><p>Qwen은 이 경로에서 새로운 박스를 추가하거나 좌표를 수정하지 않습니다. YOLO가 끝내 찾지 못한 대상의 누락은 이후 이름·관계 검토만으로 복구되지 않습니다.</p>'))
    # 05 — Feature branches and typed evidence.
    b=frame(35,127,245,310,TEAL,boxed=False)+text(157,481,'검출 영역을 잘라낸 이미지',19,MUTED,400,'middle')
    for y,name,model,color in [(64,'얼굴','YuNet → SFace',TEAL),(243,'몸 · 복장','OSNet-AIN x1.0',BLUE),(422,'전체 외형','DINOv2-small',PURPLE)]:
        b+=arrow(290,280,455,y+28,color,bend=365)
        b+=rect(473,y-19,384,108,'white',LINE)+text(498,y+17,name,22,color,700)+text(498,y+58,model,26,INK,600)
        b+=arrow(870,y+33,954,y+33,color)+vector(977,y+1,color,16,205,54)
        b+=text(1211,y+43,'유사도 비교',21,color,600)
    b+=text(490,550,'동물·물체에는 DINO 특징을 사용합니다.',20,MUTED)
    slides.append(item('외형을, 서로 비교할 수 있는 특징으로 바꿉니다','얼굴·몸·전체 모습이 각각 다른 단서를 제공합니다.','observe','05 / APPEARANCE FEATURES',b,'얼굴이 가려지면 근거가 줄어듭니다. 몸과 전체 외형도 함께 비교합니다.',
        '<p>YuNet으로 얼굴을 찾고, 크기·선명도 등 품질 조건을 만족하면 SFace 특징을 추출합니다. 사람 crop에서는 OSNet-AIN 몸 특징과 DINOv2-small 특징도 구합니다.</p><p>사람이 아닌 대상은 DINO 특징으로 비교합니다. 특징 벡터의 코사인 유사도를 사용하며, 점수는 동일 대상일 확률이 아닙니다. 여러 관측의 좋은 시점과 반복 지지를 함께 반영합니다.</p><p>DINO는 이 구조에서 박스 검출기나 이름 생성기가 아닙니다. 같은 이름이나 같은 검출 클래스 자체도 동일성 근거로 쓰지 않습니다.</p>'))
    # 06 — One-to-one tracks within a shot.
    b=''
    for i,x in enumerate([50,532,1014]):
        b+=text(x,47,f'선택 프레임 {i+1}',21,MUTED,600)
        b+=frame(x,76,350,252,TEAL,pos=.24+i*.12,boxed=True)
        b+=creature(x+268-i*15,182,.54,AMBER)+rect(x+222-i*15,153,91,52,'none',AMBER,r=3,sw=2)
        b+=circle(x+78+i*42,386,19,TEAL,TEAL)+circle(x+268-i*15,460,19,AMBER,AMBER)
    b+=arrow(148,386,612,386,TEAL,3)+arrow(675,386,1158,386,TEAL,3)
    b+=arrow(340,460,754,460,AMBER,3)+arrow(805,460,1230,460,AMBER,3)
    b+=text(1260,393,'T01',22,TEAL,700)+text(1300,467,'T02',22,AMBER,700)
    b+=chip(417,15,'외형 + 위치 + 이동',TEAL,PALE_TEAL,300)
    b+=text(720,541,'각 프레임의 대상은 하나의 트랙에 연결',22,INK,600,'middle')
    slides.append(item('한 샷 안에서는, 관측을 시간순으로 연결합니다','장면이 이어지는 동안 외형과 이동 경로가 자연스러운지 확인합니다.','observe','06 / LOCAL TRACKS',b,'샷 내부 트랙은 여러 장의 관측을 담는 작은 사진 묶음입니다.',
        '<p>외형, 박스 겹침, 중심 위치, 이전 이동량을 이용한 위치 예측, 실제 시간 간격을 함께 점수화합니다. Hungarian 일대일 할당으로 현재 관측을 이전 트랙에 연결합니다.</p><p>기본적으로 최대 3개 샘플 간격과 2초 이내의 관측을 연결 후보로 둡니다. 애매한 연결은 임시 트랙을 분리해 보존합니다. 이 단계만으로 샷을 넘는 동일성을 결정하지 않습니다.</p><p>그림의 아래 선은 관측 사이의 연결을 표현합니다. 검출하지 않은 중간 프레임에 새 박스를 보간해 만드는 것은 아닙니다.</p>'))
    # 07 — Duplicate individual vs genuinely nested subjects.
    b=rect(25,34,864,435,'white',LINE)+text(55,78,'같은 대상의 중복',27,TEAL,700)
    b+=frame(65,121,240,255,TEAL,boxed=True)+rect(150,168,95,198,'none',BLUE,r=3,sw=2.8,dash=True)
    b+=arrow(329,244,538,244,TEAL)+text(434,212,'반복 근거',22,TEAL,600,'middle')
    b+=text(434,290,['위치 · 크기','외형이 일치'],19,MUTED,400,'middle')
    b+=frame(581,121,240,255,TEAL,boxed=True)+check(823,135,TEAL)
    b+=text(446,430,'신뢰도가 높은 실제 박스 하나를 남김',22,TEAL,600,'middle')
    b+=rect(935,34,480,435,PALE_AMBER,PALE_AMBER)+text(965,78,'겹친 서로 다른 대상',27,AMBER,700)
    b+=creature(1167,232,1.85,BLUE)+creature(1193,257,.52,AMBER)
    b+=rect(1023,164,302,134,'none',BLUE,r=3,sw=2.8)+rect(1148,234,88,50,'none',AMBER,r=3,sw=2.8)
    b+=text(1175,365,'큰 대상 + 작은 대상',22,INK,600,'middle')+text(1175,415,'두 대상의 관측을 유지',23,AMBER,600,'middle')
    for i,lab in enumerate(['임시 트랙','반복 중복 융합','프레임 중복 정리','트랙 재구성']):
        x=55+i*350;b+=chip(x,504,lab,TEAL,PALE_TEAL,280)
        if i<3:b+=arrow(x+288,522,x+330,522,TEAL)
    slides.append(item('중복 박스는 줄이고, 별개의 대상은 유지합니다','같은 ID가 붙기를 기다리지 않고 중복 검출을 먼저 정리합니다.','observe','07 / OBSERVATION FUSION',b,'박스가 겹친다는 사실에 외형·크기·반복 관측 근거를 더해 판단합니다.',
        '<p>먼저 임시 샷 내부 트랙에서 여러 프레임의 중복 근거를 확인합니다. 기본적으로 지지 프레임이 2개 이상이고 공통 프레임의 절반 이상이어야 하며, 나머지 공통 관측의 위치·크기와 얼굴 충돌도 확인합니다.</p><p>다음으로 프레임별 중복을 정리하고 깨끗한 트랙을 다시 만듭니다. 같은 프레임의 매우 유사한 박스는 위치·면적·DINO 근거로 서로 다른 검출 클래스 간에도 정리할 수 있습니다.</p><p>융합 시 최고 신뢰도의 실제 관측을 유지합니다. 제거한 검출 ID·대안 클래스·연결 근거는 audit JSON에 남깁니다. 여러 대상을 품은 큰 박스로 합성하지 않습니다.</p>'))
    # 08 — Sparse visual graph; two solid associations and unresolved singleton.
    b=''
    for x,lab in [(40,'샷 A'),(372,'샷 B'),(704,'샷 C')]:b+=text(x+100,46,lab,22,MUTED,600,'middle')
    positions=[(40,99,'T01','person',TEAL,'scarf'),(372,99,'T08','person',TEAL,'front'),(704,99,'T14','person',TEAL,'back'),(40,322,'T02','bird',AMBER,'front'),(372,322,'T10','bird',AMBER,'front')]
    for x,y,lab,sub,col,pose in positions:
        b+=frame(x,y,194,154,col,sub,pose,boxed=False,label=lab)
    b+=arrow(243,177,359,177,BLUE,3)+arrow(573,177,691,177,BLUE,3)
    b+=text(292,142,'얼굴',18,BLUE,600,'middle')+text(632,142,'몸·외형',18,BLUE,600,'middle')
    b+=arrow(243,395,359,395,BLUE,3)+text(292,363,'외형',18,BLUE,600,'middle')
    b+=rect(704,322,194,154,PALE_AMBER,AMBER,dash=True)+text(801,390,'?',54,AMBER,400,'middle')+text(801,447,'미확정 관측',19,AMBER,600,'middle')
    b+=arrow(914,181,1073,181,BLUE)
    b+=f'<path d="M582 399 H644 V492 H1030 V376 H1073" fill="none" stroke="{BLUE}" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#arr-blue)"/>'
    b+=rect(1092,102,300,137,PALE_BLUE,PALE_BLUE)+text(1120,146,'V01',27,BLUE,700)+text(1120,192,'T01 + T08 + T14',21,INK,500)
    b+=rect(1092,306,300,137,PALE_BLUE,PALE_BLUE)+text(1120,350,'V02',27,BLUE,700)+text(1120,396,'T02 + T10',21,INK,500)
    b+=text(720,536,'외형의 긍정적 근거 + 묶음 전체의 동시 등장·얼굴 충돌 검사',22,BLUE,600,'middle')
    slides.append(item('Qwen을 부르기 전에, 샷 사이의 기본 연결을 만듭니다','같은 이름이 아니라, 실제 관측의 얼굴·몸·외형으로 연결합니다.','link','08 / GLOBAL VISUAL ASSOCIATION',b,'V-ID는 외형으로 구성한 기본 목록입니다. 연결되지 않은 관측은 미확정으로 보존합니다.',
        '<p>트랙당 기본 최대 5개 외형 후보를 골라 강한 연결부터 묶습니다. 얼굴 근거가 복장 유사도에 묻히지 않도록 후보 슬롯 일부를 얼굴 후보에 배정합니다.</p><p>사람 자동 연결은 얼굴 ≥0.55, 또는 얼굴 ≥0.48과 DINO ≥0.50/몸 ≥0.60 중 하나, 또는 몸 ≥0.86과 DINO ≥0.78의 결합 근거를 사용합니다. 비사람은 DINO ≥0.90이 기본입니다. 수치는 설정 가능한 휴리스틱이며 확률이 아닙니다.</p><p>연결 묶음 전체에 대해 실제 동시 등장과 신뢰할 수 있는 얼굴 모순을 확인합니다. 큰 묶음은 대표 모습이나 여러 독립 관측의 지지도 요구합니다. 한 개로 남은 트랙은 새 실제 개체가 확정되었다는 뜻이 아닙니다.</p>'))
    return slides

def main():
    slides=opening_slides()+review_slides()
    template=(ROOT/'presentation/method_slides_template.html').read_text(encoding='utf-8')
    sections=[]
    figures=ROOT/'presentation/method_figures';figures.mkdir(exist_ok=True)
    font=ROOT/'assets/NanumGothic-Regular.ttf'
    font_css=''
    if font.is_file():
        encoded=base64.b64encode(font.read_bytes()).decode('ascii')
        font_css="@font-face{font-family:MethodSans;src:url(data:font/ttf;base64,"+encoded+") format('truetype');font-weight:400;font-display:swap}:root{font-synthesis:weight!important}"
    for index,s in enumerate(slides,1):
        figure=svg(s['body'],index,s['title'])
        standalone=figure.replace('<defs>','<defs><style>'+font_css+'</style>',1)
        (figures/f'{index:02d}.svg').write_text(standalone,encoding='utf-8')
        sections.append(f'<section class="slide" data-title="{escape(s["title"],quote=True)}" data-phase="{s["phase"]}"><header class="slide-head"><div class="kicker">{escape(s["kicker"])}</div><h1>{escape(s["title"])}</h1><p class="subtitle">{escape(s["subtitle"])}</p></header><div class="diagram">{figure}</div><p class="takeaway">{escape(s["takeaway"])}</p><aside class="notes">{s["notes"]}</aside></section>')
    html=template.replace('<!--SLIDES-->','\n'.join(sections)).replace('<!--FONT_STYLE-->','<style>'+font_css+'</style>')
    target=ROOT/'METHOD_SLIDES.html';target.write_text(html,encoding='utf-8')
    outline=['# 04 방법 슬라이드','', '13장 · SVG 개념도 · 분석 결과나 정확도 측정값을 표시하는 자료가 아닙니다.','']
    for i,s in enumerate(slides,1):outline.extend([f'## {i:02d}. {s["title"]}','',s['takeaway'],'',s['notes'],''])
    (ROOT/'presentation/METHOD_SLIDES_NOTES.md').write_text('\n'.join(outline),encoding='utf-8')
    print(f'Created {target}: {len(slides)} slides, {target.stat().st_size/1024:.0f} KiB; standalone SVG figures and notes saved.')

if __name__=='__main__':main()
