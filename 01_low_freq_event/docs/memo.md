나 01\_low\_freq\_event를 다음과 같이 업그레이드 하고 싶어.
(1) 동영상 전체를 1fps로 downsampling
(2) downsampling된 영상의 이미지 각각에 대해 인물/물체를 detection + tracking
(3) tracking된 entity는 같은걸로 묶기
(4) VLM이 downsample된 이미지 전체 + entity 정보를 sliding window방식으로 받아서 각 인물/물체 entity가 무엇을 하고 있는지 분석.
언어적 표현으로 현재 배경은 어떻고 인물 1은 뭘 하고 있다 는 식으로 이야기해야함.
(5) visualize는 downsampling된 영상을 보면서 산출물을 볼 수 있는 형태로 구성해줘야 함 (원본 영상을 트는 기능도 포함)