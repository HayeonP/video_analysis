지금 결과가 아직도 처참하다



아래는 내가 수기로 기입한 같은 대상을 가리키는 ID의 리스트이다. (더 많은데, 힘들어서 어느정도까지만 했따. 문제가 심각하다는걸 파악해라.)
빨간머리인물 E0001, E0002, E0003, E0004, E0019, E0029, E0032, E0034, E0055, E0075, E0089, E0075
인물1 E0013, E0017
인물2 E0025, E0028
작은 새 E0043, E0045, E0047, E0050, E0052, E0057, E0060, E0061, E0057, E0058, E0060, E0061, E0070, E0071, E0072, E0073, E0074, E0079m 

큰 새1 E0038, E0041, 
큰 새2 E0042, E0038, E0044, E0049, E0051, E0053, E0054, 
큰 새3 E0046, 
큰 새4 E0068, E0069
용 E0062, E0063

등등...

보니까 동일 인물에대한 tracking이 거의 안되고 있다. 다음과 같은 연구, 혹은 다음과 유사한 연구들을 찾아서 같이 적용할수 있지 않은가?
연구	대상으로 삼은 문제	핵심 방법	네 작업과의 관계
MuHPC — Face, Body, Voice · ICCV Workshops 2021
CAST · Eurographics 2022
CAL — Clothes-Changing Person Re-Identification With RGB Modality Only · CVPR
Person Search in Videos with One Portrait Through Visual and Temporal Links · ECCV
이런것들을 활용해서 트래킹하면 훨씬 좋겠지? 

움직이는 대상에 대한 트래킹 모듈이 따로 적용되어야 할 것 같다. 내 생각에는 shot별로 object tracking을 해서 샷별 entity 리스트를 만들자.
그 이후 vlm이 샷별 시간에따른 crop box들을 한번에 받아서 샷 내에서 동일대상을 식별하자.
참고로 인물이 아닌 동물이나 괴물을 인물이라고 구분하는것도 있는걸 보니, detect된 object를 다시 한번 평가하는 작업이 필요할것 같다.
샷별로 entity 통일이 끝났으면, 이 entity들을 활용해서 그 샷을 요약해라.
그리고 모든 샷에 대해서 이 작업이 끝났으면, 샷들끼리 Entity 연결과 요약텍스트 보정이 필요하다.
즉, 같은 대상은 같은 대상임을 인지하면서 샷별 요약을 해야한다는거다.
이때 개별 샷에서만 봤을땐 몰랐던걸 유기적으로 이어지는 작업이 반드시 필요하다.


요약하자면,
샷별로 tracking을 활용해 human entity list와 other entity list를 만들고
entity들을 활용해 샷을 텍스트로 요약설명해라.
이후 샷들간의 관계를 파악하여 global human entity list, global other entity list를 만들고
샷들간의 요약정보와 global entity list를 함께 사용해 shot별 요약을 보정해라.
최종적으로는 요약과 entity만 가지고 대본을 쓸수 있을 정도가 되어야 한다.

이거 어떻게 작업할건지 나한테 계획해서 보고해라.