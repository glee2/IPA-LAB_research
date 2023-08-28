# 대학 기술이전 파트너 추천 시스템 개발(Inventor-licensee matchmaking for university technology licensing) 

### 연구의 필요성
- 기술 발명자와 사용자는 주로 대학의 <b>기술이전 사무소(TLO: Technology licensing office)</b> 를 통해 연결됨
- 대학 기술이전은 수익을 창출하고자 하는 <b>기술 발명자(Inventor)</b>의 공급과 사업에 필요한 기술을 구입하고자 하는 <b>기술 사용자(Licensee)</b>의 수요가 만나 거래를 형성함으로써 이루어짐
  - 기술이전 사무소는 기술 사용자가 필요로 하는 *사업 요구사항*과 기술 발명자가 제공할 수 있는 *기술 기능*을 파악하고 적절한 기술이전 파트너를 식별함
  - 그러나, 기술이전 전담인력에 의해 수행되는 만큼 적지않은 *인적・시간적* 비용을 필요로 하며, 인적 네트워크를 기반으로 하여 *좁은 탐색 범위*에 국한될 수 있음
- 정량적 데이터와 체계적인 방법론을 기반으로 하는 기술 수요-공급 매칭 프레임워크의 필요성이 대두됨
  - 기술 기능과 사업 요구사항 사이의 의미적 관계를 모델링할 수 있는 데이터 및 방법론 도입이 필요함

<!-- ![TLO를 통한 기술 수요-공급 매칭](https://github.com/glee2/Markdown-practice/blob/main/Research_example1_figure1.png?raw=true){: width="80%" height="80%} -->

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/Research_example1_figure1.png?raw=true" width="80%" height="80%"></p>

### 데이터
- 기술 기능 데이터
  - 다양한 형태의 문서로 공개된 *대학 기술 발명(연구 성과물)* 문서의 제목
    - 연구 논문
    - 학술대회 발표자료
    - 연구 과제 보고서 등
- 사업 요구사항 데이터
  - 기술이전 사무소를 통해 실제로 이루어진 *기술이전 거래 계약* 문서의 제목
- 2015년부터 2020년까지 서강대학교 기술이전 사무소에 공시된 **16,517**건의 연구 성과물과 **565**건의 실제 기술이전 계약 정보를 활용함
 
### 방법론
- fastText를 활용하여 <b>기술 기능–사업 요구사항 연계 지형(landscape)</b>을 구축함
  - fastText: 특정 단어 입력 시 주변 단어를 예측하는 skim-gram 모델을 기반으로 하여 단어를 벡터화하는 워드 임베딩 모델
- <b>코사인 유사도(Cosine similarity)</b>를 활용하여 기술 기능–사업 요구사항 벡터 간 유사도를 측정함
  - $$Cos(v_I,v_L) = {{v_I \cdot v_L} \over {|v_I| \cdot |v_L|}}, v_I: {기술\,기능\,벡터}, v_L: {사업\,요구사항\,벡터}$$
- 벡터 간 유사도가 높은 기술 기능–사업 요구사항의 기술 발명자와 사용자를 기술이전 파트너로서 적합한 것으로 판단함

<!-- ![skim-gram 모델 구조 및 기술 기능-사업 요구사항 연계 지형 예시](https://github.com/glee2/Markdown-practice/blob/main/Research_example1_figure2.png?raw=true){: width="80%" height="80%} -->

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/Research_example1_figure2.png?raw=true" width="80%" height="80%"></p>

### 연구 결과
- 성능 평가 지표
  - Top-K 기술 수요-공급 매칭률
    - 각 기술이전 거래 계약 사례의 사업 요구사항 벡터에 대해, 기술 기능-사업 요구사항 연계 지형상 가장 가까운 K개 기술 기능을 *기술 발명자 후보군*으로 선정
    - 전체 기술이전 거래 계약 중 *실제 발명자*가 K개 *발명자 후보군*에 포함된 경우의 비율

<!-- ![발명자 후보군 선정 예시 및 기술 수요-공급 매칭 결과](https://github.com/glee2/Markdown-practice/blob/main/Research_example1_figure3.png?raw=true){: width="50%" height="50%} -->

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/Research_example1_figure3.png?raw=true" width="60%" height="60%"></p>
