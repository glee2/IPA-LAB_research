# CNN을 활용한 원전 비정상 상태 판단 알고리즘 개발 </br> A convolutional neural network model for abnormality diagnosis in a nuclear power plant

### 연구의 필요성
- <b>원전(NPP: Nuclear power plant)</b>는 수천개의 개별 시스템(장비 혹은 제어 시스템)으로 이루어진 매우 크고 복잡한 시설로, 비정상 상태(Abnormal event) 발생 시 매우 큰 사회・경제적 피해를 야기함
- 원전은 이러한 비정상 상태의 발생을 식별하고 정상 상태로 복원시키기 위해 <b>비정상 절차서(AOP: Abnormal operating procedures)</b>를 운용함
  - 그러나 원전 비정상 상태 판단은 수백 개의 관련 이벤트가 존재하며 이를 식별하기 위해 수많은 센서로부터 나오는 발전소 상태변수(NPP state values)를 관찰해야 하므로 숙련된 원전 운전원에게도 어려운 작업임
  - 또한 여러 증상과 알람이 동시 다발적으로 나타날 수 있어, 해당 상황에 맞는 AOP 식별에도 어려움이 있음
- 원전 비정상 상태 판단에 필요한 시간적, 인적 비용을 절감하기 위해 데이터를 기반으로 하는 방법론 개발의 필요성이 대두됨
  - KPCA (Kernal principal component analysis), ANN (Artificial neural networks), AE (Auto-encoder) 등 다양한 데이터 마이닝 및 머신러닝 기법들이 활용된 바 있음
- 그러나 기존의 접근들은 주로 증상이 명확하며 종류가 많지 않은 원전 사고(Accident) 식별에 국한되거나, 개별 시스템 수준에서의 비정상 상태 판단을 수행하는 등 <em>한정된 수준의 분석에 머물렀음</em>
  - 개별 시스템 수준의 분석을 확장하여, 수많은 발전소 상태변수의 상호작용과 동적인 변화를 포착함으로써 전체 발전소 수준에서의 비정상 상태 판단을 수행하는 방법론의 개발이 필요함

### 데이터
- 원전 시뮬레이션 데이터
  - APR-1400 원전 모델을 모사하여 설계된 3KEYMASTER full-scope 원전 시뮬레이터로부터 정상 혹은 특정 비정상 상태의 시나리오에 따른 시뮬레이션 데이터를 생성함. 각 시나리오 하에서 시뮬레이션을 실행하여, 원전의 각 시스템에 포함된 센서로부터 나오는 <b>1,004개</b>의 발전소 상태변수에 대해 시나리오 시작 시점부터 30초에 도달하는 시점까지 매 초마다 기록된 값을 쌓아 하나의 데이터 샘플을 구성함
  - 원전의 전체 범위를 다루기 위해, 정상 상태에 더하여 10가지의 주요 비정상 상태를 식별 대상으로 선정함
  - 각 비정상 상태에 대한 개별 데이터 샘플이 고유의 특성을 가지도록, 시나리오의 초기 설정값을 임의로 바꿔가며 데이터를 생성함. 하나의 비정상 상태에 대해 300개의 데이터 샘플을 생성하여 총 <b>3,300개</b>의 데이터 샘플을 활용함

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/3_NPP_abnormality/Figure1.png?raw=true" width="80%" height="80%"></p>
<p align="center"><u><b> APR1400 원전의 구조 및 선정된 10개의 식별 대상 비정상 상태 </b></u></p>

### 방법론
- 발전소 상태변수의 수가 약 천 개에 달하여 ANN과 같은 단순한 머신러닝 모델로는 효과적인 학습이 어려우므로, 본 연구에서는 원전 시뮬레이션 데이터를 2차원 이미지로 변환하여 이미지 데이터 처리에 강점이 있는 <b>CNN (Convolutional neural networks)</b>을 활용함
- 2차원 이미지 데이터 변환
  - 원전 시뮬레이션 데이터는 각 시점에서 1,004개의 발전소 상태변수 값을 가지므로 (1, 1004) 크기의 1차원 벡터로 표현됨. 해당 벡터에 20개의 0을 더하여 총 1,024개의 값을 가지는 벡터로 구성하고, 이를 다시 <b>(32, 32)</b> 크기의 행렬로 변환하여 2차원 이미지로 표현함

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/3_NPP_abnormality/Figure2.png?raw=true" width="100%" height="100%"></p>
<p align="center"><u><b> 1차원 벡터의 2차원 이미지 변환 예시 </b></u></p>

  - 2차원 이미지로 변환된 데이터는 특정 시점에서의 발전소 상태변수를 나타냄. 이에 더하여 발전소 상태의 동적인 변화를 포착하기 위해, 5초의 간격을 두고 각 발전소 상태변수 값의 차이를 계산하여, 위와 같은 방식으로 2차원 이미지로 표현함. 이는 발전소 상태변수의 변화 패턴을 나타냄

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/3_NPP_abnormality/Figure3.png?raw=true" width="40%" height="40%"></p>
<p align="center"><u><b> 특정 시점의 발전소 상태변수에 대한 2차원 이미지 </b></u></p>

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/3_NPP_abnormality/Figure4.png?raw=true" width="40%" height="40%"></p>
<p align="center"><u><b> 발전소 상태변수의 변화량에 대한 2차원 이미지 </b></u></p>

  - 각 시뮬레이션 데이터 샘플은 발전소 상태변수의 변화 패턴을 얻을 수 있는 5초부터 30초까지 각 시점에 대해 이미지 변환을 수행하여 다시 25개의 서브샘플로 나뉘어짐. 이에 따라 11개의 정상 혹은 비정상 상태에 대한 3,300개의 데이터 샘플을 활용하여 총 82,500개의 서브샘플을 추출하였고, 이를 CNN 모델의 학습용 데이터셋으로 활용함

- CNN 구조
  - 본 연구에서 활용한 CNN 구조는 일반적인 CNN 구조를 기반으로 하여, 특정 시점에서의 발전소 상태에 대한 2차원 이미지와 해당 시점으로부터 5초 전의 상태와의 차이를 나타내는 2차원 이미지, 총 2개의 이미지 데이터를 겹쳐서 2개의 채널을 통해 입력받도록 구성함

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/3_NPP_abnormality/Figure5.png?raw=true" width="100%" height="100%"></p>
<p align="center"><u><b> 2채널 이미지를 입력으로 받는 CNN 구조 </b></u></p>

### 실험 설계
- 5-fold 교차 검증
  - 전체 데이터셋을 5개의 부분집합으로 나누어 CNN의 학습 및 평가를 여러 번 수행하여, 모든 데이터 샘플을 최소 한 번 이상 성능 평가에 활용함
- 예측 성능 평가
  - 원전 비정상 상태 판단은 다수의 클래스에 대한 분류 문제에 해당하므로, 다음과 같이 분류 예측을 위한 대표적인 성능 평가 지표를 도입함

|오차 행렬|예측: NO|예측: YES|
|-----|-----|-----|
|실제: NO|tn|fp|
|실제: YES|fn|tp|

- 위 오차 행렬을 바탕으로, $i$번째 클래스에 대한 성능 평가 지표를 다음과 같이 계산함

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/3_NPP_abnormality/Perf_eval.png?raw=true" width="80%" height="80%"></p>

### 연구 결과
- 확보한 전체 데이터셋에 5-fold 교차 검증 방식을 통해 다음의 성능 평가 결과를 얻었으며, 정상 상태와 10가지의 비정상 상태 모두에 대해 0.97 이상의 높은 정확도 및 분류 예측 성능 평가 지표를 달성하였음

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/3_NPP_abnormality/Figure6.png?raw=true" width="80%" height="80%"></p>