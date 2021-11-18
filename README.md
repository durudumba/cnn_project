# CNN-NUMPY-PROJECT
학부 수업용으로 구현된 작은 사이즈의 뉴럴네트워크. 

- 구현 및 수정, 번역: 허종욱
- 뉴럴넷 구조 설계: [@eyyub_s](https://twitter.com/eyyub_s)
- 관련 자료: http://cs231n.stanford.edu/assignments.html

## Requirements
- Python 3.7
- Numpy

## What I do?

* Training 과정 분석

  * forward 연산 → Loss 계산 → Backward 연산 → 가중치 update 과정으로 epochs만큼 학습수행

* ReLU 활성화 함수 구현

  * `if x<0 then y=0 else y=x`
  * Backward 연산에서는 미분된 형태로 `if x<0 then y=0 else y=1`

* Adam Optimizer 구현

  * 관성(Mommentum)과 가속도(Velocity)을 반영해 최적화

* Weight 초기값

  * 비선형 함수에 효과적이지만 ReLU함수에 비효율적인 Xavier보다 He가 더 효율적임

  * 
    $$
    Var(W)=\sqrt {2 \over n_{in}}
    $$

* 나만의 CNN 구성

  * ConV - Leaky ReLU - ConV - Leaky ReLU - Flatten - FC - ReLU - FC - Linear

* *MaxPooling Layer 구현*

  * forward : 각 영역별 최대값 전달, 최대값 인덱스 backward로 전달
  * backward : 최대값 인덱스로 Downstream Gradient 전달

## Comments

MaxPooling Layer가 작동한다고 생각했는데 정확도가 줄지 않고 연산시간이 매우 오래걸렸는데 아마 잘못 만들어서 그런 것 같다. 그 외 MaxPooling Layer를 제외하고 네트워크를 설계하면 Loss가 잘 수렴하고 정확도가 올라가는 모습을 볼 수 있다. 라이브러리 모듈만 사용하다가 직접 만들어보니 어려웠다...

