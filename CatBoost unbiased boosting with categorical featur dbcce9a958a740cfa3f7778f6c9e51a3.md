# CatBoost : unbiased boosting with categorical features

날짜: 2023년 3월 25일
담당자: 진욱 이

[CatBoost_unbiased boosting with categorical features.pdf](CatBoost%20unbiased%20boosting%20with%20categorical%20featur%20dbcce9a958a740cfa3f7778f6c9e51a3/CatBoost_unbiased_boosting_with_categorical_features.pdf)

[CatBoost_unbiased boosting with categorical features ko.pdf](CatBoost%20unbiased%20boosting%20with%20categorical%20featur%20dbcce9a958a740cfa3f7778f6c9e51a3/CatBoost_unbiased_boosting_with_categorical_features_ko.pdf)

- Deepl을 이용한 번역본

![Untitled](CatBoost%20unbiased%20boosting%20with%20categorical%20featur%20dbcce9a958a740cfa3f7778f6c9e51a3/Untitled.png)

## INDEX

[Abstract](https://www.notion.so/CatBoost-unbiased-boosting-with-categorical-features-dbcce9a958a740cfa3f7778f6c9e51a3)

1. [Introduction](https://www.notion.so/CatBoost-unbiased-boosting-with-categorical-features-dbcce9a958a740cfa3f7778f6c9e51a3)
2. [Background](https://www.notion.so/CatBoost-unbiased-boosting-with-categorical-features-dbcce9a958a740cfa3f7778f6c9e51a3)
3. [Categorical feature](https://www.notion.so/CatBoost-unbiased-boosting-with-categorical-features-dbcce9a958a740cfa3f7778f6c9e51a3)
4. [Prediction shift and ordered boosting](https://www.notion.so/CatBoost-unbiased-boosting-with-categorical-features-dbcce9a958a740cfa3f7778f6c9e51a3)
5. [Practical implementation of ordered boosting](https://www.notion.so/CatBoost-unbiased-boosting-with-categorical-features-dbcce9a958a740cfa3f7778f6c9e51a3)
6. Experiments
7. Conclusion

Appendices

---

<aside>
<img src="https://www.notion.so/icons/stars_orange.svg" alt="https://www.notion.so/icons/stars_orange.svg" width="40px" /> **Summary(with CHAT GPT)**

</aside>

:

CatBoost는 범주형 데이터를 처리하는 새로운 방법을 제시하고, 기존의 그래디언트 부스팅 알고리즘을 조작하여 타겟 누수를 개선하는 알고리즘입니다. 이 논문은 CatBoost의 주요 알고리즘 기술을 제시하고 있으며, CatBoost가 다른 부스팅 구현체들보다 다양한 데이터셋에서 우수한 성능을 보이도록 만든 알고리즘 기술들에 대해 다루고 있습니다.

CatBoost의 주요 알고리즘 기술로는 ordered boosting과 counter-based feature generation 등이 있습니다. ordered boosting은 범주형 특성 값들의 순서를 고려하여 부스팅을 수행하는 기법입니다. 이는 범주형 데이터를 처리하는 데 있어서 중요한 역할을 합니다. counter-based feature generation은 범주형 데이터의 특성을 고려하여 새로운 특성을 생성하는 기법입니다. 이를 통해 범주형 데이터의 정보를 더 잘 활용할 수 있습니다.

CatBoost는 또한 기존 그래디언트 부스팅 알고리즘에서 발생할 수 있는 타겟 누수(target leakage)를 방지하기 위한 몇 가지 기술들을 적용합니다. 타겟 누수란 예측 시점에서 사용할 수 없는 데이터가 데이터셋에 포함되어 있는 오류를 말합니다. CatBoost는 이를 방지하기 위해 데이터셋을 무작위로 섞는 기술과 범주형 데이터의 처리 방법을 개선하는 기술 등을 도입합니다.

CatBoost는 다양한 데이터셋에서 우수한 성능을 보이는 것으로 알려져 있습니다. 예를 들어, CatBoost는 Kaggle에서 열린 "TalkingData AdTracking Fraud Detection Challenge" 대회에서 1위를 차지하였으며, Criteo에서 열린 "Display Advertising Challenge" 대회에서도 우수한 성적을 거두었습니다. 이러한 결과는 CatBoost가 범주형 데이터를 처리하는 데 있어서 우수한 성능을 보이는 것으로 해석됩니다.

---

## Abstract

이 논문에서는 Catboost의 핵심 알고리즘을 설명하며 Catboost에 도입된 두 가지 중요한 알고리즘 발전을 소개한다.

1. **기존 알고리즘을 대체하는 순열기반의 부스팅(orderd boosting)의 구현**
2. **categorical feature 처리를 위한 혁신적인 알고리즘**

이 두가지 기술은 부스팅 알고리즘에 존재하는 타깃 누출(target leakage)로 인한 예측이동(prediction shift)을 방지하기 위해 만들어졌다.

본 논문에서는 이 문제에 대하여 자세한 분석을 제공하고 알고리즘이 그 문제를 효과적으로 해결하여 우수한 결과를 보여준다.

## 1. Introduction

**그레디언트 부스팅(Gradient boosting)**은 수년동안 많은 문제를 학습하는 주요 방법으로 사용되어져 왔으며 좋은 결과를 달성하는 기법이다.  또한 기본적으로 그레디언트 하강(gradient desent)을 수행하여 ensemble predicter 를 구축하며 탄탄한 이론적 결과를 뒷받침한다.

이 논문에서 기존의 그레디언트 부스팅 구현에서 통계적 문제에 직면해 있음을 보여준다. 

여러 단계의 부스팅을 거처 얻은 예측 모델 F는 모든 훈련 예제의 target에 의존하며 그렇기에 결국 prediction shift으로 이어진다. 

*(test sample $x$ 의 $F(x)| x$ 에서 train sample $x_k$의 $F(x_k)|x_k$분포로의 이동)

또한 categorical feature를 전처리하는 알고리즘에서 비슷한 문제가 있는데 그레디언트 부스팅에 사용하는 효과적인 방법 중 하나인 범주를 target statistics로 변환하는 것이다. 하지만 target statistics는 target leakage 와 prediction shift 를 유발할 수 있다. (section3에서 다룬다)

위 두가지의 문제를 해결하기 위해 정렬원리(ordering principle)를 제안하며 이를 바탕으로 새로운 알고리즘을 도출한다.

## 2. Background

**그레디언트 부스팅(Gradient boosting)은** 부스팅계열의 앙상블 알고리즘이다.

약한예측기를 만들어 강한예측기를 만드는 부스팅이며

각 모델을 순차적으로 생성하여 만들 모델을 바탕으로 다음 모델을 생성한다.

최종적으로 만들어진 weaker model들을 결합하여 최종 모델을 만든다.

### **gradient boosting procedure**

<aside>
<img src="https://www.notion.so/icons/stars_orange.svg" alt="https://www.notion.so/icons/stars_orange.svg" width="40px" /> **procedure**

**A.**    $h^t = arg\underset{h \in H}{min}\mathfrak{L}(F^{t-1} + h) = arg\underset{h \in H}{min}\mathbb{E}L(y, F^{t-1}(x)+h(x))$.                       **

**B.     $F^t = F^{t-1}+\alpha h^t$**

($F^t: \mathbb{R}^m \to \mathbb{R}$  ,         $h^t: \mathbb{R}^m \to \mathbb{R}$  

traing dataset :$D = \left\{ \\ (x_k,y_k  \right )\}_{k=1..n}$    

$where, x_k =(x^1_k,...,x^m_k)$, 

</aside>

B : t번째 부스팅 모델은 t-1번쨰 부스팅모형(지금까지 누적된 모형)에 현제시점에서의 찾은 모형($h^t$)과 그의 가중치 $\alpha$ 를 고려한 모델을 더해준것

A :  expectstion Loss를 최소화하는 함수이다 → t-1시점까지 누적된 모델에 어떤 h의 모형을 넣었을때 정답과 누적된 모형에 의해서 추정된 loss를 최소화하는  $h^t$ 를 찾는것

*weak model $h^t$ 를 만든후 이를 바탕으로 $F^t$ model을 생성 (expectstion Loss를 최소화하는 함수이다.)

- $\alpha$ : step size
- x는 m개의 feature이고 y는 target이다. $(x_k,y_k)$는 미지의 분포를 따라 동일하게 분포
- 여기서 손실  $\mathfrak{L}(F) := \mathbb{E}L(y,F(x))$
- $L(· , ·)$: 평활화(smooth) 손실함수

<aside>
<img src="https://www.notion.so/icons/stars_orange.svg" alt="https://www.notion.so/icons/stars_orange.svg" width="40px" /> **gradinet bossdting**

( i )    $h^t = arg\underset{h \in H}{min}\mathbb{E}(-g^t(x,y) - h(x) )^2$  

                                                   **

( ii )  $where, -g^y(x,y) := \frac{\partial L(y,s)}{\partial s}$ $|_{s=F^{t-1}(x)}$

</aside>

<aside>
<img src="https://www.notion.so/icons/stars_orange.svg" alt="https://www.notion.so/icons/stars_orange.svg" width="40px" /> **Loss funtion: Squared Error**

(iii)   $L(y, F^{t-1}(x)+h(x)) = \frac{1}{2}(y-(F^{t-1}(x) + h(x)))^2$

(iv).  $-g^y(x,y) := \frac{\partial L(y,s)}{\partial s}|_{s=F^{t-1}(x)}$ $= y-(F^{t-1}(x) + h(x)) = Residual$

</aside>

(i) : 네거티브 그레디언트(negative gradient)에서 현제 h(x)함수를 추정했을때 그차이에대한 expectation을 최소화하되는 t번째 시점에서 트리를 만드는것  

loss funtion에 관한 최소화 문제는 그레디언트 디센트(gradient descent) 으로 해결할 수 있다.

1번식과같이  네거티브 그레디언트를 잘 예측하는 h(x)를 h^t로 선택해야한다. 

그레디언트 부스팅을 각 단계의 잔차를 예측하여 전체로스를 줄이는 알고리즘이라고 하는 이유를

3번식과 4번식이 보여주고 있다.

### statistical Issue

캣부스트 이전에 존재했던 그레디언트 부스팅에는 몇가지 statistical Issue가 존재했다.

**1.prediction shift**

각각의 단계에서 그레디언트를 구할때 매번 같은 traing data를 사용하기 때문에

이에 대해서만 학습하게 되고 실제 test data가 주어졌을때 [도입부분의 설명](https://www.notion.so/CatBoost-unbiased-boosting-with-categorical-features-dbcce9a958a740cfa3f7778f6c9e51a3)처럼 분포의 이동이 일어난다.

<aside>
<img src="https://www.notion.so/icons/stars_orange.svg" alt="https://www.notion.so/icons/stars_orange.svg" width="40px" /> $arg\underset{h \in H}{min}\mathbb{E}(-g^t(x,y) - h(x) )^2   \approx  arg\underset{h \in H}{min} \frac{1}{n}\sum_{k=1}^{n}(-g^t(x_k,y_k) - h(x_k) )^2$

                                $F^{t-1}(x_k) | x_k \neq  F^{t-1}(x) | x$  

</aside>

**2.Preprocessing Categoricasl Features(Target Leakage)**

기존 one-hot encoding을 하게되면 이를 표현하기 위한 벡터가 많이 필요하기에

부스팅 모델들은 타겟의 통게량(target statistic)으로 만들어 주었다.

*target statistic : Categoric feature들을 이들의 target값의 statistic으로 대체

![ViXB709.png](CatBoost%20unbiased%20boosting%20with%20categorical%20featur%20dbcce9a958a740cfa3f7778f6c9e51a3/ViXB709.png)

하지만 target statistic는 예측시점에 알수 없는 정보를 이용하기때문에

이로 인한 **Target Leakage** 가 일어난다

⇒ 이두가지를 해결하기 위해 논문에서는 

**Odering Principle** 을 제안하고 있다.

***Prediction shift → Odered boosting**

***Target Leakage→  Ordered TS**

## 3. Categorical features

### 3.1 Related work on categorical features

기존의 전처리방법

One-hot encoding: 

- 각 category에 대해 Feature를 나타내는 새로운 Binary Feature를 추가하는 것
- High cadinality일 경우 infeasible하게 많은 binary Feature가 필요

위의 범주가 많아질때의 문제를 방지하기 위해 범주를 clustering하여 수를 줄이거나 새로운 숫자형 데이터로 뱐화하는 target statistic 방식을 소개한다.

Micci-Barreca는 TS를 새로운 수치Feature로 고려할 것을 제안

LightGBM의 경우 부스팅 각 단계에서 그레이언트 통걔량으로 변환

2가지의 문제

-각 단계에서  각 범주의 값에 대한 통계량을 계산하기에 계산시간 문제

-분할된 각 노드에 범주를 저장하기에 메모리소비 문제

 ⇒ 문제를 해결하기 위해 tail부분을 한 범주로 그룹화하고 정보의 일부를 순환시킴

TS를 새로운 Feature로써 사용하는것이 적은 정보손실을 가지는 효율적인 방법으로 보인다.

그렇기에 이 논문에서는 TS를 계산하는것에 더 초점을 둔다

### 3.2 Target statistics

스무딩(smoothing)에서 해결하고자 하는 이슈는 노이즈 데이터로 인해 다른 값들에 비해 극단적인 값들이 나올

이슈들을 해결하기 위해 스무딩을 한다.

$\hat x_k^i \approx \mathbb E(y|x^i = x_k^i)$

- Greedy TS without smoothing

학습데이이터에 존재하는 객체들에서 각 카테고리에 경우의 수에 해당하는 그 객체들이 가지는 y값의레 평균값

- Greedy TS with smoothing

$$
\hat x_k^i = \frac{\sum_{j = 1}^{n}\textbf{1}_{\left\{x_j^i = x_k^i \right\}}·y_j+ap}{\sum_{j = 1}^{n}\textbf{1}_{\left\{x_j^i = x_k^i \right\}}+a}
$$

- a>0 :파라미터
- p: 전체데이터셋에 대한 타겟값

**example(binary classification)**

|  | y=1 | y=0 | TS |
| --- | --- | --- | --- |
| A | 10 | 10 | 0.5 |
| B | 40 | 10 | 0.8 |
| C | 10 | 40 | 0.2 |
| D | 25 | 25 | 0.5 |
| E | 1 | 0 | 1 |

**Greedy TS 의 한계**

![스크린샷 2023-04-01 오후 10.10.27.png](CatBoost%20unbiased%20boosting%20with%20categorical%20featur%20dbcce9a958a740cfa3f7778f6c9e51a3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-01_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_10.10.27.png)

그렇기에 논문에서 저자는 Target statistic 이 가져야할 두가지 속성을 주장한다.

- **property1**

$\mathbb E(\hat x^i|y = v) = \mathbb E(\hat x^i_k| y_k = v)$   $where$   $(x_k,y_k)$ $is$ $the$  $k-th$ $training$ $example$

*TS산출과정에서 target leakage를 방지하는 것이다.

*학습데이터와 테스트 데이터에 관해서 해당하는 expectation값이 같아야한다.

- **property2**

Effective usage of all training data for calculating TS features and for learning a model

*가급적이면 학습을 하는데(TS를 학습) 있어서 모든 데이터셋을 효과적으로 사용해라

위방법들의 일반적인 아이디어들을 제시한다

- $x_k$를 제외한  $\mathcal{D}_k\subset\mathcal{D}$ \ $\left\{x_k \right\}$ 의 부분집합에서 TS를 계산

$$
\hat x_k^i = \frac{\sum_{x_j\in\mathcal{D}_k}^{}\textbf{1}_{\left\{x_j^i = x_k^i \right\}}·y_j+ap}{\sum_{x_j\in\mathcal{D}_k}^{}\textbf{1}_{\left\{x_j^i = x_k^i \right\}}+a} .....(5)
$$

위 방법을 구현하기 위한 방법론 두가지를 소개한다

### 1.**Holdout TS**

$\mathcal{D}   = \mathcal{\hat D_0}\cup\mathcal{\hat D_1}$

데이터 셋을 두개로 나눈다

- $\mathcal{\hat D_0}$ 는  ts를 계산하는것에만 쓰이고 $\mathcal{\hat D_1}$ 만 학습하는데 쓰인다.

**이는 p1 은 만족하지만 모든 데이터셋을 사용하지 않은 p2를 위반한다.**

### 2.Leave-one-out TS

Dataset을 분할후, TS를 계산하는데 쓰인 dataset은 traing에 사용하지 않음

쉽게말해 (5)번식처럼 해당 데이터를 제외하고 TS를 산출하는 것이다.

이는 **p2는 만족하지만 p1은 만족어ㄷ**

 

**그렇기에 Target Leakage가 발생하지 않기위해 이 논문에서**

**ordered TS를 제시한다**

### Ordered TS

가지고있는 학습데이터가 시계열데이터가 아니더라고 artificial time을 부여하고

random permutation 을 시켜준다 →$\sigma$

$\mathcal{D_k} = \left\{x_j:\sigma(j)<\sigma(k) \right\}$

![스크린샷 2023-04-02 오전 11.50.15.png](CatBoost%20unbiased%20boosting%20with%20categorical%20featur%20dbcce9a958a740cfa3f7778f6c9e51a3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-02_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_11.50.15.png)

k-th Ts를 계산할때 k-1번까지의 target value만을 사용하는 것이다

또한 모든 dataset을 사용할 수 있기에

**p1, p2를 모두 만족한다0**

## 4. Prediction shift and oredered boosting

### 4.1 prdiction shift

기존의 그레디언트 부스팅에서 predicion shift 의 문제가 있다. 

target value를 이용하기에 train, test data의 분포차이로 인해 오버피팅을 발생시킨다. 

그렇기에 학습하는 단계마다 independent samples 를 사용해야한다. 

$$
h^t = arg\underset{h \in H}{min}\mathbb{E}L(y,F^{t-1}(x)+h(x) ) ...(i)
$$

$$
 = arg\underset{h \in H}{min}\mathbb{E}(-g^t(x,y) - h(x) )^2 ...(ii)
$$

$$
 = arg\underset{h \in H}{min} \frac{1}{n}\sum_{k=1}^{n}(-g^t(x_k,y_k) - h(x_k) )^2 ...(iii)
$$

$$
F^t = F^{t-1}+\alpha h^t ...(iv)
$$

- Chain of shift

1.Conditinal distribution of the gradient $g^t(x_k,y_k)|x_k$는 test example의 distribution $g^t(x,y)|x$로부터 shift되고

2.이로 인해 Equation(iii)의해 정의된 base predictor $h^t$는 Equation(ii)의 solution $h^t$로부터 biased

3.따라서 equation  (iv)에 의해 biased $h^t$로 학습된 model $F^t$의 generalization abiliy 영향을 미친다

prediction shift를 피하기 위해서는 $F^{t+1}$을 학습하기 위해 $F^t$의 gradient $-g^t(x_k,y_k)$필요

이때, $F^t$ training 과정에서 $x_k$가 사용되지 않았어야 unbiased $-g^t(x_k,y_k)$도출 가능

 

각 부스팅 단계에서 $-g^t(x_k,y_k)$계산시,

$F^t$를 학습하는데 쓰인 dataset과 independent dataset을 사용함으로써 prediction shift를 피할 수 있음

### 4.2 Oredered Boosting

![스크린샷 2023-04-02 오후 7.39.52.png](CatBoost%20unbiased%20boosting%20with%20categorical%20featur%20dbcce9a958a740cfa3f7778f6c9e51a3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-02_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_7.39.52.png)

 각 과정을 통해 잔차를 계산(각 단계에서 본인은 학습이 안되었음으로 그 타켓값은 영향을 미치지 않음)

이 과정의 반복을 통해 최종적으로 마지막 하나의 객체를 남기고 사용하여 모델을 만들고 잔차를 계산

ordered TS 와 oredered boosting 은 같은 permutation을 사용

## 5. Practical implementation of ordered boosting

Catboost는 두가지의 부스팅모드를 가진다.

- Plain mode

-기존에 GBDT 알고리즘과 같고 Categorical Feature만 ordered TS 를 사용하는 방식

- Ordered mode

 -앞서 설명한 Ordered Boosting 알고리즘을 적용한방법

![스크린샷 2023-04-02 오후 7.45.17.png](CatBoost%20unbiased%20boosting%20with%20categorical%20featur%20dbcce9a958a740cfa3f7778f6c9e51a3/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-02_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_7.45.17.png)

알고리즘의 자세한 설명은 Appendices B에 나와있다.

위 알고리즘을 간단하게 설명하자면 

- training dataset의 s+1개의 독립적인 무작위 순열을 생성
- $\sigma_1,...,\sigma_s$ 는 트리구조를 정의하는 분할평가에사용 & $\sigma_0$는 트리의 리프값 선택에 사용
- 주어진 순열에서 짧은 히스토리를 가진 예제의 경우,  TS와 예측 모두 ordered boosting에 의해 사용
- **하나의 순열만 사용 모델예측의 분산이 증가하지만 여려개의 순열을 사용하면 이를 줄일수 있음**

이 효과에 대해서 다음에서 설명한다

### **Building a tree**

- catboost의 base predictor은 oblivious decision trees(decision tables) :=결정테이블 or 무의식 의사결정트리
- 트리의 전체레벨에 걸쳐 동일한 분할 기준이 사용
- 균형있고 과적합이 덜 발생하고 실행속도를 크게 높일수 있음
- **Ordered boosting mode에서는 지원모델 $M_{rj}$을 유지하며, 여기서 $M_{rj}(i)$ 는 순열 $\sigma_r$의 첫번째 j예제에 기반한 i번째 예제에 대한 현제예측**
- 순열은 트리학습 절차에 영향을 준다
- **Plain boosting modes는 GBDT와 비슷하게 작동하지만, categorical features 있을때는 $\sigma_1,...,\sigma_s$ 에 기반한 TS에 해당하는 모델 $M_{r}$을 지원하는 s를 유지**

### **Choosing leaf values**

- 모든 트리가 주어지면 최종모델 F의 leaf값은 두 mode 모두에 의해 동일한 gradient boosting 절차에 의해 계산
- 훈련예제(training example) i 는 $leaf_0(i)$ 와 일치
- 여기서 순열$\sigma_0$를 사용하여 TS를 계산
- 테스트 시점에 최종모델 F를 새 예제에 적용하면 전체 훈련데이터에 계산된 TS를 사용 ($\because section3.2$)

### **Complexity**

- 실제 구현에서 알고리즘의 계산 복잡성을 크게 줄이는 한 가지 중요한 트릭을 사용한다.
- ordered mode에서  $O(s,n^2)$값 $M_{r,j}(i)$대신 $M_{r,j}^{'}(i) := M_{r,2^j}(i)$로 저장하고 업데이트한다

        ($j = 1,...,[log_2N]$    $\forall i$ ,$\sigma_r(i)<2^{j+1}$  )

### **Feature combinations**

- Catboost의 또다른 디테일은

### **Other important details**

## 6. Experiments

## 7. Conclusion

## reference

>[https://www.youtube.com/watch?v=2Yi_Jse_7JQ&t=2929s](https://www.youtube.com/watch?v=2Yi_Jse_7JQ&t=2929s)

>[https://www.youtube.com/watch?v=-w_6wDJQCZY&t=1890s](https://www.youtube.com/watch?v=-w_6wDJQCZY&t=1890s)

>[https://soobarkbar.tistory.com/34](https://soobarkbar.tistory.com/34)
>[https://gentlej90.tistory.com/100](https://gentlej90.tistory.com/100)