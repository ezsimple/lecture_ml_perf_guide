#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn

# 붓꽃 예측을 위한 사이킷런 필요 모듈 로딩
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

# 붓꽃 데이터 세트를 로딩합니다.
iris = load_iris()

# 키값 확인
iris.keys()

# iris.data는 Iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있습니다.
iris_data = iris.data
iris_features = iris.feature_names

# iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있습니다.
iris_label = iris.target
iris_target_names = iris.target_names

# 붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다.
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target

# 학습 데이터와 테스트 데이터 세트로 분리
# 관습적으로 X_ : Feature, y_ : Target을 나타냅니다.
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)
#학습용    테스트용 학습용    테스트용
#Feature  Feature Target   Target

# test_size=0.2 학습용 80%, 테스트용 20% 생성하라는 의미
# random_state는 같은 seed 값을 사용하므로, 동일한 데이터를 사용하도록 합니다.

# 학습 데이터 세트로 학습(Train) 수행
# 학습 수행
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train) # 학습용 Feature, 학습용 Target을 이용해서 학습을 진행합니다.

# 테스트 데이터 세트로 예측(Predict) 수행
# 학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행.
pred = dt_clf.predict(X_test) # 예측이므로 Feature값만 제공됩니다. (y_test(테스트용 Target)은 제공하지 않습니다)

# 예측 정확도 평가
# 학습용 y_train 데이터와, 예측치(pred)를 사용해서 정확도 점수를 산정합니다.
from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))