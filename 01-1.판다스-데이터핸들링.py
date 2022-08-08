#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import pandas as pd
pd.__version__

# %%
titanic_df = pd.read_csv('data/titanic_train.csv')

# %%
# DataFrame의 생성
dic1 = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }
dic1

# %%
# 딕셔너리를 DataFrame으로 변환
data_df = pd.DataFrame(dic1)
print(data_df)
print("#"*30)

# %%
# 새로운 컬럼명을 추가
data_df = pd.DataFrame(dic1, columns=["Name", "Year", "Gender", "Age"])
print(data_df)
print("#"*30)

# %%
# 인덱스를 새로운 값으로 할당.
data_df = pd.DataFrame(dic1, index=['one','two','three','four'])
print(data_df)
print("#"*30)
# %%
# DataFrame의 컬럼명과 인덱스
print("columns:",titanic_df.columns)
# %%
print("index:",titanic_df.index)
# %%
print("index value:", titanic_df.index.values)
# %%
print("columns:",titanic_df.columns)
# %%
print("index:",titanic_df.index)
# %%
print("index value:", titanic_df.index.values)

# %%
# info()
# DataFrame내의 컬럼명, 데이터 타입, Null건수, 데이터 건수 정보를 제공합니다.
titanic_df.info()

# %%
titanic_df.describe()


# %%
# (중요) 자주 사용됩니다.
# value_counts()
# 동일한 개별 데이터 값이 몇건이 있는지 정보를 제공합니다.
# 즉 개별 데이터값의 분포도를 제공합니다.
titanic_df['Sex'].value_counts()