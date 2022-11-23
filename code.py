#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 한글 폰트를 사용하기 위한 코드
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Linux':
    plt.rc('font', family='NanumGothic')


# In[ ]:


# data 가져오기
df = pd.read_csv('./data.csv')
# data 나누기
df = df.dropna(subset=['날짜'])
# 결측값 확인
print(df.isnull().sum())
# 현재 기온, 강수량, 일조, 일자, 적설, 전운량, 미세먼지에서 결측값을 확인할 수 있다


# In[3]:


df


# In[4]:


# KNN 적용 전 데이터 전처리, 결측값이 너무 많은 값 제외
df_im = df.drop(['날짜','시간', '지역','연료원', '강수량(mm)', '적설(cm)'], axis = 1)


# In[5]:


from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2)
data_filled = imputer.fit_transform(df_im)


# In[6]:


# numpy data -> DataFrame 으로 변환
data_filled = pd.DataFrame(data_filled, columns = df_im.columns)
# 결측값 있는지 확인
data_filled.isnull().sum()


# In[7]:


# 태양광 발전량 확인
plt.figure(dpi=200)
plt.title("태양광 발전량")
plt.xlabel('발전량')
plt.ylabel('갯수')
plt.hist(data_filled['전력거래량(MWh)'], bins=30)
plt.show()

# 전력거래량이 0인 값이 대부분임을 알 수 있다.


# In[8]:


import seaborn as sns # 시각화 라이브러리

sns.pairplot(data_filled)
plt.show()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
# 상관관계 분석 (변수 선택 과정)
cmap = sns.light_palette("darkgray", as_cmap = True)
sns.heatmap(data_filled.corr(), annot = True, cmap = cmap)
plt.show()

# 기온, 풍속, 일조, 일사 선택


# In[10]:


# data 간 상관관계 더 크게 출력

tmp = (data_filled
 .loc[:, ['일사(MJ/m2)', '일조(hr)', '풍속(m/s)', '기온(°C)', '전력거래량(MWh)', 'PM10 평균']]
 .melt(id_vars='전력거래량(MWh)',
       var_name='분류', value_name='발전량')
 )
tmp


# In[11]:


sns.lmplot(tmp, x='발전량',  y='전력거래량(MWh)', col='분류', col_wrap=2,
           facet_kws={'sharex': False, 'sharey': False})
plt.show()

# 그래프를 살펴본 후, 우상향 그래프를 보이지 않는 풍속과 미세먼지 데이터를 제거


# In[12]:


# 편차를 줄이기 위해 표준화 과정 진행
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# 표준화 과정 진행
df_1 = sc.fit_transform(data_filled)

# 데이터 
df_sc = pd.DataFrame(df_1, columns = data_filled.columns)
df_sc


# In[13]:


# 태양광 발전량 확인
plt.figure(dpi=200)
plt.title("태양광 발전량")
plt.xlabel('발전량')
plt.ylabel('갯수')
plt.hist(df_sc['전력거래량(MWh)'], bins=30)
plt.show()


# In[14]:


# 일조 + 일사 합친 컬럼 SUN 을 만들어준다.

df_sc = (df_sc
 .assign( SUN = lambda df: (df_sc['일조(hr)'] + df_sc['일사(MJ/m2)']) / 2)
 )


sns.lmplot(df_sc, x='SUN', y='전력거래량(MWh)')
plt.show()

# 전반적으로 비례 관계를 보인다


# In[15]:


# 기온, 풍속, 일조, 일사, SUN 선택

df_res = df_sc.drop(['습도(%)','전운량(10분위)', 'PM10 평균', 'PM2.5 평균', '풍속(m/s)'], axis = 1)
df_res # feature 모음집 -> 전력거래량 -> label, 나머지 특징들이 -> train


# In[16]:


# data 간 상관관계 DataFrame 출력

tmp = (df_res
 .loc[:, ['일사(MJ/m2)', '일조(hr)', '기온(°C)', 'SUN', '전력거래량(MWh)']]
 .melt(id_vars='전력거래량(MWh)',
       var_name='분류', value_name='변수 크기')
 )
tmp


# In[17]:


# 상관관계 그래프 체크
sns.lmplot(tmp, x='변수 크기',  y='전력거래량(MWh)', col='분류', col_wrap=2,
           facet_kws={'sharex': False, 'sharey': False})
plt.show()


# In[18]:


# 상관관계 다시 분석

cmap = sns.light_palette("darkgray", as_cmap = True)
sns.heatmap(df_res.corr(), annot = True, cmap = cmap)
plt.show()


# In[19]:


#data, label 나누기 + 풍속 제외
df_train = df_res.drop(['전력거래량(MWh)'], axis = 1) 
df_label = df_res.drop(['기온(°C)', '일조(hr)','일사(MJ/m2)','SUN'], axis = 1)


# In[20]:


df_train


# In[21]:


df_label


# In[22]:


# df -> np
df_res = df_train.values
df_label = df_label.values


# In[23]:


print(df_res.shape)
print(df_label.shape)


# In[24]:


df_res


# In[25]:


from sklearn.model_selection import train_test_split

# train, test data 분리 (8 : 2)
x_train, x_test, y_train, y_test = train_test_split(df_res, df_label, test_size=0.2, shuffle=False)


# In[26]:


# 잘 분리됐는지 확인
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[40]:


from sklearn import ensemble
from sklearn.model_selection import GridSearchCV


N_ESTIMATORS = 1000

#params = {
#   'max_features':[0.6, 0.8, 1],
#    'max_depth':[3,4,5,6,7,8,9]
#} 

rf = ensemble.RandomForestRegressor(n_estimators=1000, random_state=0,
                                    max_depth=7, max_features=0.8,
                                    verbose=True,
                                    n_jobs=-1)

#grid_cv = GridSearchCV(rf, params, cv=5, n_jobs=-1, verbose=1)
#grid_cv.fit(x_train, y_train)


# In[39]:


print('최적 하이퍼 파라미터 : \n', grid_cv.best_params_)


# In[41]:


rf.fit(x_train, y_train)


# In[42]:


from sklearn.metrics import accuracy_score

# 정확도를 보기 위해 float -> int 형으로 형변환
pred = rf.predict(x_test)
pred = np.round(pred, 0).flatten()
y_test = np.round(y_test, 0).flatten()

# 정확도 출력
acc = accuracy_score(y_test, pred)
print("랜덤 포레스트 정확도 : {0:.4f}".format(acc * 100))


# In[30]:


pred


# In[31]:


y_test


# In[ ]:




