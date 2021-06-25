#!/usr/bin/env python
# coding: utf-8

# In[131]:


import numpy as np
import pandas as pd


# In[132]:


filepath = "titanic3_new.xls"


# In[133]:


all_df = pd.read_excel(filepath)


# In[134]:


all_df[:2]


# In[135]:


col = ['survived', 'name', 'pclass','sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ]
# sibsp, parch = 兄弟姊妹，子女 在船數量
all_df = all_df[col]


# In[136]:


all_df[:2]


# In[137]:


all_df.isnull().sum()


# In[138]:


df = all_df.drop(['name'], axis=1) #要以dataframe 的形式 才能刪除


# In[139]:


avgage = df['age'].mean()
df['age'] = df['age'].fillna(avgage)


# In[140]:


avgfare = df['fare'].mean()
df['fare'] = df['fare'].fillna(avgfare)


# In[141]:


df['sex']= df['sex'].map({'female':0, 'male':1}).astype(int) #map 的形式 轉成 int


# In[142]:


df[:2]


# In[143]:


df.isnull().sum()


# In[144]:


x_df = pd.get_dummies(data = df, columns =['embarked']) # do onehot encoding for embarked


# In[145]:


x_df[:2]


# In[146]:


ndarray = x_df.values


# In[147]:


ndarray.shape


# In[148]:


feat = ndarray[:,1:]
label = ndarray[:, 0]


# In[149]:


print(feat.shape)
print(label.shape)


# In[150]:


from sklearn import preprocessing as pps


# In[151]:


minmax = pps.MinMaxScaler(feature_range=(0,1))


# In[152]:


feat1 = minmax.fit_transform(feat) #將原先的feature 標準化至 0-1


# In[153]:


feat1[:2]


# In[165]:


sp = np.random.rand(len(all_df))<0.8 #這裡要使用all_df 而不是df 才可以做後續的模組化
train_df = all_df[sp]
test_df = all_df[~sp]


# In[166]:


print(len(df), len(train_df), len(test_df))


# In[167]:


print("train_df: ",train_df[:2])
print("test_df: ", test_df[:2])


# In[168]:


def pps(raw_df): # 將prepross 模組化
    #col = ['survived', 'name', 'pclass','sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',]
    # sibsp, parch = 兄弟姊妹，子女 在船數量
    #all_df = raw_df[col]    
   

    df = raw_df.drop(['name'], axis=1) #刪除col，修改nan
    avgage = df['age'].mean()
    df['age'] = df['age'].fillna(avgage)
    avgfare = df['fare'].mean()
    df['fare'] = df['fare'].fillna(avgfare)
    df['sex']= df['sex'].map({'female':0, 'male':1}).astype(int)
    
    x_df = pd.get_dummies(data = df, columns =['embarked'])#one hot encoding
    ndarray = x_df.values
    feat = ndarray[:,1:]
    label = ndarray[:, 0]
    
    from sklearn import preprocessing as pps    #標準化 0-1
    minmax = pps.MinMaxScaler(feature_range=(0,1))
    feat1 = minmax.fit_transform(feat)
    
    return feat1,label
    


# In[169]:


train_feat, train_label = pps(train_df)
test_feat, test_label = pps(test_df)


# In[172]:


train_feat[:2], train_label[:2]


# In[171]:


test_feat[:2],test_label[:2]


# In[ ]:




