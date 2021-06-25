#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
from sklearn import preprocessing as pps


# In[47]:


import tensorflow as tf
from tensorflow import keras


# In[48]:


all_df = pd.read_excel(filepath)


# In[49]:


col = ['survived', 'name', 'pclass','sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ]
# sibsp, parch = 兄弟姊妹，子女 在船數量
all_df = all_df[col]


# In[50]:


sp = np.random.rand(len(all_df))<0.8 #這裡要使用all_df 而不是df 才可以做後續的模組化
train_df = all_df[sp]
test_df = all_df[~sp]


# In[51]:


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
    


# In[52]:


train_feat, train_label = pps(train_df)
test_feat, test_label = pps(test_df)


# In[84]:


ver = np.random.rand(len(train_df)) <0.1
val_feat = train_feat[ver]
df_feat = pd.DataFrame(val_feat)
val_label = train_label[ver]
df_label = pd.DataFrame(val_label)
val_data = pd.concat([df_feat,df_label])


# In[81]:


val_data[:2]


# In[53]:


from tensorflow.keras import layers


# In[54]:


model = keras.Sequential()


# In[55]:


model.add(layers.Dense(40, activation="relu", input_shape=(9,)))
#model.add(layers.Dense(units=40, input_dim=9,kernal_initializer ='uniform', activation='relu'))


# In[56]:


model.add(layers.Dense(30, activation="relu"))


# In[57]:


model.add(layers.Dense(1, activation="sigmoid"))


# In[58]:


model.summary()


# In[63]:


model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])


# In[88]:



train_his = model.fit(x = train_feat, y=train_label,batch_size=30, validation_split = 0.1,epochs =30,  verbose=2)


# In[94]:


import matplotlib.pyplot as plt


# In[95]:


def show_train(train_his, train, val):
    plt.plot(train_his.history[train])
    plt.plot(train_his.history[val])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc = 'center right')
    plt.show
    


# In[96]:


show_train(train_his, 'acc', 'val_acc')


# In[ ]:




