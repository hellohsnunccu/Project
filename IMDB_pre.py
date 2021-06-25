#!/usr/bin/env python
# coding: utf-8

# In[2]:


import urllib.request as urq
import os
import tarfile as tar


# In[3]:


url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urq.urlretrieve(url,filepath)
    print(" ", result)


# In[4]:


#if not os.path.exists("data/aclImdb"):
#    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
#   result=tfile.extractall('data/')
if not os.path.exists("data/aclImdb"):
    tfile = tar.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result = tfile.extractall('data/')


# In[5]:


import tensorflow as tf
from tensorflow import keras


# In[6]:


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


# In[7]:


import re
def re_tag(text): #移除HTML tag
    re_tag = re.compile(r'<[^>]+>') #正規表示式
    return re_tag.sub('', text) #將text文字中，符合正規表示式的字，替換成空字串


# In[8]:


import os
def read_files(filetype): #train/test
    path = "data/aclImdb/"
    filelist = []
    
    pos_path = path + filetype + "/pos/"
    for f in os.listdir(pos_path):  #將pos_path all file add in filelist
        filelist += [pos_path + f] 
    neg_path = path + filetype + "/neg/" #將neg_path all file add in filelist
    for f in os.listdir(neg_path):
        filelist += [neg_path + f]
    
    print('read', filetype, 'files', len(filelist))
    
    all_label = ([1]*12500 + [0]*12500) # 1: pos 0:neg
    all_text=[]
    
    for t in filelist:
        with open(t, encoding='utf-8') as input:
            all_text += [re_tag(" ".join(input.readlines()))]
    
    return all_label, all_text


# In[9]:


train_label, train_text = read_files("train")


# In[10]:


test_label, test_text = read_files("test")


# In[11]:


train_text[0]


# In[12]:


test_text[0]


# In[13]:


train_label[0]


# In[15]:


test_label[0]


# In[17]:


token = Tokenizer(num_words =2000) #輸入參數num_word，設定長度2000
token.fit_on_texts(train_text)


# In[18]:


train_text[0]


# In[19]:


print(token.document_count)


# In[20]:


print(token.word_index) #依照次數將 字詞排名


# In[21]:


train_seq = token.texts_to_sequences(train_text)
test_seq = token.texts_to_sequences(test_text)


# In[22]:


print(train_text[0])
print(train_seq[0])


# In[24]:


f_train_seq = sequence.pad_sequences(train_seq, maxlen =100)
f_test_seq = sequence.pad_sequences(test_seq, maxlen =100)


# In[25]:


print("before pad:", len(train_seq[0]))
print(train_seq[0])


# In[26]:


print("after pad:", len(f_train_seq[0]))
print(f_train_seq[0])


# In[27]:


print("before pad:", len(train_seq[222]))
print(train_seq[222])


# In[28]:


print("after pad:", len(f_train_seq[222]))
print(f_train_seq[222])


# In[42]:


train = f_train_seq
test = f_test_seq


# In[43]:


train[0]


# In[48]:





# In[29]:


from tensorflow.keras.datasets import imdb
import numpy as np


# In[33]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten, Embedding


# In[34]:


model = Sequential()


# In[35]:


model.add(Embedding(output_dim=32,input_dim=2000,input_length=100))
model.add(Dropout(0.2))


# In[36]:


model.add(Flatten())


# In[37]:


model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.35))


# In[38]:


model.add(Dense(units=1, activation='sigmoid'))


# In[39]:


model.summary()


# In[41]:


model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics =['acc'])


# In[45]:


ver = np.random.rand(len(train)) <0.1
val = train[ver]


# In[50]:


train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

full_dataset = tf.data.TFRecordDataset(FLAGS.input_file)
full_dataset = full_dataset.shuffle()
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)


# In[49]:


train_his = model.fit(train,train_label,batch_size=100, epochs=20, verbose=2, validation_data = val)


# In[ ]:




