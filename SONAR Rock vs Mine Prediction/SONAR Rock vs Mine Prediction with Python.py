#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


sonar_data =pd.read_csv(r'C:\Users\omkar\OneDrive\Desktop\dim date\sonar data\Copy of sonar data.csv',header=None)


# In[3]:


sonar_data.head()


# In[4]:


sonar_data.shape


# In[5]:


sonar_data.describe()


# In[6]:


sonar_data[60].value_counts()


# In[7]:


sonar_data.groupby(60).mean()


# In[8]:


x= sonar_data.drop(columns=60,axis=1)
y= sonar_data[60]


# In[9]:


print(x)
print(y)


# In[10]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size =0.1,stratify = y,random_state=1)


# In[11]:


print(x.shape,x_train.shape,x_test.shape)


# In[12]:


print(x_train)
print(y_train)


# In[13]:


model = LogisticRegression()


# In[14]:


model.fit(x_train, y_train)


# In[15]:


x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)


# In[16]:


print('Accuracy on traning data:',training_data_accuracy)


# In[17]:


x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[18]:


print('Accuracy on traning data:',test_data_accuracy)


# In[19]:


input_data =(0.0100,0.0171,0.0623,0.0205,0.0205,0.0368,0.1098,0.1276,0.0598,0.1264,0.0881,0.1992,0.0184,0.2261,0.1729,0.2131,0.0693,0.2281,0.4060,0.3973,0.2741,0.3690,0.5556,0.4846,0.3140,0.5334,0.5256,0.2520,0.2090,0.3559,0.6260,0.7340,0.6120,0.3497,0.3953,0.3012,0.5408,0.8814,0.9857,0.9167,0.6121,0.5006,0.3210,0.3202,0.4295,0.3654,0.2655,0.1576,0.0681,0.0294,0.0241,0.0121,0.0036,0.0150,0.0085,0.0073,0.0050,0.0044,0.0040,0.0117
            )
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# In[20]:


input_data= (0.0192,0.0607,0.0378,0.0774,0.1388,0.0809,0.0568,0.0219,0.1037,0.1186,0.1237,0.1601,0.3520,0.4479,0.3769,0.5761,0.6426,0.6790,0.7157,0.5466,0.5399,0.6362,0.7849,0.7756,0.5780,0.4862,0.4181,0.2457,0.0716,0.0613,0.1816,0.4493,0.5976,0.3785,0.2495,0.5771,0.8852,0.8409,0.3570,0.3133,0.6096,0.6378,0.2709,0.1419,0.1260,0.1288,0.0790,0.0829,0.0520,0.0216,0.0360,0.0331,0.0131,0.0120,0.0108,0.0024,0.0045,0.0037,0.0112,0.0075,)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# In[ ]:




