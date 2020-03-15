#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()


# In[3]:


dir(digits)


# In[11]:


for i in range(4):
    plt.matshow(digits.images[i])


# In[17]:


digits.data[:1]


# In[19]:


df = pd.DataFrame(digits.data)
df.head()


# In[25]:


df['target'] = digits.target
df.head(3)


# In[26]:


from sklearn.model_selection import train_test_split


# In[71]:


x = df.drop('target',axis = 'columns')
x[:5]


# In[73]:


y = df['target']
y[:5]


# In[74]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[75]:


len(x_train)


# In[76]:


len(x_test)


# In[83]:


from sklearn.ensemble import RandomForestClassifier


# In[91]:


model = RandomForestClassifier(n_estimators=20)


# In[92]:


model.fit(x_train,y_train)


# In[90]:


model.score(x_test,y_test)


# In[94]:


y_predicted = model.predict(x_test)


# In[96]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[97]:


import seaborn as sn


# In[99]:


sn.heatmap(cm,annot=True)
plt.xlabel('Prediction')
plt.ylabel('Truth')


# In[ ]:




