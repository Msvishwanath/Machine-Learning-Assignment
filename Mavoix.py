#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split

file_path = (r'C:\Users\Admin\Downloads\mavoix_ml_sample_dataset.xlsx')
data = pd.read_excel(file_path)


# In[29]:


data.head()


# In[30]:


filter_data= data.dropna(axis=1)


# In[31]:


data_features = ['Deep Learning (out of 3)','Current Year Of Graduation']
data2 = ['Python (out of 3)']
x = filter_data[data_features]
y = filter_data[data2]


# In[32]:


filter_data.head()


# In[33]:


from sklearn.tree import DecisionTreeRegressor
ml_model1 = DecisionTreeRegressor(random_state=1)
z = ml_model1.fit(x,y)
print(z)


# In[34]:


from sklearn.metrics import mean_absolute_error
predicted_ml_alg = ml_model1.predict(x)
mean_absolute_error(y, predicted_ml_alg)


# In[35]:


from sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=0)
ml_model1 = DecisionTreeRegressor()
ml_model1.fit(train_x,train_y)
val_predict = ml_model1.predict(val_x)
print(val_predict)


# In[36]:


train_x.describe()


# In[40]:



import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)


plt.plot(x, x, label='linear')


plt.legend()


plt.show()


# In[ ]:




