# For PCCP submission: Deciphering Tip-Enhanced Raman Imaging of Carbon Nanotubes with Deep Learning Neural Networks
# Usant Kajendirarajah, University of Western Ontario, 2020.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical


# In[2]:


dataframe = pd.read_csv('trainingdata.csv')
dataset = dataframe.values

# dataframe.describe(include="all")
# sns.pairplot(dataframe, hue='Label')

sns = sns.heatmap(dataframe.corr(), annot=True)

df3 = dataframe['Label'].value_counts().reset_index()
df3.columns = ['Label', 'count']
print (df3)


# In[3]:


factor = pd.factorize(dataframe['Label'])
dataframe['Label'] = factor[0]
definitions = factor[1]
print(definitions)


# In[4]:


X = dataset[:,0:7].astype(float)
print(X)
Y = dataset[:,7]
print(Y)


# In[5]:


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)


# In[6]:


# splitting into testing and validation set
train_x, test_x, train_y, test_y = model_selection.train_test_split(X,dummy_y,test_size = 0.3, random_state = 0)


# In[7]:


# architecture and topology of neural network model
model = Sequential()
model.add(Dense(20, input_dim=7, kernel_initializer='random_normal', activation='relu'))
model.add(Dense(20, kernel_initializer='random_normal', activation='relu'))
model.add(Dense(2, kernel_initializer='random_normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, epochs = 100, batch_size = 500)

scores = model.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[8]:


# serialize model to JSON
model_json = model.to_json()
with open("modelG.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelG_json.h5")
print("Saved model to disk")


# In[ ]:


# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model_yaml.h5")
print("Saved model to disk")

