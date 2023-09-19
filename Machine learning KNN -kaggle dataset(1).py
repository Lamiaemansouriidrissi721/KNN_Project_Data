#!/usr/bin/env python
# coding: utf-8

# In[8]:


#import mylibraries
import pandas as pd
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[3]:


#load dataset
data=pd.read_csv(r"C:\Users\kumak\Desktop\selfstudy\Python\KNN_Project_Data")
data


# In[5]:


#Explore data
data.isnull().sum() #m ready to go to the auther stage 


# In[16]:


scaler = StandardScaler()


# In[18]:


scaler.fit(data.drop('TARGET CLASS',axis=1))


# In[22]:


scaled_features = scaler.transform(data.drop('TARGET CLASS',axis=1))


# In[33]:


data_new = pd.DataFrame(scaled_features)
data_new.head()


# In[36]:


# Assuming 'data' is your DataFrame with features and target
from sklearn.model_selection import train_test_split
x = data_new
y = data['TARGET CLASS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[37]:


#knn classifier train

#train test split
print("x_train shape:{}".format(x_train.shape))
print("y_train shape:{}".format(y_train.shape))
print("x_test shape:{}".format(x_test.shape))
print("y_test shape:{}".format(y_test.shape))


# In[38]:


#Create a KNN model instance with n_neighbors=1
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[39]:


#Training the data
knn.fit(x_train,y_train)


# In[49]:


#predict :Use the predict method to predict values using your KNN model and X_test.
predictions = knn.predict(x_test)

#Create a confusion matrix and classification report.
from sklearn.metrics import confusion_matrix,classification_report

p=confusion_matrix(y_test,predictions)
p


# In[43]:


#score
print(classification_report(y_test,predictions))


# In[48]:


import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(p, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[52]:


# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:




