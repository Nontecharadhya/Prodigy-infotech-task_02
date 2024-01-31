#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[24]:


os.getcwd()


# In[25]:


os.chdir('C:\\Users\\Abhi\\downloads')


# # Data Loading & Processing

# In[26]:


train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')


# In[27]:


train.head()


# In[13]:


test.head()


# In[14]:


train.describe(include="all")


# In[15]:


train.count()


# # DATA CLEANING

# In[23]:


# CHECK THE MISSING VALUES IN THE DATASET
print(train.isnull().sum())


# DROP UNNECESSERY COLUMNS THAT WONT CONTRIBUTE TO THE ANALYSIS
train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)


# # EDA

# ## Exploring the relationships between variables and identify and trends in the data

# In[15]:


# Exploring the age distribution of passengers
plt.hist(train['Age'],bins=25,color='yellow')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age distribution')
plt.show()


# In[35]:


# PLOTTING HISTPLOT
train.hist(figsize=(10,10))
plt.show()


# In[36]:


#PLOTTING PAIRPLOT
sns.pairplot(train,hue='Survived',palette='PuRd_r')
plt.show()


# In[17]:


train.groupby('Survived').get_group(1)


# In[18]:


male_index =len(train[train['Sex']=='male'])
print("no of male in Titanic:",male_index)


# In[19]:


fem_index =len(train[train['Sex']=='female'])
print("no of female in Titanic:",fem_index)


# In[33]:


# plotting 
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
gender =['male','female']
index=[577,314]
ax.bar(gender,index)
plt.xlabel('gender')
plt.ylabel('no of people onboarding ship')
plt.show()


# In[7]:


alive =len(train[train['Survived'] == 1])
dead =len(train[train['Survived'] ==0])


# In[12]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
l=['C=Cherbourg','Q=Queenstown','S=Southampton']
s=[0.553571,0.389610,0.336957]
ax.pie(s,labels=l,autopct='%0.1f%%')
plt.show()


# In[ ]:




