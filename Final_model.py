#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[9]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# # Import the test & training data
# 
# After performing the KNN-imputation(Done in R)

# In[2]:


train=pd.read_csv("train_complete.csv")
test=pd.read_csv("test_complete.csv")


# # Further Data Pre-processing

# In[3]:


application_key=test['application_key']


# In[4]:


train=train.drop(['Unnamed: 0','application_key'],axis=1)
test=test.drop(['Unnamed: 0','application_key'],axis=1)


# Encoding the **'mvar47'** column

# In[5]:


le=LabelEncoder()
train['mvar47']=le.fit_transform(train['mvar47'])
test['mvar47']=le.fit_transform(test['mvar47'])


# In[ ]:


test=test.drop(['default_ind'],axis=1)


# ### Separate out the target variable and feature from training set

# In[6]:


train_label=train['default_ind']
train_x=train.drop(['default_ind'],axis=1)


# # SMOTE

# In[7]:


sm = SMOTE(random_state=42)
train_x,train_label = sm.fit_resample(train_x, train_label)


# In[8]:


train_x


# # Model Building (Random Forest & XGBoost)

# ## Random Forest

# In[183]:


rf=RandomForestClassifier()


# ## Parameter Grid

# In[180]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 11)]
max_depth = [int(x) for x in np.linspace(4, 20, num = 9)]


param_grid_rf = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               }


# ## Grid Search CV

# In[ ]:


grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_rf, 
                          cv = 5, n_jobs = -1, verbose = 2)


# In[ ]:


model=grid_search.fit(train_x, train_label)


# **Best Parameters obtained and score**

# In[ ]:


model.best_params_


# In[ ]:


model.best_score_


# In[ ]:


best_model_rf=model.best_estimator_


# In[ ]:


best_model_rf


# ## Prediction on test set

# In[153]:


predictions_rf=best_model_rf.predict(test)


# ## Dataframe for submission

# In[154]:


df1=pd.DataFrame(predictions_rf)
df2=pd.DataFrame(application_key,dtype=np.int64)


# In[155]:


frames_rf = [df2, df1]
result_rf = pd.concat(frames_rf,axis=1)


# In[157]:


result_rf.to_csv('Data_poltergeists_rf.csv')


# # XGBoost

# In[ ]:


xg=XGBClassifier()


# ## Parameter Grid

# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 40, stop = 160, num = 7)]
max_depth = [int(x) for x in np.linspace(4,12, num =5)]
learning_rate=[0.05,0.3]

param_grid_xg = {'n_estimators': n_estimators,
               'max_depth': max_depth,
              'learning_rate':learning_rate,
             }


# ## Grid Search CV

# In[ ]:


grid_search = GridSearchCV(estimator = xg, param_grid = param_grid_xg, 
                          cv = 5, n_jobs = -1, verbose = 2)


# In[ ]:


model_xg=grid_search.fit(X_train, y_train)


# ## Best parameter and score

# In[ ]:


model_xg.best_params_


# In[ ]:


model_xg.best_score_


# In[ ]:


xg_best=model_xg.best_estimator_


# ## Prediction

# In[ ]:


prediction_xg=xg_best.predict(test)


# In[ ]:


df3=pd.DataFrame(prediction_xg)
df4=pd.DataFrame(application_key,dtype=np.int64)


# In[ ]:


frames_xg = [df4, df3]
result_xg = pd.concat(frames_xg,axis=1)


# In[ ]:


result_xg.to_csv('Data_poltergeists_xg.csv')

