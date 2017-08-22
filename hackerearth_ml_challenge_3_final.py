
# coding: utf-8

# In[1]:


#The final solution for the hackerearth - machine learning challenge #3. Ranked 5th / 444.
#Author - Aakash Kerawat
import pandas as pd
import numpy as np

import xgboost

import sklearn.metrics as metrics

from sklearn.preprocessing import LabelEncoder


import warnings

warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('hackerearth_predict_click/train.csv', dtype={'siteid':'O', 'offerid':'O', 'category':'O', 'merchant':'O'}, parse_dates=['datetime'])
df_t = pd.read_csv('hackerearth_predict_click/test.csv', dtype={'siteid':'O', 'offerid':'O', 'category':'O', 'merchant':'O'}, parse_dates=['datetime'])
sub = pd.read_csv('hackerearth_predict_click/sample_submission.csv')


# In[744]:


def parse_dates(df):
    df['hour'] = df.datetime.dt.hour
    df['day_of_week'] = df.datetime.dt.weekday
#     df['week_of_year'] = df.datetime.dt.weekofyear
#     df['day_of_month'] = df.datetime.dt.day


# In[745]:


#date features
parse_dates(df)


# In[746]:


parse_dates(df_t)


# In[747]:


df_t['click'] = np.nan


# In[749]:


#sort by datetime
df.sort_values('datetime', inplace=True)


# In[750]:


df.reset_index(drop=True, inplace=True)


# In[751]:


#We have same browser with different names, lets makes them same.
def rename_browser(df):
    df.loc[df.browserid.isin(['Internet Explorer', 'InternetExplorer']), 'browserid'] = 'IE'
    df.loc[df.browserid.isin(['Mozilla Firefox', 'Mozilla']), 'browserid'] = 'Firefox'
    df.loc[df.browserid=='Google Chrome', 'browserid'] = 'Chrome'


# In[752]:


rename_browser(df)


# In[753]:


rename_browser(df_t)


# In[754]:


#Filling null values with 'Unknown'
df.browserid.fillna('Unknown', inplace=True)
df_t.browserid.fillna('Unknown', inplace=True)


# In[755]:


#Filling null values with 'Unknown'
df.devid.fillna('Unknown', inplace=True)
df_t.devid.fillna('Unknown', inplace=True)


# In[831]:


#feature - how many ads are launched at a time on under particular site, category, offer etc.
def get_onego_features(df):
    for col in ['siteid', 'category', 'offerid', 'merchant']:
        print(col)
        collection_temp = df.groupby([col, 'datetime'])['ID'].agg(['count']).reset_index()
        collection_temp.columns = [col, 'datetime', col+'_count_together']

        df = df.merge(collection_temp, 'left', [col, 'datetime'])
    return df


# In[836]:


df = get_onego_features(df)


# In[837]:


df_t = get_onego_features(df_t)


# In[840]:


#feature - number of unique merchants on a particular site
def get_unique_merchants(df, df_t):
    df_all = pd.concat([df, df_t]).reset_index(drop=True)
    temp = df_all.groupby(['siteid'])['merchant'].apply(lambda x: x.nunique()).reset_index()
    

    df = df.merge(temp, 'left', ['siteid'], suffixes=('', '_siteid_count'))
    df_t = df_t.merge(temp, 'left', ['siteid'], suffixes=('', '_siteid_count'))
    return df, df_t


# In[842]:


d = get_unique_merchants(df, df_t)


# In[843]:


df = d[0]
df_t = d[1]


# In[844]:


del d


# In[907]:


#feature - number of unique offers on a particular site
def get_unique_offers(df, df_t):
    df_all = pd.concat([df, df_t]).reset_index(drop=True)
    temp = df_all.groupby(['siteid'])['offerid'].apply(lambda x: x.nunique()).reset_index()
    

    df = df.merge(temp, 'left', ['siteid'], suffixes=('', '_siteid_count'))
    df_t = df_t.merge(temp, 'left', ['siteid'], suffixes=('', '_siteid_count'))
    return df, df_t


# In[908]:


d = get_unique_offers(df, df_t)


# In[909]:


df = d[0]
df_t = d[1]


# In[910]:


del d


# In[6]:


def get_availble_on(df, col):
    c = df.groupby('datetime')[col].nunique().reset_index()
    df = df.merge(c, 'left', 'datetime', suffixes=('', '_datetime_count'))
    return df


# In[7]:


#feature - on how many sites at a particular time that offer is available
df = get_availble_on(df, 'offerid')


# In[8]:


df_t = get_availble_on(df_t, 'offerid')


# In[18]:


#removing ID, datetime and target columns from predictors
to_drop = ['ID', 'datetime', 'click']


# In[19]:


predictors = df.columns.drop(to_drop)


# In[90]:


df_all = pd.concat([df, df_t])


# In[ ]:


#label encoding the string types
le = LabelEncoder()
for col in predictors:
    if df_all[col].dtype=='O' or df_all[col].dtype=='category':
        print(col)
        try:
            df_all[col] = le.fit_transform(df_all[col])
        except TypeError:
            df_all[col] = le.fit_transform(df_all[col].fillna('NAN'))


# In[94]:


df = df_all.loc[df_all.ID.isin(df.ID.unique())].reset_index(drop=True)


# In[95]:


df_t = df_all.loc[df_all.ID.isin(df_t.ID.unique())].reset_index(drop=True)


# In[ ]:


del df_all


# In[ ]:


xgb = xgboost.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.015)


# In[ ]:


#undersampling to a 1:1 ratio
df_train = pd.concat([df.loc[df.click==0].sample(437000), df.loc[df.click==1]]).reset_index(drop=True)


# In[ ]:


#training
xgb.fit(df_train[predictors], df_train['click'])


# In[ ]:


#predicting probabilities
preds = xgb.predict_proba(df_t[predictors])[:,1]
df_t['click'] = preds


# In[ ]:


df_t[['ID', 'click']].to_csv('hackerearth_predict_click/submission.csv', index=False)

