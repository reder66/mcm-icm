
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

data_az = pd.read_csv('...\ProblemCData_AZ.csv')
data_ca = pd.read_csv('...\ProblemCData_CA.csv')
data_nm = pd.read_csv('...\ProblemCData_NM.csv')
data_tx = pd.read_csv('...\ProblemCData_TX.csv')

feature1 = ['CLPRB','CLTCB','FFTCB','MGTCB','NGMPB','NGTCB','P1TCB','PAPRB','REPRB','RETCB','RFTCB']
feature2 = ['HYTCB','GETCB','SOTCB','WYTCB','BMTCB','WWTCB']
#Coal production.
#Coal total consumption.
#Fossil fuelS, total consumption.
#Geothermal energy total consumption.
#Motor gasoline total consumption.
#Natural gas marketed production.
#Natural gas total consumption
#Asphalt and road oil
#Crude oil production
#Renewable energy production.
#Renewable energy total consumption.
#Residual fuel oil total consumption.
def create_x(data,k):
    X = pd.DataFrame(index = range(1960,2010))
    if k==1:
        for i in feature1:
            X[i] = data[data['MSN']==i]['Data'].values
        y = data[data['MSN']=='TETPB']['Data'].values #TETPB is total consumption per capita
    elif k==2:
        for i in feature2:
            X[i] = data[data['MSN']==i]['Data'].values
        y = data[data['MSN']=='RETCB']['Data'].values
        
    return X,y



# In[2]:

x1_az,y1_az = create_x(data_az,1)
x1_ca,y1_ca = create_x(data_ca,1)
x1_nm,y1_nm = create_x(data_nm,1)
x1_tx,y1_tx = create_x(data_tx,1)

x2_az,y2_az = create_x(data_az,2)
x2_ca,y2_ca = create_x(data_ca,2)
x2_nm,y2_nm = create_x(data_nm,2)
x2_tx,y2_tx = create_x(data_tx,2)


# In[3]:

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import operator

def xgbTraining(X,y):
    X_ = (X-X.mean())/X.std()
    y_ = (y-y.mean())/y.std()
    
    dtrain=xgb.DMatrix(X_,y_)
    dtest=xgb.DMatrix(X_)
    param = {}
    param['eta'] = 0.05
    param['max_depth'] = 4
    param['mmin_child_weight'] = 3
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['silent'] = 1

    alg = xgb.train(param,dtrain,5000)
    Y = alg.predict(dtest)
    Y = Y*y.std() + y.mean()
    
    importance = alg.get_fscore()  
    importance = sorted(importance.items(), key=operator.itemgetter(1))  
  
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
    df['fscore'] = df['fscore'] / df['fscore'].sum()  
    return Y,df


# In[75]:

Y1_az,df1_az = xgbTraining(x1_az,y1_az)
Y1_ca,df1_ca = xgbTraining(x1_ca,y1_ca)
Y1_nm,df1_nm = xgbTraining(x1_nm,y1_nm)
Y1_tx,df1_tx = xgbTraining(x1_tx,y1_tx)

Y2_az,df2_az = xgbTraining(x2_az,y2_az)
Y2_ca,df2_ca = xgbTraining(x2_ca,y2_ca)
Y2_nm,df2_nm = xgbTraining(x2_nm,y2_nm)
Y2_tx,df2_tx = xgbTraining(x2_tx,y2_tx)


# In[76]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def plot(df_az,df_ca,df_nm,df_tx):
    plt.figure(figsize=(20,20))
    plt.subplot(221)
    plt.barh(range(df_az.shape[0]),df_az['fscore'],facecolor='#B8860B')
    plt.yticks(range(df_az.shape[0]),df_az['feature'],fontsize=18)
    plt.title('AZ feature importance',fontsize=21)
    plt.subplot(222)
    plt.barh(range(df_ca.shape[0]),df_ca['fscore'],facecolor='#8B0000')
    plt.yticks(range(df_ca.shape[0]),df_ca['feature'],fontsize=18)
    plt.title('CA feature importance',fontsize=21)
    plt.subplot(223)
    plt.barh(range(df_nm.shape[0]),df_nm['fscore'],facecolor='#8FBC8F')
    plt.yticks(range(df_nm.shape[0]),df_nm['feature'],fontsize=18)
    plt.title('NM feature importance',fontsize=21)
    plt.subplot(224)
    plt.barh(range(df_tx.shape[0]),df_tx['fscore'])
    plt.yticks(range(df_tx.shape[0]),df_tx['feature'],fontsize=18)
    plt.title('TX feature importance',fontsize=21)
    plt.show()


# In[77]:

plot(df1_az,df1_ca,df1_nm,df1_tx)


# In[78]:

plot(df2_az,df2_ca,df2_nm,df2_tx)


# In[79]:

def cal_w(df,X):
    w = df['fscore'].values
    W = np.dot(X,w)
    return W

w1_az = cal_w(df1_az,x1_az)
w1_ca = cal_w(df1_ca,x1_ca)
w1_nm = cal_w(df1_nm,x1_nm)
w1_tx = cal_w(df1_tx,x1_tx)

plt.figure(figsize=(10,8))
plt.plot(x1_az.index,w1_az,label = 'AZ')
plt.plot(x1_ca.index,w1_ca,label = 'CA')
plt.plot(x1_nm.index,w1_nm,label = 'NM')
plt.plot(x1_tx.index,w1_tx,label = 'TX')

plt.title('W = Xw , represent 4 states energy use',fontsize=21 )
plt.legend(loc='best')
plt.xlabel('year',fontsize=18)
plt.ylabel('W_value',fontsize=18)
plt.show()


# In[80]:

w2_az = cal_w(df2_az,x2_az)
w2_ca = cal_w(df2_ca,x2_ca)
w2_nm = cal_w(df2_nm,x2_nm)
w2_tx = cal_w(df2_tx,x2_tx)

plt.figure(figsize=(10,8))
plt.plot(x2_az.index,w2_az,label = 'AZ')
plt.plot(x2_ca.index,w2_ca,label = 'CA')
plt.plot(x2_nm.index,w2_nm,label = 'NM')
plt.plot(x2_tx.index,w2_tx,label = 'TX')

plt.title('W = Xw , represent 4 states renewable energy use',fontsize=21 )
plt.legend(loc='best')
plt.xlabel('year',fontsize=18)
plt.ylabel('W_value',fontsize=18)
plt.show()


# In[73]:

print('In 2009 these four states renewable energy use:')
print('Arizona:%d \nCalifornia:%d \nNew Mexico:%d \nTexas:%d'%(        w2_az[-1],w2_ca[-1],w2_nm[-1],w2_tx[-1]))

