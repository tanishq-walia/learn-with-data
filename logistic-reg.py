# -*- coding: utf-8 -*-
"""
Created on Tue jul  9 10:50:51 2016
#Tikam Singh

@author: hduser
"""
from pandas import DataFrame
import numpy as np
import pandas as pd
# Plot imports
import seaborn as sns
sns.set_style('whitegrid')
# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
# Dataset Import
import statsmodels.api as sm
df = sm.datasets.fair.load_pandas().data
df.head()
def affair_fun(x):
    if x != 0:
        return 1
    else:
        return 0
df['Had Affair']=df['affairs'].apply(affair_fun)
df.groupby('Had Affair').mean()
sns.factorplot('age',data=df,hue='Had Affair',palette='coolwarm',kind='count')


sns.factorplot('yrs_married',data=df,hue='Had Affair',palette='coolwarm',kind='count')

sns.factorplot('educ',data=df,hue='Had Affair',palette='coolwarm',kind='count')

sns.factorplot('children',data=df,hue='Had Affair',palette='coolwarm',kind='count')
#prepare data by classifiyng dummy variables
dummy_occ=pd.get_dummies(df['occupation'])
dummy_hus_occ=pd.get_dummies(df['occupation_husb'])
dummy_occ.head()
dummy_occ.columns=['occ1','occ2','occ3','occ4','occ5','occ6']
dummy_hus_occ.columns=['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']


X=df.drop(['occupation','occupation_husb','Had Affair'],axis=1)

#concate the prepare data 
dummies=pd.concat([dummy_occ,dummy_hus_occ],axis=1)
X=pd.concat([X,dummies],axis=1)
# Dropping one column of each dummy variable set to avoid multicollinearity
X = X.drop('occ1',axis=1)
X = X.drop('hocc1',axis=1)

# Drop affairs column so Y target makes sense
X = X.drop('affairs',axis=1)
#taking y as target level

Y=df['Had Affair']

#convert Y as array of data
Y = np.ravel(Y)
#fit the curve usin X and Y
log_model = LogisticRegression()

log_model.fit(X,Y)
#get the score of model
log_model.score(X,Y)
#check the mean
Y.mean()
coeff_df = DataFrame(zip(X.columns,np.transpose(log_model.coef_)))
coeff_df
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
log_model2 = LogisticRegression()

log_model2.fit(X_train,Y_train)
class_predict = log_model2.predict(X_test)
print metrics.accuracy_score(Y_test,class_predict)














