# -*- coding: utf-8 -*-
"""
Created on Tue jul  9 10:50:51 2016
#Tikam Singh

@author: hduser
"""
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import sklearn
from sklearn import datasets
boston=datasets.load_boston()
print boston.DESCR
#histogram for prices 
plt.hist(boston.target,bins=50)
plt.xlabel("prices in dollars $1000's")
plt.ylabel('Number of houses')
#scatter plot prices and no of rooms 
plt.scatter(boston.data[:,5],boston.target)
plt.ylabel("prices in dollars $1000's")
plt.xlabel('Number of rooms')
#use pandas to form dataframe of boston data
boston_df=pd.DataFrame(boston.data)
boston_df.columns=boston.feature_names
boston_df.head()
boston_df['Price']=boston.target

sns.lmplot('RM','Price',data=boston_df)

#using np for univivariate regression
#set x as median value
X=boston_df.RM
#make x two dimensional using np
X=np.vstack(boston_df.RM)
#set up y as target prices of house
Y=boston_df.Price
X=np.array( [ [value,1] for value in X ] )
m,c=np.linalg.lstsq(X,Y)[0]
# First the original points, Price vs Avg Number of Rooms
plt.plot(boston_df.RM,boston_df.Price,'o')
# Next the best fit line
x= boston_df.RM
plt.plot(x, m*x + c,'r',label='Best Fit Line')
result=np.linalg.lstsq(X,Y)
error=result[1]
rmse=np.sqrt(error/len(X))
print(' the root mean sq error %.2f'%rmse)
#rmse =6.60 mean we say that the house price lie with in 2sigma with singnificance
#level 95%


