import numpy as np
import csv as csv

import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
titanic_df=pd.read_csv("/home/hduser/Downloads/train.csv")

sns.factorplot('Sex', data=titanic_df, kind="count")
# For .read_csv, always use header=0 when you know row 0 is the header row
#df = pd.read_csv("/home/hduser/Downloads/train.csv", header=0)
sns.factorplot('Sex',data=titanic_df,kind="count",hue="Pclass")
def male_female_child(passenger):
    age,sex=passenger
    if age<16:
        return 'child'
    else:
        return sex
titanic_df['person']=titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
sns.factorplot('Pclass',data=titanic_df,kind="count",hue='person')
titanic_df['person'].value_counts()
titanic_df['Age'].hist(bins=20)
fig=sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
     
fig=sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()



fig=sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

#drop null values)
deck=titanic_df['Cabin'].dropna()
levels=[]
for level in deck:
    levels.append(level[0])
cabin_df=DataFrame(levels)
cabin_df=cabin_df[cabin_df.Cabin !='T']
cabin_df.columns=['Cabin']
sns.factorplot("Cabin",data=cabin_df,palette='summer',kind='count')
sns.factorplot('Embarked',data=titanic_df,hue='Pclass',x_order=['C','Q','S'],kind='count')


titanic_df['Alone']=titanic_df.SibSp + titanic_df.Parch
titanic_df['Alone'].loc[titanic_df['Alone']>0]='with family'
titanic_df['Alone'].loc[titanic_df['Alone']==0]='Alone'
sns.factorplot('Alone',data=titanic_df,palette='Blues',kind='count')

titanic_df['Survivar']=titanic_df.Survived.map({0:'no',1:'yes'})
sns.factorplot('Survivar',data=titanic_df,palette='Set1',kind='count')

sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)


generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)
sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)

