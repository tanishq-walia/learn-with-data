# -*- coding: utf-8 -*-
"""
Created on Tue jul  9 10:50:51 2016
#Tikam Singh

@author: hduser
"""
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
X=iris.data
iris_data=DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
Y=iris.target
iris_target=DataFrame(Y,columns=['species'])
#convert the class labels into names 
def flower(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Veriscolour'
    else:
        return 'Virginica'

iris_target = iris_target['species'].apply(flower)
# combine the data set 
iris = pd.concat([iris_data,iris_target],axis=1)
# use pair flot to visualize and analysis the pattern
sns.pairplot(iris,hue='species',size=2)
#use histogram to see thr separtion btw the trhee class

sns.factorplot('Petal Length',data=iris,hue='species',size=10,kind='count')

logreg = LogisticRegression()
logreg.fit(X,Y)
# Split the data into training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5,random_state=3)

# Train the model with the training set
logreg.fit(X_train, Y_train)
#predict the label class
Y_pred = logreg.predict(X_test)
Accuracy= metrics.accuracy_score(Y_test,Y_pred)
#------------------------------------------------------------------------
# Test k values 1 through 20
k_range = range(1, 21)

# Set an empty list
accuracy = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(Y_test, Y_pred))
    k= accuracy.index(max(accuracy))
  #--------------------------------------------------------------------------


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)
import matplotlib.pyplot as plt
plt.plot(k_range, accuracy)
plt.xlabel('K value for for kNN')
plt.ylabel('Testing Accuracy')


print 'accuracy before we find the no of k should we take' ,Accuracy
print'accuracy after k test ', metrics.accuracy_score(Y_test,Y_pred)
