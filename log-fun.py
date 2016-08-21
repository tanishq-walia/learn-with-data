# -*- coding: utf-8 -*-
"""
Created on Tue jul  9 10:50:51 2016
#Tikam Singh

@author: hduser
"""
import numpy as np
# Math
import math
# Plot imports
import matplotlib.pyplot as plt
# Machine Learning Imports
# Logistic Function
def logistic(t):
    return 1.0 / (1 + math.exp((-1.0)*t) )

# Set t from -6 to 6 ( 500 elements, linearly spaced)
t = np.linspace(-5,5,500)

# Set up y values (using list comprehension)
y = np.array([logistic(ele) for ele in t])

# Plot
plt.plot(t,y)
plt.title(' Logistic Function ')

