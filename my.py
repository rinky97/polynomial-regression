# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:41:41 2017

@author: Rinky
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#poly regression
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=2)
X_poly =  pol_reg.fit_transform(X)
pol_reg.fit(X_poly,y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(X_poly),color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
