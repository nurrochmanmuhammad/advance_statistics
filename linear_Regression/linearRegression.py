# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:58:05 2020

@author: OMEN
"""

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

boston_dataset = load_boston()
## build a DataFrame
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

X = boston[['RM']]
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state=1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

model = LinearRegression()
model.fit(X_train,Y_train)

y_test_predicted=model.predict(X_test)

plt.scatter(X_test, Y_test,label='testing data', color='r');
plt.plot(X_test,y_test_predicted,label='prediction', linewidth=3)
plt.xlabel('RM'); plt.ylabel('MEDV')
plt.legend(loc='upper left')
plt.show()



residuals = Y_test - y_test_predicted

# plot the residuals
plt.scatter(X_test, residuals)
# plot a horizontal line at y = 0
plt.hlines(y = 0, 
  xmin = X_test.min(), xmax=X_test.max(),
  linestyle='--')
# set xlim
plt.xlim((4, 9))
plt.xlabel('RM'); plt.ylabel('residuals')
plt.show()

#manual MSE
MSE=(residuals**2).mean()
print('MSE=',MSE)
#MSE from lib
from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, y_test_predicted)


#Model Evaluation
#R-SQ
model.score(X_test, Y_test)

#Total Variance
tot_variance=((Y_test-Y_test.mean())**2).sum()

#Sum SQ of Residual
model_var=(residuals**2).sum()
print(model_var)

#proportion of total variation
prop_total_variance=1-model_var/tot_variance