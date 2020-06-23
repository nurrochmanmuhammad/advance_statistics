# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:45:01 2020

@author: OMEN
"""

import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

Y = boston['MEDV']
X2 = boston.loc[:, boston.columns != 'MEDV']

X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y, 
                                                    test_size = 0.3, 
                                                    random_state=1)
model2 = LinearRegression()
model2.fit(X2_train, Y_train)
y_test_predicted2 = model2.predict(X2_test)
print(y_test_predicted2)

from sklearn.metrics import mean_squared_error
print('MSE=',mean_squared_error(Y_test, y_test_predicted2))
print('Model Score=',model2.score(X2_test,Y_test))

plt.style.use('classic')
boston.boxplot(column='RM');