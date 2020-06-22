# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:20:48 2020

@author: OMEN
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

data = pd.read_csv('D:\datasets_180_408_data.csv')

#Get Target data 
y = data['diagnosis']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['id','diagnosis','Unnamed: 32'], axis = 1)

logModel = LogisticRegression()
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['newton-cg'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

clf = GridSearchCV(logModel, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)

best_clf = clf.fit(X,y)
best_clf.pre
best_clf.best_estimator_

print (f'Accuracy - : {best_clf.score(X,y):.3f}')
print (f'Accuracy - : {logModel.score(X,y):.3f}')