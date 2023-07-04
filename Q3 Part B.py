#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question 3  Part B 
@author: alikhan orynbassar 
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os.path
import sys

X = np.loadtxt('features.dat', unpack = True)
X = X.reshape((1000,20))
y = np.loadtxt('Class_labels.dat', unpack = True)
#please check the parameters if they are inline with assignment 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

params = {'criterion':['gini', 'entropy'],'max_depth': [1,5,7,10,50,100],'max_leaf_nodes': list(range(2, 5)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=5)
grid_search_cv.fit(X_train, y_train)

best_model_parameters = grid_search_cv.best_params_
print(best_model_parameters)

# **best_model_parameters: **kwargs send a dictionary into a function,
# here instead of sending the parameters of the function one by one we directly send the best model dictionary
clf0 = DecisionTreeClassifier(**best_model_parameters)
#fit the data to the model
clf0.fit(X_train, y_train)
#model's performance on the test set
print("Test Accuracy of the decision tree without Dimensionality reduction :", clf0.score(X_test,y_test))

pca = PCA(n_components=15)
pca.fit(X_train)

X_train_reduced = pca.transform(X_train)
X_test_reduced =  pca.transform(X_test)

params = {'criterion':['gini', 'entropy'],'max_depth': [1,5,7,10,50,100],'max_leaf_nodes': list(range(2, 5)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=5)
grid_search_cv.fit(X_train_reduced, y_train)

best_model_parameters = grid_search_cv.best_params_
print(best_model_parameters)

# **best_model_parameters: **kwargs send a dictionary into a function,
# here instead of sending the parameters of the function one by one we directly send the best model dictionary
clf1 = DecisionTreeClassifier(**best_model_parameters)
#fit the data to the model
clf1.fit(X_train_reduced, y_train)
#model's performance on the test set
print("Test Accuracy of the decision tree with Dimensionality reduction :", clf1.score(X_test_reduced,y_test))

sys.exit('Program stopped here - end of program') 


