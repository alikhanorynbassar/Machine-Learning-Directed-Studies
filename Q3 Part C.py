#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question 3 Part C

@author: alikhanorynbassar
"""

import numpy as np
# first neural network with keras tutorial
from numpy import loadtxt
import os.path
from keras.models import Sequential
from keras.layers import Dense
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

np.random.seed(42)


save_path = os.path.dirname(os.path.abspath(__file__)) 
file_name = 'features' 
completeName = os.path.join(save_path, file_name+".dat") 
X = np.loadtxt(completeName)
file_name = 'Class_Labels' 
completeName = os.path.join(save_path, file_name+".dat") 
y = np.loadtxt(completeName)
print ('feature_data.shape = ', X.shape)
print ('class_labels.shape = ', y.shape)
np.unique(y)
print ('np.unique(class_labels) = ', np.unique(y)) 

X.shape

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(9, activation='softmax'))

# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

# fit the keras model on the dataset
history = model.fit(X, y,validation_split=0.2, epochs=125, batch_size=128)

#estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)

import matplotlib.pyplot as plt

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

plt.savefig('./Models.png', dpi=300)
plt.show()

FileName = 'Output.dat'
Path1 = os.path.dirname(os.path.abspath(__file__))
NewPath = os.path.join(Path1, "Output.dat")
print(NewPath)
w = open(NewPath, "w")
w.write(str('Accuracy: %.2f' % (accuracy*100)))
w.close()