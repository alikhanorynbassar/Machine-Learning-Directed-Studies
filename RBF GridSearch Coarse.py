#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:01:18 2022

@author: alikhanorynbassar
"""

#===================================================================== 
import sys
import os.path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as mpatches
from pylab import *
from sklearn import preprocessing
#===================================================================== 

#===================================================================== import data
save_path = os.path.dirname(os.path.abspath(__file__)) 
file_name = 'Features'
completeName = os.path.join(save_path, file_name+".dat") 
feature_data = np.loadtxt(completeName)
file_name = 'Class_Labels' 
completeName = os.path.join(save_path, file_name+".dat") 
class_labels = np.loadtxt(completeName)
print ('features_data.shape = ', feature_data.shape)
print ('class_labels.shape = ', class_labels.shape)
np.unique(class_labels)
print ('np.unique(class_labels) = ', np.unique(class_labels))   #Returns the sorted unique elements of an array.
#===================================================================== import data

#===================================================================== shuffle data
perm = np.random.permutation(class_labels.size)    # get the index numbers for random shuffle (permutation )
print ('perm = ', perm)

feature_data = feature_data[perm]                 # based on perm shuffle features
class_labels = class_labels[perm]                   # based on perm shuffle targets
#===================================================================== shuffle data

#===================================================================== scale
min_max_scaler = preprocessing.MinMaxScaler() # x = (x - x_min)/(x_max - x_min) 
feature_data = min_max_scaler.fit_transform(feature_data)
#===================================================================== scale

#========================================================================================== cross-validation
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.10, random_state=42) #(user-defined)

#========================================================================================== cross-validation

#========================================================================================== RBF
C_range = np.logspace(-2, 10, 13) #(user-defined) ######
gamma_range = np.logspace(-9, 3, 13)  #(user-defined)

parameter_grid = dict(gamma=gamma_range, C=C_range)  #(user-defined)
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=parameter_grid, cv=cv, return_train_score=True)  #(user-defined)
print ('parameter_grid = ', parameter_grid) #####
#========================================================================================== RBF
 
#========================================================================================== train
grid.fit(feature_data, class_labels)
#========================================================================================== train

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))  
#========================================================================================== output results
file_name = 'Grid_results_RBF' 
completeName = os.path.join(save_path, file_name+".dat") 
file2 = open(completeName, "w")
file2.write(" %s  %s \n" % ('parameter_grid = ', parameter_grid)) 
results = pd.DataFrame(grid.cv_results_) 
file2.write(" %s \n" % ('=========================================== =========================================== '))    
file2.write(" %s\n" % (results)) 
file2.write(" %s \n" % ('=========================================== =========================================== '))     
file2.write(" %s %s %s \n" % ('The best parameters are %s with a score of = ', grid.best_params_, grid.best_score_))
file2.write(" %s \n" % ('=========================================== =========================================== '))     
file2.write(" %s %s\n" % ('scores = ', scores))
file2.close()
#========================================================================================== output results

#========================================================================================== plot matlab
#(user-defined parameters for plot)
from matplotlib.colors import Normalize
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

import matplotlib.pyplot as plt     
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))  
                                                                                        
print ('scores =',  scores)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
              
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.1, right=0.95, bottom=0.25, top=0.92)

plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
        #norm=MidpointNormalize(vmin=0.7, midpoint=0.9))
        norm=MidpointNormalize(vmin=0.7, midpoint=0.84))           
           
# Set the tick labels font
ax = plt.subplot() # Defines ax variable by creating an empty plot
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    #label.set_fontsize(90)
    label.set_fontsize(20)
                      
'''
plt.xlabel('gamma', fontsize=90)
plt.ylabel('C', fontsize=90)
'''
plt.xlabel('gamma', fontsize=20)
plt.ylabel('C', fontsize=20)

from pylab import *
#plt.colorbar()
cb0 = colorbar()
'''
cb0.ax.tick_params(labelsize=80) # Font size of the labels
cb0.set_label(label='Accuracy',weight='bold',fontsize=90)
'''
cb0.ax.tick_params(labelsize=20) # Font size of the labels
cb0.set_label(label='Accuracy',weight='bold',fontsize=20)

#plt.xticks(np.arange(len(gamma_range)), log10(gamma_range), rotation=45)
#plt.yticks(np.arange(len(C_range)), log10(C_range))
plt.xticks(np.arange(len(gamma_range)), (gamma_range), rotation=45)
plt.yticks(np.arange(len(C_range)), (C_range))

#plt.title('Validation accuracy', fontsize=90)
plt.title('Validation accuracy', fontsize=10)

plt.savefig('./RBF_Coarse_G_S.png', dpi=300)
plt.show()
#========================================================================================== plot matlab

#======================================================================== plot in Tecplot
#sscores = np.zeros(ss)
file_name = 'Scores_RBF' 
completeName = os.path.join(save_path, file_name+".dat") 
file3 = open(completeName, "w")
TITLE1 = "Scores"
TITLE2 = 'VARIABLES = "C", "Gamma", "Accuracy"'
TITLE3 = 'ZONE T="Zone 1", I= ' + str(C_range.size) + ', J= ' + str(gamma_range.size) + ', F=POINT'    
file3.write("%s\n" % (TITLE1))
file3.write("%s\n" % (TITLE2))
file3.write("%s\n" % (TITLE3))


for i in range(C_range.size):
    for j in range(gamma_range.size):
        file3.write(" %16.8f %16.8f %18.10f \n" % (math.log10(C_range[i]),math.log10(gamma_range[j]) , scores[i,j]))
        #print (i,j, 'score = ', scores[i,j])
file3.close()                
#======================================================================== plot in Tecplot
sys.exit('Program stopped here - end of program') 

# 
