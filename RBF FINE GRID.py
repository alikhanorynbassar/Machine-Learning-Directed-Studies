#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:17:23 2022

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
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as mpatches
from pylab import *
from sklearn import preprocessing
#===================================================================== 
np.random.seed(42)
#===================================================================== import data
save_path = os.path.dirname(os.path.abspath(__file__)) 
file_name = 'features' 
completeName = os.path.join(save_path, file_name+".dat") 
feature_data = np.loadtxt(completeName)
file_name = 'Class_Labels' 
completeName = os.path.join(save_path, file_name+".dat") 
class_labels = np.loadtxt(completeName)
print ('feature_data.shape = ', feature_data.shape)
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

#===================================================================== 
X_feature = feature_data
y_class = class_labels
#===================================================================== 
#========================================================== RBF parameters
# Number of random trials
NUM_TRIALS = 20 #(user-defined)
splits_num = 10 #(user-defined)
# We will use a Support Vector Classifier with "rbf" kernel
svr = svm.SVC(kernel='rbf')
#p_grid = {"C": [10000, 90000, 100000, 110000, 1000000],
#         "gamma": [1.0, 10, 50, 90, 100, 110, 150]}                             
      
C_size = 4  #(user-defined) 
C_range = np.zeros(C_size)

C_ini = 100.0 #(user-defined)
C_ini = 10.0 #(user-defined)
#==========#(user-defined)
C_range[0] = C_ini
for i in range(1, int(C_size/2+1)):
    C_range[i] = C_range[i-1] + C_range[0] 
    
for i in range(int(C_size/2+1), C_size):
    C_range[i] = C_range[i-1] + C_range[int(C_size/2)] 
#==========#(user-defined)
    
g_size = 5  #(user-defined)
g_range = np.zeros(g_size)
g_ini = 2.0e-2  #(user-defined)

#==========#(user-defined)
g_range[0] = g_ini
for i in range(1, int(g_size/2+1)):
    g_range[i] = g_range[i-1] + g_range[0] 
    
for i in range(int(g_size/2+1), g_size):
    g_range[i] = g_range[i-1] + g_range[int(C_size/2)] 
#==========#(user-defined)    
#C_range = np.linspace(50, 200, num=5,dtype= int)
#g_range = np.linspace(50, 200, num=5,dtype= int)
C_range =  [50 , 75, 100, 125, 150, 175, 200]
g_range =  [ 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 10]
p_grid = {"C": C_range, "gamma": g_range}      
print ('C_range = ', C_range)
print ('g_range = ', g_range)
p_size = C_size * g_size 
p_grid = {"C": C_range, "gamma": g_range}                           
#========================================================== RBF parameters

#========================================================= grid search
# Arrays to store scores
model_scores = np.zeros(NUM_TRIALS)
model_index = np.zeros(NUM_TRIALS)
model_std = np.zeros(NUM_TRIALS)
model_votes = np.zeros(p_size)

file_name = 'RANDOM TRIALS' 
completeName = os.path.join(save_path, file_name+".dat") 
file0 = open(completeName, "w")

file_name = 'Best_parameters_RBF' 
completeName = os.path.join(save_path, file_name+".dat") 
file1 = open(completeName, "w")

file_name = 'Grid_results_RBF' 
completeName = os.path.join(save_path, file_name+".dat") 
file2 = open(completeName, "w")
file2.write(" %s  %5i  %s  %5i \n" % ('NUM_TRIALS = ', NUM_TRIALS, ' , splits_num = ', splits_num))  
file2.write(" %s  %s \n" % ('p_grid = ', p_grid))  

models_C_g = []
models_C_g2 = []

# Loop for each trial
#for i in range(NUM_TRIALS):
for i in range(NUM_TRIALS):             # different random splits
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
    inner_cv = KFold(n_splits=splits_num, shuffle=True, random_state=i) #(user-defined)
    

    clf = GridSearchCV(estimator=svr, param_grid=p_grid, cv=inner_cv, return_train_score=True, verbose=False)  #(user-defined)
    clf.fit(X_feature, y_class)
    model_scores[i] = clf.best_score_
    model_index[i] = clf.best_index_
    models_C_g.append(clf.best_params_)

    #models_C_g[i] = str(clf.best_params_)
    #model_std[i] = clf.std_
        
    #print Best_parameters
    file1.write(" %5i %s %5i %s %s %s %s \n" % (i, 'clf.best_index_ = ',  clf.best_index_, '    clf.best_params_ = ', clf.best_params_, ' clf.best_score_ = ',  clf.best_score_))       
    print (' RANDOM TRIALS = ', i, clf.best_params_, clf.best_index_ )  # Best C and gamma,  
    file0.write(" %s %5i %s %s %s %s \n" % (' RANDOM TRIALS = ', i, clf.best_params_, clf.best_index_, '    ', clf.best_score_  )) 

                                              # Index specifying C and gamma pair (the parameter starting from the leftmost varries slower than the others)
    results = pd.DataFrame(clf.cv_results_)   # mean_test_score for each pair of param_C param_gamma over the splits,  
               
                                              # the results for k-fold cross validation and the relevant scores
    #print Grid_results                                        
    file2.write(" %s \n" % ('=========================================== =========================================== '))  
    file2.write(" %s  %5i \n" % ('NUM_TRIALS = ', i))   
    file2.write(" %s\n" % (results))  

file1.close()
file2.close()
#========================================================= grid search

#========================================================= output results
file_name = 'Best_parameters_Votes_RBF' 
completeName = os.path.join(save_path, file_name+".dat") 
file3 = open(completeName, "w")
TITLE1 = "Cross Validation"
TITLE2 = 'VARIABLES = "Index", "Vote" "Mean Score"'
TITLE3 = 'ZONE T="Zone 1", I=' + str(p_size) + ', F=POINT'    
file3.write("%s\n" % (TITLE1))
file3.write("%s\n" % (TITLE2))
file3.write("%s\n" % (TITLE3))

file_name = 'Best_parameters_Votes_RBF_2' 
completeName = os.path.join(save_path, file_name+".dat") 
file4 = open(completeName, "w")
TITLE1 = "Cross Validation"
TITLE2 = 'VARIABLES = "Index", "Vote" "Mean Score"'
TITLE3 = 'ZONE T="Zone 1", I=' + str(p_size) + ', F=POINT'    
file4.write("%s\n" % (TITLE1))
file4.write("%s\n" % (TITLE2))
file4.write("%s\n" % (TITLE3))

#file_name = 'model_votes' 
#completeName = os.path.join(save_path, file_name+".dat") 
#file3 = open(completeName, "w")

model_score_mean = np.zeros(p_size)

for ii in range(p_size):
    models_C_g2.append("0")

for ii in range(p_size):
    for jj in range(NUM_TRIALS):
        if model_index[jj]==ii:
            model_votes[ii] = model_votes[ii] + 1
            model_score_mean[ii] = model_score_mean[ii] + model_scores[jj]
            models_C_g2[ii] = models_C_g[jj]
       
print (p_size       )
      
for ii in range(p_size):
    if model_votes[ii] !=0.0: 
        model_score_mean[ii] = model_score_mean[ii]/model_votes[ii]
            
    file3.write(" %5i %5i %16.8f \n" % (ii, model_votes[ii],  model_score_mean[ii]))        
    file4.write(" %5i %5i %16.8f %s \n" % (ii, model_votes[ii],  model_score_mean[ii], models_C_g2[ii]))        
     
file0.close()
file3.close()
file4.close()

print   (model_votes)      
print   (model_scores)
#========================================================= output results

#===================================================================== plot
vote = plt.bar(range(p_size), model_votes)
plt.xlabel("Index(C, gamma)")
plt.legend([vote],
           ["Frequency"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("Vote", fontsize="14")
#===========================  
plt.savefig('./Frequency.png', dpi=300)
plt.show()
#=========================== plot
#===================================================================== plot
sys.exit('Program stopped here') 
 