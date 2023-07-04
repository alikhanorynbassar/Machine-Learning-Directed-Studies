
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

#===================================================================== import data
save_path = os.path.dirname(os.path.abspath(__file__)) 
file_name = 'Features' 
completeName = os.path.join(save_path, file_name+".dat") 
feature_data = np.loadtxt(completeName)
feature_data = feature_data.reshape((1000,20))
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

#===================================================================== split data to tranining data and testing data
isize_train = int(class_labels.size * 8/10)  #(User-defined)
isize_test = int(class_labels.size - isize_train)
print ('isize_train = ', isize_train)
print ('isize_test = ', isize_test)
feature_train = feature_data[0:isize_train] 
class_labels_train = class_labels[0:isize_train]
feature_test = feature_data[isize_train:class_labels.size]
class_labels_test = class_labels[isize_train:class_labels.size]
#print ' = ',class_labels_test
#===================================================================== split data to tranining data and testing data

#===================================================================== Classification, svm Support Vector Machine Algorithm
# Create and fit a svm Support Vector Machine classifier
X = feature_train 
y = class_labels_train
my_C = 30  # SVM regularization parameter  #(User-defined)
my_gamma = 0.13 #(User-defined)
svm_SVC_rbf = svm.SVC(kernel='rbf', gamma=my_gamma, C=my_C)   #(User-defined)
svm_SVC_rbf.fit(X, y) 
print ('svm_SVC_rbf score  = ', svm_SVC_rbf.score(feature_data[isize_train:], class_labels[isize_train:]))
#===================================================================== Classification, svm Support Vector Machine Algorithm

#===================================================================== SVM classifier testing (Test data)
file_name = 'Mis_class_deg_Test' 
completeName = os.path.join(save_path, file_name+".dat") 
file1 = open(completeName, "w")
file_name = 'Score_Test' 
completeName = os.path.join(save_path, file_name+".dat") 
file2 = open(completeName, "w")
file_name = 'Prediction_Test' 
completeName = os.path.join(save_path, file_name+".dat") 
file3 = open(completeName, "w")

rs = np.zeros(class_labels_test.size)
pr = np.zeros(class_labels_test.size)
rs2 = np.ones((class_labels_test.size), dtype=bool)

for i in range(0,class_labels_test.size): 
    pr[i] = svm_SVC_rbf.predict([feature_test [i]]) 
    rs [i] = pr[i] - class_labels_test [i] 

    if rs[i]==0.0:
        rs2[i] = True 
        file3.write("%2i %2i \n" % (pr[i], class_labels_test[i]))
    else:    
        rs2[i] = False 
        file3.write("%2i %2i %s\n" % (pr[i], class_labels_test[i], str(rs2[i])))

    file1.write('{:4.2f}\n'.format(rs[i]))
    file2.write(str(rs2[i]))
    file2.write('\n')
    
print (isize_train, class_labels_test.size, isize_train + class_labels_test.size)

file1.close()
file2.close()
file3.close()
#===================================================================== SVM classifier testing (Test data)

#===================================================================== SVM classifier testing(Training data)
file_name = 'Mis_class_deg_Train' 
completeName = os.path.join(save_path, file_name+".dat") 
file1 = open(completeName, "w")
file_name = 'Score_Train' 
completeName = os.path.join(save_path, file_name+".dat") 
file2 = open(completeName, "w")
file_name = 'Prediction_Train' 
completeName = os.path.join(save_path, file_name+".dat") 
file3 = open(completeName, "w")

rs = np.zeros(class_labels_train.size)
pr = np.zeros(class_labels_train.size)
rs2 = np.ones((class_labels_train.size), dtype=bool)

for i in range(0,class_labels_train.size): 
    pr[i] = svm_SVC_rbf.predict([feature_train [i]]) 
    rs [i] = pr[i] - class_labels_train [i] 

    if rs[i]==0.0:
        rs2[i] = True 
        file3.write("%2i %2i \n" % (pr[i], class_labels_train[i]))
    else:    
        rs2[i] = False 
        file3.write("%2i %2i %s\n" % (pr[i], class_labels_train[i], str(rs2[i])))

    file1.write('{:4.2f}\n'.format(rs[i]))
    file2.write(str(rs2[i]))
    file2.write('\n')
    
print (isize_train, class_labels_train.size, isize_train + class_labels_train.size)

file1.close()
file2.close()
file3.close()
#===================================================================== SVM classifier testing (Training data)

#===================================================================== Plot  decision boundaries
# Plotting decision regions
#(User-defined parameters for plot)
plt.close('all')
'''
x_min, x_max = \
X[:, 0].min() - abs(X[:, 0].max()-X[:, 0].min()) * 0.01, X[:, 0].max() + abs(X[:, 0].max()-X[:, 0].min()) * 0.01
y_min, y_max = \
X[:, 1].min() - abs(X[:, 1].max()-X[:, 1].min()) * 0.01, X[:, 1].max() + abs(X[:, 1].max()-X[:, 1].min()) * 0.01
'''
x_min, x_max = X[:, 0].min(), X[:, 0].max() 
y_min, y_max = X[:, 1].min(), X[:, 1].max()

print ('x_min, x_max = ', x_min, x_max)                   
print ('y_min, y_max = ', y_min, y_max) 

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005),
                     np.arange(y_min, y_max, 0.005))                                                              
print ('xx.shape = ', xx.shape)
print ('yy.shape = ', yy.shape)

Z = svm_SVC_rbf.predict(np.c_[xx.ravel(), yy.ravel()])    # get prediction for each grid point
print ('Z.shape = ', Z.shape)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
print ('Z.shape = ', Z.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)  # color the grid points based on the calss labels  

# Plot also the training points
#plt.suptitle('Classification', fontsize=40)
plt.xlabel('Plume Vol.', fontsize=10)
plt.ylabel('Slab Vol.', fontsize=10)


# Set the tick labels font
ax = plt.subplot() # Defines ax variable by creating an empty plot
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')       # label font
    label.set_fontsize(20)            # label font size

#---------------------- scatter
plt.scatter(X[:, 0], X[:, 1], c=y, s=15, cmap=plt.cm.Paired, edgecolors='k')    # show the data points and misclassifications by class colors 
#plt.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap='binary_r')   # show the data points and misclassifications by class colors

plt.title('SVC-Linear Algorithm, C=6000', fontsize=10)        # title
plt.axis('tight')
#ax.tick_params(direction='out', length=16, width=2, colors='g', labelsize=40)
 
#---------------------- legend
cb = colorbar()
cb.ax.tick_params(which='major',width=1, length=1, labelsize=15, pad=1) # Font size of the labels
cb.set_label(label='Class',weight='bold',fontsize=20)
cb.set_ticks([1,3,5,7,9,11,13,15])
#---------------------- legend

plt.savefig('./SVC_RBF.png', dpi=300)
plt.show()
#===================================================================== Plot  decision boundaries
print ('svm_SVC_rbf score  test = ', svm_SVC_rbf.score(feature_data[isize_train:], class_labels[isize_train:]))
print ('svm_SVC_rbf score  train = ', svm_SVC_rbf.score(feature_data[0:isize_train], class_labels[0:isize_train]))
print('class_labels[isize_train:].size = ', class_labels[isize_train:].size)
print('class_labels[0:isize_train].size = ', class_labels[0:isize_train].size)
#===================================================================== 
sys.exit('Program stopped here - end of program') 


