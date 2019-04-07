#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:32:56 2019

@author: jamali
"""

import pandas as pd   
from sklearn.svm import SVC 
from sklearn import metrics
import numpy as np

#_lbp_vect

X_train=pd.read_csv("/home/jamali/data/50_50_1_1/train_feats_lbp_vect.csv", delimiter=",") #, delimiter="\t" 
y_train= pd.read_csv("/home/jamali/data/50_50_1_1/train_labels_lbp_vect.csv", delimiter=",")  



X_test=pd.read_csv("/home/jamali/data/50_50_1_1/test_feats_lbp_vect.csv", delimiter=",") #, delimiter="\t"
y_test= pd.read_csv("/home/ayoub/data/50_50_1_1/test_labels_lbp_vect.csv", delimiter=",")  



'''
trMaxs = np.amax(X_train,axis=0) #Finding maximum along each column
trMins = np.amin(X_train,axis=0) #Finding maximum along each column
trMaxs_rep = np.tile(trMaxs,(len(X_train),1)) #Repeating the maximum value along the rows
trMins_rep = np.tile(trMins,(len(X_train),1)) #Repeating the minimum value along the rows
trainFeatsNorm = np.divide(X_train-trMins_rep,trMaxs_rep) #Element-wise division



trMaxs = np.amax(X_test,axis=0) #Finding maximum along each column
trMins = np.amin(X_test,axis=0) #Finding maximum along each column
trMaxs_rep = np.tile(trMaxs,(len(X_test),1)) #Repeating the maximum value along the rows
trMins_rep = np.tile(trMins,(len(X_test),1)) #Repeating the minimum value along the rows
testFeatsNorm = np.divide(X_test-trMins_rep,trMaxs_rep) #Element-wise division
'''


svclassifier = SVC(C=1000,kernel='linear',decision_function_shape='ovo',verbose=True)
svclassifier.fit(X_train,y_train) 


 
y_pred = svclassifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
