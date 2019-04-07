#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:32:56 2019

@author: jamali
"""

import pandas as pd   
from sklearn.svm import SVC 
from sklearn import metrics
from joblib import dump

#_lbp_vect

X_train=pd.read_csv("/home/jamali/data/50_50_1_1/train_feats_lbp_vect.csv", delimiter=",") #, delimiter="\t" 
y_train= pd.read_csv("/home/jamali/data/50_50_1_1/train_labels_lbp_vect.csv", delimiter=",")  



X_test=pd.read_csv("/home/jamali/data/50_50_1_1/test_feats_lbp_vect.csv", delimiter=",") #, delimiter="\t"
y_test= pd.read_csv("/home/jamali/data/50_50_1_1/test_labels_lbp_vect.csv", delimiter=",")  


svclassifier = SVC(C=1,kernel='linear',decision_function_shape='ovo',verbose=True)
svclassifier.fit(X_train,y_train) 


 
y_pred = svclassifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


dump(svclassifier, '/home/jamali/data/50_50_1_1/fruit.joblib') 


















