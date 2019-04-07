# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os

import cv2 as cv
import imutils

from skimage import feature
#Rename images
for file in os.listdir():
    i=0
    for img in os.listdir(file):
       os.rename(os.path.join(os.getcwd()+"/"+file,img),os.path.join(os.getcwd()+"/"+file,str(i)+".ppm"))
       i=i+1
#Preprocessing 
count=0
X=[]
Y=[]
for file in os.listdir():
    for i in os.listdir(file):
        try :
            img =cv.imread(os.getcwd()+"/"+file+"/"+i)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            edged = imutils.auto_canny(gray)
            cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            maxi=max(cnts, key=cv.contourArea)
            (x, y, w, h) = cv.boundingRect(maxi)
            Sign = gray[y:y+h,x:x+w]
            Sign = cv.resize(Sign,(50,50))
            H = feature.hog(Sign , orientations=9, pixels_per_cell=(10,10),cells_per_block=(2, 2), transform_sqrt=True , block_norm="L1")
            X.append(H)
            Y.append(file)
            count=count+1
            
        except:
            print(count)
#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
#Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score=0
for i in range(len(cm)):
    score+=cm[i][i]
score=score/11760
score
    