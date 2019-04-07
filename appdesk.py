#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:44:02 2019

@author: jamali
"""



from PIL import Image
from skimage.feature import local_binary_pattern
import numpy as np
import glob
import os
import pandas as pd
from skimage.transform import resize
from joblib import load
from termcolor import colored



def data(path):
    Final_training_dir = path
    dfs_train = []
    for train_file in glob.glob(os.path.join(Final_training_dir, '*/GT-*.csv')):
        train_folder = train_file.split('/')[-2]
        df_train = pd.read_csv(train_file, sep=';')
        df_train['Filename'] = df_train['Filename'].apply(lambda x: os.path.join(Final_training_dir, train_folder, x))
        dfs_train.append(df_train)
        
    train_df = pd.concat(dfs_train, ignore_index=True)
    return train_df



def dataset_lbp_vect(D,i,x,y,pts,r):
    
    label = []
    Feats= [] 


    img_file = D['Filename'][i]
    img = Image.open(img_file)
    #plt.imshow(img)
    img.show()
    img_gray = img.convert('L')
    cropped_image = img_gray.crop((D['Roi.X1'][i],D['Roi.Y1'][i],D['Roi.X2'][i],D['Roi.Y2'][i]))

    img_arr = np.array(cropped_image.getdata()).reshape(cropped_image.size[1],cropped_image.size[0]) 
    resized_image = resize(img_arr,(x,y)) 
    # LBP
     
    feat_lbp = local_binary_pattern(resized_image ,pts,r,'uniform').reshape(x*y)
   
    Feats.append(feat_lbp) 
    # Class label
    label.append(D['ClassId'][i])
    Feats_arr = np.array(Feats)
    Label_arr = np.array(label) 
    return Feats_arr,Label_arr 



test = data('/home/jamali/data/Final_Test')

clf = load( '/home/jamali/data/50_50_1_1/fruit.joblib')


def  predict(D,x):
    a,b = dataset_lbp_vect(D,x,50,50,1,1)
    signe_names=pd.read_csv("/home/jamali/data/GTSRB/Final_Training/Images/sign_names.csv", delimiter=",")
    c=signe_names['SignName'][b].to_string()
    print("L'image actuel est :",c)
    y_pred = clf.predict(a)
    print("le resultat de la prediction est :",colored(signe_names['SignName'][y_pred].to_string(),'green'))
    
    return


test = data('/home/jamali/data/Final_Test')

clf = load( '/home/jamali/data/50_50_1_1/fruit.joblib')
while(1):
    print("********************************************************************************")
    a = int(input('entrer l indice de l image:'))
    predict(test,a)
    










