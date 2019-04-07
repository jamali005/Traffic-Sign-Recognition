#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:50:40 2019

@author: jamali
"""

from PIL import Image
from skimage.feature import local_binary_pattern
import numpy as np
import glob
import os
import pandas as pd
from skimage.transform import resize


def data(path):
    Final_training_dir = path
    dfs_train = []
    for train_file in glob.glob(os.path.join(Final_training_dir, '*/GT-*.csv')):
        train_folder = train_file.split('/')[-2]
        df_train = pd.read_csv(train_file, sep=';')
        df_train['Filename'] = df_train['Filename'].apply(lambda x: os.path.join(Final_training_dir, train_folder, x))
        dfs_train.append(df_train)
        print(df_train['Filename'])
        
    train_df = pd.concat(dfs_train, ignore_index=True)
    return train_df

def dataset(D,x,y,pts,r):
    
    label = []
    Feats= [] 
    for tr in range(len(D)):
        print(str(tr+1)+'/'+str(len(D)))
    
        img_file = D['Filename'][tr]
        img = Image.open(img_file)
        img_gray = img.convert('L')
        cropped_image = img_gray.crop((D['Roi.X1'][tr],D['Roi.Y1'][tr],D['Roi.X2'][tr],D['Roi.Y2'][tr]))
    
        img_arr = np.array(cropped_image.getdata()).reshape(cropped_image.size[1],cropped_image.size[0]) 
        resized_image = resize(img_arr,(x,y)) 
        # LBP
         
        feat_lbp = local_binary_pattern(resized_image ,pts,r,'uniform').reshape(x*y)
        n_bins = int(feat_lbp.max()) + 1
        lbp_hist,_ = np.histogram(feat_lbp,bins=n_bins, range=(0,n_bins))
        lbp_hist = np.array(lbp_hist,dtype=float)
        lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
       
        Feats.append(lbp_prob) 
        # Class label
        label.append(D['ClassId'][tr])
    Feats_arr = np.array(Feats)
    Label_arr = np.array(label) 
    return Feats_arr,Label_arr 


def dataset_lbp_vect(D,x,y,pts,r):
    
    label = []
    Feats= [] #Feature vector of each image 
    for tr in range(len(D)):
        print(str(tr+1)+'/'+str(len(D)))
    
        img_file = D['Filename'][tr]
        img = Image.open(img_file)
        img_gray = img.convert('L')
        cropped_image = img_gray.crop((D['Roi.X1'][tr],D['Roi.Y1'][tr],D['Roi.X2'][tr],D['Roi.Y2'][tr]))
    
        img_arr = np.array(cropped_image.getdata()).reshape(cropped_image.size[1],cropped_image.size[0]) 
        resized_image = resize(img_arr,(x,y)) 
        # LBP
         
        feat_lbp = local_binary_pattern(resized_image ,pts,r,'uniform').reshape(x*y)
       
        Feats.append(feat_lbp) 
        # Class label
        label.append(D['ClassId'][tr])
    Feats_arr = np.array(Feats)
    Label_arr = np.array(label) 
    return Feats_arr,Label_arr 

train = data('/home/jamali/data/GTSRB/Final_Training/Images')
test = data('/home/ayoub/data/Final_Test')

train_feats,train_labels =dataset(train,17,17,1,1)
test_feats,test_labels =dataset(test,17,17,1,1)
#save train_table & test_table into csv files

np.savetxt(r'/home/jamali/data/17_17_1_1/train_feats.csv',train_feats,delimiter=',')
np.savetxt(r'/home/jamali/data/17_17_1_1/train_labels.csv',train_labels,delimiter=',')

np.savetxt(r'/home/jamali/data/17_17_1_1/test_feats.csv',test_feats,delimiter=',')
np.savetxt(r'/home/jamali/data/17_17_1_1/test_labels.csv',test_labels,delimiter=',')

