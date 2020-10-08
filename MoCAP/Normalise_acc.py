# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:19:36 2020

@author: STUDENT
"""
import pandas as pd
import sys
import numpy as np
import os
import pickle
from sliding_window import sliding_window
import glob
import csv
import scipy.interpolate as sp

def normalize(data_n, min_max, string):
    #print(len(min_max), len(min_max[0]))
    
    for j in range(len(data_n[0])):
        data_n[:,j] = (data_n[:,j] - min_max[j][1])/(min_max[j][0] - min_max[j][1]) 
    test = np.array(data_n[:,1:data_n.shape[1]])
        
    if (string=="train"):
        if(np.max(test)>1.001):
            print("Error",np.max(test))
        if(np.min(test)<-0.001):
            print("Error",np.min(test))
    else:
        test[test > 1] = 1
        test[test < 0] = 0
    #data = data.reshape(data.shape[0],1,data.shape[1], data.shape[2])
    #data = torch.tensor(data)
    return data_n

if __name__ == '__main__':
        
    ## Normalise the data
    value = pd.read_csv("/data/sawasthi/data/MoCAP_data/train_csv/value.csv") 
    value = value.values
    data = pd.read_csv("/data/sawasthi/data/MoCAP_data/train_csv/train.csv") 
    print("read training data")
    data = data.values
    print(data.shape[0],data.shape[1])
    data_norm = normalize(data,value,"train")
    print("normalized")
    print(data_norm.shape[0],data_norm.shape[1])
    
    t = np.zeros((1,data_norm.shape[1]))
    data_norm = np.concatenate((t,data_norm))
    np.savetxt("/data/sawasthi/data/MoCAP_data/train_csv/train_normal.csv", data_norm, delimiter=',')
