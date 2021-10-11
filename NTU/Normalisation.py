# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 07:53:30 2020

@author: Shrutarv
"""
import numpy as np
from torch.utils.data import DataLoader
import torch
import csv
import pandas as pd

def max_min_values(data, values):
    temp_values = []
    #data = data.numpy()
    #data = data.reshape(data.shape[0],data.shape[2], data.shape[3])
    #print(data.shape)
    temp_values = []
    for i in range(data.shape[1]):
        attribute = []
        temp_max = np.max(data[:,i])
        temp_min = np.min(data[:,i])
        if (values[i][0] > temp_max):
            attribute.append(values[i][0])
        else:
            attribute.append(temp_max)
        if(values[i][1] < temp_min):
            attribute.append(values[i][1])
        else:
            attribute.append(temp_min)
        temp_values.append(attribute)  
    values = temp_values
    return values
   
'''
Input
data - input matrix to normalize
min_max - list of max and min values for all channels across the entire training and test data

output
returns normalized data between [0,1]

'''

#path = '/data/sawasthi/data/MoCAP_data/trainData/'
path = 'S:/Datasets/nturgbd_skeletons_s001_to_s017/train_data_tf.csv'
batch_size = 50
if torch.cuda.is_available():  
          dev = "cuda:1" 
else:  
          dev = "cpu"  
device = torch.device(dev)
print("preparing data for normalisation")

train_dataset = CustomDataSet(path)
dataLoader_train = DataLoader(train_dataset, shuffle=True,
                              batch_size=batch_size,
                               num_workers=0,
                               pin_memory=True,
                               drop_last=True)
df = pd.read_csv(path);
data_x = df.to_numpy()
# Initialise values list to store max and min values across all channels
value = []
for k in range(200):
    temp_list = []
    maxim = -9999
    minim = 9999
    temp_list.append(maxim)
    temp_list.append(minim)
    value.append(temp_list)
  
for b, harwindow_batched in enumerate(dataLoader_train):
    data_x = harwindow_batched["data"]
    data_x.to(device)
    value = max_min_values(data_x,value)
print("size of max min list of list", len(value),len(value[0]))

with open('S:/Datasets/nturgbd_skeletons_s001_to_s017/norm_values.csv', 'w') as f:
    fc = csv.writer(f, lineterminator='\n')
    fc.writerow(["min","max"])
    fc.writerows(value)
    

