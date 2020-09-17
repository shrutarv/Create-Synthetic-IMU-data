# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 08:41:06 2020

@author: STUDENT
"""
import pandas as pd
import scipy as sp
from pyearth import Earth
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle

def max_min_values(data):
    values = []
    
    #print(data.shape)
    
    for attr in range(data.shape[1]):
        attribute = []
        temp_max = np.max(data[:,attr])
        temp_min = np.min(data[:,attr])
        attribute.append(temp_min)
        attribute.append(temp_max)
        values.append(attribute)  
    
    return values

def normalize(data, min_max):
    #print(len(min_max), len(min_max[0]))
    
    #print(data.shape)
   
    for j in range(1,len(data[0])-1):
        data[:,j] = (data[:,j] - min_max[j-1][0])/(min_max[j-1][1] - min_max[j-1][0])
    return data       
# up sampling rate
up = 4
df = pd.read_csv('S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/train_data25_39.csv')
data = df.values
data_new = data[:,1:31]
attr = np.zeros((100,1))
value = max_min_values(data_new)
'''
with open("S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/norm_values.csv", 'w') as f:
    fc = csv.writer(f, lineterminator='\n')
    fc.writerow(["min","max"])
    fc.writerows(value)
plt.plot(data[:,0],data[:,1])
'''
data = normalize(data,value)
i=21
index = 12
x_sampled = np.linspace(np.min(data[:,0]), np.max(data[:,0]), len(data)*up)
y_sampled = []
for i in range(1,(data.shape[1]-1)):
    #for index in range(12,len(data[0])*up-12):
            
        #data_new = data[index-12:index+12,:]   
     
     f = sp.interpolate.interp1d(data[:,0],data[:,i], kind='linear')
     #f = sp.interpolate.UnivariateSpline(data[:,0],data[:,1])
     y_sampled.append(f(x_sampled))
     # plt.plot(data[1:10,0],data[1:10,i],'o',x_new[1:10],y_new,'x')

ws = 100
ss = 1 
k = 0
data_dir = 'S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/pkl_files/'

for j in range((int((len(y_sampled[0])-ws)/ss)) + 1):    
     attr = np.zeros((100,1))
     
     for i in range(0,len(y_sampled)): 
       # out = sp.interpolate.splrep(data[k:k + ws,0],data[k:k + ws:,i])
        #out2 = sp.interpolate.splev(data[k:k + ws,0],out)
        #plt.plot(data[k:k + ws,0],data[k:k + ws,i],'o',data[k:k + ws,0], out2)
        resample = sp.interpolate.splrep(x_sampled[k:k + ws],y_sampled[i][k:k + ws],k=2)
        resample2 = sp.interpolate.splev(x_sampled[k:k + ws],resample)
        #plt.plot(data[k:k + ws,0],data[k:k + ws,i],'o',x_new[k:k + ws], resample2)
        attr = np.concatenate((attr,np.reshape(resample2,(len(resample2),1))),axis=1)
       
        
     attr_save = attr[:,1:]
     dir = data_dir + 'window' + str(j) + '.pkl'
     f = open(dir, 'wb')
     
     pickle.dump(attr_save, f, protocol=pickle.HIGHEST_PROTOCOL)
     
     #print("dumping")
     f.close()
     k = k + ss
resample2 = sp.interpolate.splev(x_new,out, der=2)
for i in range(1,10,2):
    print(i)
with open('S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/pkl_files/window4479.pkl', 'rb') as f:
    d = pickle.load(f)