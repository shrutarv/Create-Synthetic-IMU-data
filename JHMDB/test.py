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
from scipy.interpolate import UnivariateSpline

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
df = pd.read_csv('S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/train_data.csv')
data = df.values
label = data[:,31]
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
i=10
index = 12
x_sampled = np.linspace(np.min(data[:,0]), np.max(data[:,0]), len(data)*up)
y_sampled2 = np.zeros((len(x_sampled),1))
sampled_data = []
for i in range(1,(data.shape[1]-1)):
     print(i)
    #for index in range(12,len(data[0])*up-12):
            
        #data_new = data[index-12:index+12,:]   
     resample2 = sp.splrep(data[:,0],data[:,i])
     temp = sp.splev(x_sampled,resample2, der=0)
     y_sampled2 = np.concatenate((y_sampled2,np.reshape(temp,(len(temp),1))),axis=1)
     
     spl = UnivariateSpline(data[:,0],data[:,i], k=4, s=0) 
     sampled_data = spl(x_sampled)
     y = spl.derivative(1)
     '''
     f = sp.interpolate.interp1d(data[:,0],data[:,i], kind='linear')
     #f = sp.interpolate.UnivariateSpline(data[:,0],data[:,1])
     sampled_data = f(x_sampled)
     y_sampled2 = np.concatenate((y_sampled2,np.reshape(sampled_data,(len(sampled_data),1))),axis=1)
     '''
plt.title("Blue: Sampled values.  Red: True Values")
plt.plot(data[1:100,0],data[1:100,i],'r',x_sampled[1:400],y_sampled2[1:400,31],'g')

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
     
     l = label[k:k + ws]   
     attr_save = attr[:,1:]
     seq = np.reshape(attr_save,(1,len(attr_save),attr_save.shape[1]))
     
     dir = data_dir + 'window' + str(j) + '.pkl'
     obj = {"data" : seq, "label" : l, "labels" : l}
     f = open(obj, 'wb')
     
     pickle.dump(attr_save, f, protocol=pickle.HIGHEST_PROTOCOL)
     
     #print("dumping")
     f.close()
     k = k + ss
resample = sp.interpolate.splrep(x_sampled[1:400],y_sampled[1:400,i], k=5)
resample2 = sp.interpolate.splev(x_sampled[1:400],resample, der=1)
plt.title("Blue: True values.  Red: Sampled Values")
plt.plot(data[1:100,0],data[1:100,i],'o',x_sampled[1:399],resample2[1:399],'r')  

'''
for i in range(1,10,2):
    print(i)
with open('S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/pkl/seq__0_506.pkl', 'rb') as f:
    d2 = pickle.load(f)
'''