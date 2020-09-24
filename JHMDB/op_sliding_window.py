import pandas as pd
import sys
import numpy as np
import os
import pickle
from sliding_window import sliding_window
from pre_processing import *
import glob
import csv
import scipy.interpolate as sp

NUM_CLASSES = 8
def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end = True):
    '''
    Performs the sliding window approach on the data and the labels
    
    return three arrays.
    - data, an array where first dim is the windows
    - labels per window according to end, middle or mode
    - all labels per window
    
    @param data_x: ids for train
    @param data_y: ids for train
    @param ws: ids for train
    @param ss: ids for train
    @param label_pos_end: ids for train
    '''    


    print("Sliding window: Creating windows {} with step {}".format(ws, ss))
    
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    
    # Label from the end
    if label_pos_end:
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1))])
    else:
    
        #Label from the middle
        if False:
            data_y_labels = np.asarray([[i[i.shape[0] // 2]] for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1))])
        else:
            count_l=[]
            idy = []
            #Label according to mode
            try:
                
                data_y_labels = []
                for sw in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1)):
                    labels = np.zeros((20)).astype(int)
                    count_l = np.bincount(sw[:,0], minlength = NUM_CLASSES)
                    idy = np.argmax(count_l)
                    attrs = np.sum(sw[:,1:], axis = 0)
                    attrs[attrs > 0] = 1
                    labels[0] = idy  
                    labels[1:] = attrs
                    data_y_labels.append(labels)
                print(len(data_y_labels))
                data_y_labels = np.asarray(data_y_labels)
                
            
            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf
            
            #All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1))])
    
    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)

def example_creating_windows_file(k, data_x, labels, data_dir):
        # Sliding window approach

    print("Starting sliding window")
    print(data_x.shape)
    print(labels.shape)
    X, y, y_all = opp_sliding_window(data_x, labels,
                                     sliding_window_length,
                                     sliding_window_step, label_pos_end = False)
    print(X.shape)
    print(y.shape)
    print(y_all.shape)
    counter_seq = 0
    value = 0
    if (X.shape[0]<y.shape[0]):
        value = X.shape[0]
    else:
        value = y.shape[0]
   # for f in range(X.shape[0]):
    for f in range(value):
       # try:
        sys.stdout.write('\r' + 'Creating sequence file '
                                'number {} with id {}'.format(f, counter_seq))
        sys.stdout.flush()

        # print "Creating sequence file number {} with id {}".format(f, counter_seq)
        seq = np.reshape(X[f], newshape = (1, X.shape[1], X.shape[2]))
        seq = np.require(seq, dtype=np.float)
        # interpolation
        dir = data_dir + "seq_"  + "_" + str(k) + "_" + str(counter_seq) + ".pkl"
        obj = {"data" : seq, "label" : y[f], "labels" : y_all[f]}
        #f = open(os.path.join(dir, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
        f = open(dir, 'wb')
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        counter_seq += 1
        print("dumping")
        f.close()
 
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


def normalize(data, min_max, string):
    #print(len(min_max), len(min_max[0]))
    
    for j in range(1,len(data[0])-1):
        data[:,j] = (data[:,j] - min_max[j-1][0])/(min_max[j-1][1] - min_max[j-1][0]) 
    test = np.array(data[:,1:31])
        
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
    return data
   

# up sampling rate
up = 4
df = pd.read_csv('/data/sawasthi/Thesis--Create-Synthetic-IMU-data/JHMDB/train_data.csv')
#df = pd.read_csv('S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/train_data.csv')
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

data = normalize(data,value, "train")
print("train data normalized")

x_sampled = np.linspace(np.min(data[:,0]), np.max(data[:,0]), len(data)*up)
y_sampled = np.zeros((len(x_sampled),1))
sampled_data = []
for i in range(1,(data.shape[1]-1)):
    #for index in range(12,len(data[0])*up-12):
            
        #data_new = data[index-12:index+12,:]   
     
     f = sp.interp1d(data[:,0],data[:,i], kind='linear')
     #f = sp.interpolate.UnivariateSpline(data[:,0],data[:,1])
     sampled_data = f(x_sampled)
     resample = sp.splrep(x_sampled,sampled_data)
     acc = sp.splev(x_sampled,resample, der=2)
     y_sampled = np.concatenate((y_sampled,np.reshape(acc,(len(acc),1))),axis=1)
     
     #y_sampled.append(f(x_sampled))
     # plt.plot(data[1:10,0],data[1:10,i],'o',x_new[1:10],y_new,'x')
data = y_sampled[:,1:]

#ws = (100,31)
ws = (100,30) 
ss = (25,30)     
#ss = (25,31)
sliding_window_length = 100   
#sliding_window_length = 100    
sliding_window_step = 25


#data_dir = "/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/"
#data_dir = "S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/"
#df = pd.read_csv('S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/train_data25_39.csv')

#for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1)):
#    print (np.shape(i[:,0]))
#dataset = 'S:/MS A&R/4th Sem/Thesis/PAMAP2_Dataset/'
#dataset = '/vol/actrec/PAMAP/'
data_dir =  '/data/sawasthi/data/JHMDB/trainData_a/'
#data_dir = 'S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/pkl/'
label = data[:,31].astype(int)
lab = np.zeros((len(label),20), dtype=int)
lab[:,0] = label
X = data[:,1:31]
k = 0
example_creating_windows_file(k, X, lab, data_dir)
print("train data pickled")
#data_dir = 'S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/pkl'
data_dir =  '/data/sawasthi/data/JHMDB/testData_a/'
df = pd.read_csv('/data/sawasthi/Thesis--Create-Synthetic-IMU-data/JHMDB/test_data.csv')
data = df.values
data = normalize(data,value, "test")
print("test data normalized")
label = data[:,31].astype(int)
lab = np.zeros((len(label),20), dtype=int)
lab[:,0] = label
X = data[:,1:31]
k = 0
example_creating_windows_file(k, X, lab, data_dir)
print("test data pickled")

data_dir =  '/data/sawasthi/data/JHMDB/validationData_a/'
#data_dir =  '/data/sawasthi/data/JHMDB/validationData/'
df = pd.read_csv('/data/sawasthi/Thesis--Create-Synthetic-IMU-data/JHMDB/validation_data.csv')
data = df.values
data = normalize(data,value, "validation")
print("validation data normalized")
label = data[:,31].astype(int)
lab = np.zeros((len(label),20), dtype=int)
lab[:,0] = label
X = data[:,1:31]
k = 0
example_creating_windows_file(k, X, lab, data_dir)
print("validation data pickled")
#os.chdir('/vol/actrec/DFG_Project/2019/Mbientlab/recordings_2019/07_IMU_synchronized_annotated/' + folder_name)
#os.chdir("/vol/actrec/DFG_Project/2019/MoCap/recordings_2019/14_Annotated_Dataset/" + folder_name)
#os.chdir("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/")
#os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/" + folder_name)
#os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/OMoCap data/" + folder_name)


      

