import pandas as pd
import sys
import numpy as np
import os
import pickle
from sliding_window import sliding_window
from opportunity import *
import glob
import csv

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
    X, y, y_all = opp_sliding_window(data_x, labels.astype(int),
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
        dir = data_dir + "seq_"  + "_" + str(k) + "_" + str(counter_seq) + ".pkl"
        obj = {"data" : seq, "label" : y[f], "labels" : y_all[f]}
        #f = open(os.path.join(dir, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
        f = open(dir, 'wb')
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        counter_seq += 1
        print("dumping")
        f.close()
 
def max_min_values(data, values):
    temp_values = []
    for i in range(data_x.shape[1]):
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
   
config = {
    "NB_sensor_channels":113,
    "sliding_window_length":24,
    "proportions":1,
    "sliding_window_step":12,
    "filter_size":5,
    "num_filters":64,
    "network":"cnn",
    "output":"softmax",
    "num_classes":18,
    "reshape_input":False,
    #"dataset_root":'/vol/actrec/Opportunity/',
    "dataset_root":'S:/MS A&R/4th Sem/Thesis/OpportunityUCIDataset/OpportunityUCIDataset/',
    "dataset":'gesture',
    "label_pos":'end'
    }


ws = (24,113)
ss = (12,113)    
#ss = (25,31)
sliding_window_length = 24  
#sliding_window_length = 100    
sliding_window_step = 12
opp = Opportunity(config, 'train')
x_train, y_train = opp.load_data()
opp_test = Opportunity(config,'val')
x_val, y_val = opp.load_data()
opp_test = Opportunity(config,'test')
x_test, y_test = opp.load_data()

data_dir = "/data/sawasthi/data/opportunity/trainData/"
#data_dir = "S:/MS A&R/4th Sem/Thesis/OpportunityUCIDataset/OpportunityUCIDataset/pklfile/train/"
#data_dir = "S:/MS A&R/4th Sem/Thesis/PAMAP2_Dataset/pkl files/"
#for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1)):
#    print (np.shape(i[:,0]))
#dataset = 'S:/MS A&R/4th Sem/Thesis/PAMAP2_Dataset/'
label = y_train.astype(int)
lab = np.zeros((len(label),20), dtype=int)
lab[:,0] = label
X = x_train.astype(object)
k = 0
example_creating_windows_file(k, X, lab, data_dir)

data_dir = "/data/sawasthi/data/opportunity/testData/"
label = y_test.astype(int)
lab = np.zeros((len(label),20), dtype=int)
lab[:,0] = label
x = x_test.astype(object)
k = 0
example_creating_windows_file(k, X, lab,data_dir)

data_dir =  "/data/sawasthi/data/opportunity/validationData/"
label = y_val.astype(int)
lab = np.zeros((len(label),20), dtype=int)
lab[:,0] = label
#Y_train = np.reshape(Y_train,(len(label),1))
x = x_val.astype(object)
k = 0
example_creating_windows_file(k, X, lab,data_dir)
#os.chdir('/vol/actrec/DFG_Project/2019/Mbientlab/recordings_2019/07_IMU_synchronized_annotated/' + folder_name)
#os.chdir("/vol/actrec/DFG_Project/2019/MoCap/recordings_2019/14_Annotated_Dataset/" + folder_name)
#os.chdir("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/")
#os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/" + folder_name)
#os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/OMoCap data/" + folder_name)


      

