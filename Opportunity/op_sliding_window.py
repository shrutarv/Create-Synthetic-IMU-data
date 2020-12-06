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

def opp_sliding_window(data_x, data_y):
        ws = config['sliding_window_length']
        ss = config['sliding_window_step']

        logging.info('        Dataloader: Sliding window with ws {} and ss {}'.format(ws, ss))

        # Segmenting the data with labels taken from the end of the window
        data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
        if config['label_pos'] == 'end':
            data_y_labels = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
        elif config['label_pos'] == 'middle':
            # Segmenting the data with labels from the middle of the window
            data_y_labels = np.asarray([[i[i.shape[0] // 2]] for i in sliding_window(data_y, ws, ss)])
        elif config['label_pos'] == 'mode':
            data_y_labels = []
            for sw in sliding_window(data_y, ws, ss):
                count_l = np.bincount(sw.astype(int), minlength=config['num_classes'])
                idy = np.argmax(count_l)
                data_y_labels.append(idy)
            data_y_labels = np.asarray(data_y_labels)

        # Labels of each sample per window
        data_y_all = np.asarray([i[:] for i in sliding_window(data_y, ws, ss)])

        logging.info('        Dataloader: Sequences are segmented')

        return data_x.astype(np.float32), \
               data_y_labels.reshape(len(data_y_labels)).astype(np.uint8), \
               data_y_all.astype(np.uint8)
'''               
def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end = True):
    
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
'''
def example_creating_windows_file(k, data_x, labels, data_dir):
        # Sliding window approach

    print("Starting sliding window")
    print(data_x.shape)
    print(labels.shape)
    X, y, y_all = opp_sliding_window(data_x, labels.astype(int))
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
   
config = {
    "NB_sensor_channels":113,
    "sliding_window_length":24,
    "proportions":0.2,
    "sliding_window_step":12,
    "filter_size":5,
    "num_filters":64,
    "network":"cnn",
    "output":"softmax",
    "num_classes":18,
    "reshape_input":False,
    "dataset_root":'/vol/actrec/Opportunity/',
    #"dataset_root":'S:/MS A&R/4th Sem/Thesis/OpportunityUCIDataset/OpportunityUCIDataset/',
    "dataset":'gesture',
    "label_pos":'mode'
    }


ws = (24,113)
ss = (12,113)    
#ss = (25,31)
sliding_window_length = 24  
#sliding_window_length = 100    
sliding_window_step = 12
opp = Opportunity(config, 'train')
x_train, y_train = opp.load_data()
opp_val = Opportunity(config,'val')
x_val, y_val = opp_val.load_data()
opp_test = Opportunity(config,'test')
x_test, y_test = opp_test.load_data()

data_dir = "/data/sawasthi/data/opportunity/trainData_20/"
#data_dir = "S:/MS A&R/4th Sem/Thesis/OpportunityUCIDataset/OpportunityUCIDataset/pklfile/train/"
#data_dir = "S:/MS A&R/4th Sem/Thesis/PAMAP2_Dataset/pkl files/"
#for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1)):
#    print (np.shape(i[:,0]))
#dataset = 'S:/MS A&R/4th Sem/Thesis/PAMAP2_Dataset/'
X = x_train.astype(object)
k = 0
example_creating_windows_file(k, X , y_train, data_dir)

data_dir = "/data/sawasthi/data/opportunity/testData/"
x = x_test.astype(object)
k = 0
example_creating_windows_file(k, X, y_test,data_dir)

data_dir =  "/data/sawasthi/data/opportunity/validationData/"
#Y_train = np.reshape(Y_train,(len(label),1))
x = x_val.astype(object)
k = 0
example_creating_windows_file(k, X, y_val,data_dir)
#os.chdir('/vol/actrec/DFG_Project/2019/Mbientlab/recordings_2019/07_IMU_synchronized_annotated/' + folder_name)
#os.chdir("/vol/actrec/DFG_Project/2019/MoCap/recordings_2019/14_Annotated_Dataset/" + folder_name)
#os.chdir("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/")
#os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/" + folder_name)
#os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/OMoCap data/" + folder_name)


      

