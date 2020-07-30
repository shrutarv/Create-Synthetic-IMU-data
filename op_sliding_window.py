import pandas as pd
import sys
import numpy as np
import os
import pickle
NUM_CLASSES = 6
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
                data_y_labels = np.asarray(data_y_labels)
            
            
            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf
            
            #All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1))])
    
    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)








def example_creating_windows_file():
        # Sliding window approach

    print("Starting sliding window")
    X, y, y_all = opp_sliding_window(data_x, labels.astype(int),
                                     sliding_window_length,
                                     sliding_window_step, label_pos_end = False)
    counter_seq = 1
    for f in range(X.shape[0]):
       # try:
        
        sys.stdout.write('\r' + 'Creating sequence file '
                                'number {} with id {}'.format(f, counter_seq))
        sys.stdout.flush()

        # print "Creating sequence file number {} with id {}".format(f, counter_seq)
        seq = np.reshape(X[f], newshape = (1, X.shape[1], X.shape[2]))
        seq = np.require(seq, dtype=np.float)

        obj = {"data" : seq, "label" : y[f], "labels" : y_all[f]}
        f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        counter_seq += 1
        f.close()
    

data_dir =  "S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/"
data_y = pd.read_csv("S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S07/L01_S07_R01_labels.csv") 
data_y = data_y.values
data_x = a
data_x=data_x[:,1:31]
for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1)):
    
    print (np.shape(i[:,0]))