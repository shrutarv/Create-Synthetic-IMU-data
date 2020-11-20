'''
@author: Fernando Moya Rueda, Shrutarv Awasthi
Pattern Recognition Group
Technische Universitaet Dortmund
Process the Pamap2 dataset. It selects the files, sensor channels. In addition, it normalizes
and downsamples the signal measurements.
It creates a cPickle file the three matrices (train, validation and test),
containing the sensor measurements (row for samples and columns for sensor channels) and the annotated label
The dataset can be downloaded in  
http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
'''


import os
import numpy as np
import pickle
import logging


# Number of sensor channels employed in the Pamap2
NB_SENSOR_CHANNELS = 112


# File names of the files defining the PAMAP2 data.
Opportunity_training_files = [ 'S1-ADL1.dat',                'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                          'S2-ADL1.dat', 'S2-ADL2.dat',                               'S2-ADL5.dat', 'S2-Drill.dat',
                          'S3-ADL1.dat', 'S3-ADL2.dat',                               'S3-ADL5.dat', 'S3-Drill.dat', 
                          'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'
                          ]
Opportunity_validation_files = ['S1-ADL2.dat','S2-ADL3.dat']
Opportunity_test_files = ['S2-ADL4.dat', 'S3-ADL3.dat', 'S3-ADL4.dat']


def divide_x_y(self, raw_data):
    """Segments each sample into features and label

    :param raw_data: numpy integer matrix
        Sensor data
    :param task: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Recording time, Features encapsulated into a matrix and labels as an array
    """

    try:
        data_t = raw_data[:, 0]
        data_x = raw_data[:, 1:114]
        if self.config['dataset'] not in ['locomotion', 'gesture']:
            raise RuntimeError("Invalid label: '%s'" % self.config['dataset'])
        if self.config['dataset'] == 'locomotion':
            logging.info("        Dataloader: Locomotion")
            data_y = raw_data[:, 114]  # Locomotion label
        elif self.config['dataset'] == 'gesture':
            logging.info("        Dataloader: Gestures")
            data_y = raw_data[:, 115]  # Gestures label
    except KeyError:
        logging.error(KeyError)

    return data_t, data_x, data_y




def adjust_idx_labels(data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function adjust the labels picking the labels
    for the protocol settings
    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """


    data_y[data_y == 406516] = 1
    data_y[data_y == 406517] = 2
    data_y[data_y == 404516] = 3
    data_y[data_y == 404517] = 4
    data_y[data_y == 406520] = 5
    data_y[data_y == 404520] = 6
    data_y[data_y == 406505] = 7
    data_y[data_y == 404505] = 8
    data_y[data_y == 406519] = 9
    data_y[data_y == 404519] = 10
    data_y[data_y == 406511] = 11
    data_y[data_y == 404511] = 12
    data_y[data_y == 406508] = 13
    data_y[data_y == 404508] = 14
    data_y[data_y == 408512] = 15
    data_y[data_y == 407521] = 16
    data_y[data_y == 405506] = 17
    return data_y

def normalize(self, raw_data, max_list, min_list):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param max_list: numpy integer array
        Array containing maximums values for every one of the 113 sensor channels
    :param min_list: numpy integer array
        Array containing minimum values for every one of the 113 sensor channels
    :return:
        Normalized sensor data
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(raw_data.shape[1]):
        raw_data[:, i] = (raw_data[:, i] - min_list[i]) / diffs[i]
    #     Checking the boundaries
    raw_data[raw_data > 1] = 0.99
    raw_data[raw_data < 0] = 0.00
    return raw_data

def process_dataset_file(data):
    """Function defined as a pipeline to process individual Pamap2 files
    :param data: numpy integer matrix
        channel data: samples in rows and sensor channels in columns
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into samples-channel measurements (x) and labels (y)
    """

    # Data is divided in time, sensor data and labels
    data_t, data_x, data_y =  divide_x_y(data)

    print ("data_x shape {}".format(data_x.shape))
    print ("data_y shape {}".format(data_y.shape))
    print ("data_t shape {}".format(data_t.shape))
    
    # Labels are adjusted
    data_y = adjust_idx_labels(data_y)
    data_y = data_y.astype(int)
    
    # Select correct columns
    
   
    print ("SIZE OF THE SEQUENCE IS CERO")
    
    print ("data_x shape {}".format(data_x.shape))
    print ("data_y shape {}".format(data_y.shape))
    print ("data_t shape {}".format(data_t.shape))
    
    return data_x, data_y




def generate_data(dataset, target_filename):
    """Function to read the Pamap2 raw data and process the sensor channels
    of the protocol settings
    :param dataset: string
        Path with original pamap2 folder
    :param target_filename: string
        Path of the expected file.
    """


    X_train = np.empty((0, NB_SENSOR_CHANNELS))
    y_train = np.empty((0))
    
    X_val = np.empty((0, NB_SENSOR_CHANNELS))
    y_val = np.empty((0))
    
    X_test = np.empty((0, NB_SENSOR_CHANNELS))
    y_test = np.empty((0))
   
    print ('Processing dataset files ...')
    for filename in Opportunity_training_files:
        
        # Train partition
        try:
            print ('Train... file {0}'.format(filename))
            data = np.loadtxt(dataset + filename)
            print ('Train... data size {}'.format(data.shape))
            x, y = process_dataset_file(data)
            print (x.shape)
            print(y.shape)
            X_train = np.vstack((X_train, x))
            y_train = np.concatenate([y_train, y])
        except KeyError:
            print ('ERROR: Did not find {0} in zip file'.format(filename))
            
    for filename in Opportunity_validation_files:          
        # Validation partition
         try:
             print ('Val... file {0}'.format(filename))
             data = np.loadtxt(dataset + filename)
             print ('Val... data size {}'.format(data.shape))
             x, y = process_dataset_file(data)
             print (x.shape)
             print (y.shape)
             X_val = np.vstack((X_val, x))
             y_val = np.concatenate([y_val, y])
         except KeyError:
             print ('ERROR: Did not find {0} in zip file'.format(filename))
         
    for filename in Opportunity_test_files:        
        # Testing partition
        try:
            print ('Test... file {0}'.format(filename))
            data = np.loadtxt(dataset + filename)
            print ('Test... data size {}'.format(data.shape))
            x, y = process_dataset_file(data)
            print (x.shape)
            print (y.shape)
            X_test = np.vstack((X_test, x))
            y_test = np.concatenate([y_test, y])
        except KeyError:
            print( 'ERROR: Did not find {0} in zip file'.format(filename))
        

    print ("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))

    #obj = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    #f = open(os.path.join(target_filename), 'wb')
   # pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    #f.close()

    return X_train,y_train,X_val, y_val, X_test, y_test




def get_Opportunity_data(pamap2_dataset, output):
    
    X_train,y_train,X_val, y_val, X_test, y_test = generate_data(pamap2_dataset, output)
        
    print ('Done')
    return X_train,y_train,X_val, y_val, X_test, y_test