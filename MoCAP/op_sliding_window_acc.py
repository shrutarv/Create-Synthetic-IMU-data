import pandas as pd
import sys
import numpy as np
import os
import pickle
from sliding_window import sliding_window
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



def example_creating_windows_file(k, folder_name, data_x, labels):
        # Sliding window approach

    print("Starting sliding window")
   
    X, y, y_all = opp_sliding_window(data_x, labels.astype(int),
                                     sliding_window_length,
                                     sliding_window_step, label_pos_end = False)
 
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
        dir = data_dir + "seq_" + "_" + str(k) + "_" + str(counter_seq) + ".pkl"
        obj = {"data" : seq, "label" : y[f], "labels" : y_all[f]}
        #f = open(os.path.join(dir, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
        f = open(dir, 'wb')
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        counter_seq += 1
        print("dumping")
        f.close()
 
def max_min_values(dat, values):
    temp_values = []
    for i in range(dat.shape[1]):
        attribute = []
        temp_max = np.max(dat[:,i])
        temp_min = np.min(dat[:,i])
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

def normalize(data_n, min_max, string):
    #print(len(min_max), len(min_max[0]))
    
    for j in range(len(data_n[0])):
        data_n[:,j] = (data_n[:,j] - min_max[j][0])/(min_max[j][1] - min_max[j][0]) 
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
    
    #ws = (100,31)
    ws = (200,134)  #for MoCAP
    ss = (25,134)     #for MoCAP
    #ss = (25,31)
    sliding_window_length = 200   # for MoCAP
    #sliding_window_length = 100    
    sliding_window_step = 25
    #df =  pd.read_csv('S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Train_data_csv/train10.csv',chunksize=1000)
    df = pd.read_csv('/data/sawasthi/data/MoCAP_data/train_csv/train.csv')
    data = df.values
    sampled_time = np.linspace(0,len(data)/200,len(data))
    y_sampled = np.zeros((len(sampled_time),1))
    for i in range(data.shape[1]-1):
        #for index in range(12,len(data[0])*up-12):
                
            #data_new = data[index-12:index+12,:]   
         
         #f = sp.interp1d(data[:,0],data[:,i], kind='linear', fill_value="extrapolate")
        
         #sampled_data = f(x_sampled)
         #acc = derivative(f, x_sampled)
         #acc = np.diff(sampled_data,2)
         resample = sp.splrep(sampled_time,data[:,i])
         acc = sp.splev(sampled_time,resample, der=2)
         y_sampled = np.concatenate((y_sampled,np.reshape(acc,(len(acc),1))),axis=1)
    data_new = y_sampled[:,1:]
    # creating labels
    
    #data_dir = "/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/"
    #df = pd.read_csv('S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/train_data25_39.csv')
    data_dir =  "/data/sawasthi/data/MoCAP_data/trainData_acc/"
    #data_dir = 'S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/pkl/'
    labels = pd.read_csv('/data/sawasthi/data/MoCAP_data/train_csv/trainLabels.csv')
    labels = labels.values
    lab = np.zeros((len(labels),20), dtype=int)
    lab[:,0] = labels
    #X = data[:,1:31]
    X = data_new
    k = 0
    example_creating_windows_file(k,data_dir, X,lab)
    print("train data pickled")
    
    '''
    trainData = np.empty([1, 126])
    # training set : S01, S02,S03,S04,S07,S08,S09,S10
    # validation set : S05,S11,S12
    # test set : S06,13,14
    data_dir =  "/data/sawasthi/data/MoCAP_data/validationData/"
    #data_dir = "/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/"
    #data_dir = "S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/"
    #data_dir = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Train_data/"
    #for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1)):
    FolderList = ['P06','P13','P14']
    #    print (np.shape(i[:,0]))
    #folder_name = "P09"
    for folder_name in FolderList:
        
        FileList_y = []
        #os.chdir('/vol/actrec/DFG_Project/2019/Mbientlab/recordings_2019/07_IMU_synchronized_annotated/' + folder_name)
        os.chdir("/vol/actrec/DFG_Project/2019/MoCap/recordings_2019/14_Annotated_Dataset/" + folder_name)
        #os.chdir("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/")
        #os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/" + folder_name)
       # os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/OMoCap data/" + folder_name)
        FileList_y = glob.glob('*labels.csv')
        #os.chdir('/vol/actrec/DFG_Project/2019/Mbientlab/recordings_2019/07_IMU_synchronized_annotated/P13')
        #List = glob.glob('*labels.csv')
        #FileList_y = FileList_y + List
                
        FileList_x = []
        #os.chdir('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13')
        #os.chdir('/vol/actrec/DFG_Project/2019/Mbientlab/recordings_2019/07_IMU_synchronized_annotated/P14')
        #os.chdir("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/")
        FileList_x = glob.glob('*.csv')
        #os.chdir('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S14')
        #os.chdir('/vol/actrec/DFG_Project/2019/Mbientlab/recordings_2019/07_IMU_synchronized_annotated/P14')
        set_x = set(FileList_x)
        set_y = set(FileList_y)
        FileList_x = list(set_x - set_y)
        FileList_x.sort()
        FileList_y.sort()
        k = 0 
        
        for i,j in zip(FileList_x, FileList_y):
            k += 1
            data_y = pd.read_csv("/vol/actrec/DFG_Project/2019/MoCap/recordings_2019/14_Annotated_Dataset/" + folder_name + "/" + j) 
            #data_y = pd.read_csv("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/"+j)
            #data_y = pd.read_csv("S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/" + folder_name+ "/" + j) 
            #data_y = pd.read_csv("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/OMoCap data/" + folder_name + "/" + j) 
            data_y = data_y.values
            labels = data_y[:,0]
            data_x = pd.read_csv("/vol/actrec/DFG_Project/2019/MoCap/recordings_2019/14_Annotated_Dataset/"+ folder_name + "/" + i) 
            #data_x = pd.read_csv("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/"+i)
            #data_x = pd.read_csv("S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/" + folder_name +"/" + i)
            #data_x = pd.read_csv("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/OMoCap data/" + folder_name + "/" + i)
            data_x = data_x.values
            data_x = np.delete(data_x,np.s_[68:74], axis=1)
            data_x = data_x[:,2:128]
            trainData = np.concatenate((trainData,data_x))
    t = np.zeros((1,trainData.shape[1]))
    trainData = np.concatenate((t,trainData))      
    np.savetxt("/data/sawasthi/data/MoCAP_data/train_csv/test.csv", trainData, delimiter=',')
        #example_creating_windows_file(k, folder_name, data_x, labels)
        #if(k == 2):
          #  break
      
    
      # Save max min values
    value = []
    for k in range(200):
        temp_list = []
        maxim = -9999
        minim = 9999
        temp_list.append(maxim)
        temp_list.append(minim)
        value.append(temp_list) 
    data_y = pd.read_csv("/data/sawasthi/data/MoCAP_data/train_csv/train.csv") 
    #data_y = pd.read_csv("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Train_data_csv/" + i) 
    data_y = data_y.values
    value = max_min_values(data_y, value)
    with open("/data/sawasthi/data/MoCAP_data/train_csv/value.csv", 'w') as f:
        fc = csv.writer(f, lineterminator='\n')
        fc.writerow(["max","min"])
        fc.writerows(value)
    
    
    os.chdir("/data/sawasthi/data/MoCAP_data/train_csv/")
    FileList = glob.glob('*.csv')
    for i in FileList:
        
        #data_y = pd.read_csv("/vol/actrec/DFG_Project/2019/MoCap/recordings_2019/14_Annotated_Dataset/" + folder_name + "/" + j) 
        #data_y = pd.read_csv("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/"+j)
        data_y = pd.read_csv("/data/sawasthi/data/MoCAP_data/train_csv/" + i) 
        data_y = pd.read_csv("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Train_data_csv/train.csv") 
        data_y = data_y.values
        value = max_min_values(data_y, value)
    np.savetxt("/data/sawasthi/data/MoCAP_data/train_csv/value.csv", value, delimiter=',')   
    
    
    value = pd.read_csv("/data/sawasthi/data/MoCAP_data/train_csv/value.csv") 
    value = value.values
    data = pd.read_csv("/data/sawasthi/data/MoCAP_data/train_csv/train.csv") 
    print("read training data")
    data = data.values
    data_norm = normalize(data,value,"train")
    print("normalized")
    data_norm = np.concatenate((np.reshape(data[:,0],(len(data[:,0]),1)),data_norm),axis=1)
    
    t = np.zeros((1,data_norm.shape[1]))
    data_norm = np.concatenate((t,data_norm))
    np.savetxt("/data/sawasthi/data/MoCAP_data/train_csv/train_normal.csv", data_norm, delimiter=',')
    '''
    '''
    trainData = np.empty([1, 1])
    # training set : S01, S02,S03,S04,S07,S08,S09,S10
    # validation set : S05,S11,S12
    # test set : S06,13,14
     #for i in sliding_window(data_y,(ws,data_y.shape[1]),(ss,1)):
    FolderList = ['P06','P13','P14']
    #    print (np.shape(i[:,0]))
    #folder_name = "P09"
    for folder_name in FolderList:
        
        FileList_y = []
        #os.chdir('/vol/actrec/DFG_Project/2019/Mbientlab/recordings_2019/07_IMU_synchronized_annotated/' + folder_name)
        os.chdir("/vol/actrec/DFG_Project/2019/MoCap/recordings_2019/14_Annotated_Dataset/" + folder_name)
        #os.chdir("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/")
        #os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/" + folder_name)
        #os.chdir("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/OMoCap data/" + folder_name)
        FileList_y = glob.glob('*labels.csv')
        #os.chdir('/vol/actrec/DFG_Project/2019/Mbientlab/recordings_2019/07_IMU_synchronized_annotated/P13')
        #List = glob.glob('*labels.csv')
        #FileList_y = FileList_y + List
                
        
        FileList_y.sort()
        k = 0 
        
        for j in FileList_y:
            k += 1
            data_y = pd.read_csv("/vol/actrec/DFG_Project/2019/MoCap/recordings_2019/14_Annotated_Dataset/" + folder_name + "/" + j) 
            #data_y = pd.read_csv("/media/shrutarv/Drive1/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/S13/"+j)
            #data_y = pd.read_csv("S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/" + folder_name+ "/" + j) 
            #data_y = pd.read_csv("S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/OMoCap data/" + folder_name + "/" + j) 
            data_y = data_y.values
            labels = data_y[:,0]
           
            trainData = np.concatenate((trainData,np.reshape(labels,(len(labels),1))))
         
    np.savetxt("/data/sawasthi/data/MoCAP_data/train_csv/testLabels.csv", trainData, delimiter=',')
    '''