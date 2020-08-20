import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import sys
import time
import torch.optim as optim
import pickle
from DataLoader import CustomDataSet, CustomDataSetTest
from torch.utils.data import DataLoader
from Network import Network
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import csv
import pickle
import pandas as pd


# not called anymore. This method normalizes each attribute of a 2D matrix separately
'''
def normalize(data,ws):
    for k in range(len(data)):
        list1 = []
        temp = torch.tensor(list1)
        data_new = data[k]    
        data_new = torch.reshape(data_new,(200,30))
        data_new = data_new.cpu().detach().numpy()
        for i in range(data_new.shape[1]):
            max = np.max(data_new[:,i])
            min = np.min(data_new[:,i])
            for j in range(ws-1):
                data_new[j,i] = (data_new[j,i] - min)/(max - min)
        data_new = np.reshape(data_new,(1,200,30))
        data_new = torch.tensor(data_new).float()
        temp = torch.cat((temp, data_new), 0)
        #data_new = torch.tensor(data_new)        
        return data_new
'''

'''
Calculates precision and recall for all class using confusion matrix (cm)
returns list of precision and recall values
'''  
def get_precision_recall(targets, predictions):
        precision = torch.zeros((8))
        recall = torch.zeros((8))
        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)
        x = torch.ones(predictions.size())
        y = torch.zeros(predictions.size())

        #x = x.to('cuda', dtype=torch.long)
        #y = y.to('cuda', dtype=torch.long)

        for c in range(8):
            selected_elements = torch.where(predictions == c, x, y)
            non_selected_elements = torch.where(predictions == c, y, x)

            target_elements = torch.where(targets == c, x, y)
            non_target_elements = torch.where(targets == c, y, x)

            true_positives = torch.sum(target_elements * selected_elements)
            false_positives = torch.sum(non_target_elements * selected_elements)

            false_negatives = torch.sum(target_elements * non_selected_elements)

            try:
                precision[c] = true_positives.item() / float((true_positives + false_positives).item())
                recall[c] = true_positives.item() / float((true_positives + false_negatives).item())

            except:
                # logging.error('        Network_User:    Train:    In Class {} true_positives {} false_positives {} false_negatives {}'.format(c, true_positives.item(),
                #                                                                                                                              false_positives.item(),
                #                                                                                                                              false_negatives.item()))
                continue

        return precision, recall
        

    
def performance_metrics(cm):
    precision = []
    recall = []
    for i in range(len(cm)):
        tp = cm[i,i]
        fp = cm.sum(axis=0)[i] - cm[i,i]
        fn = cm.sum(axis=1)[i] - cm[i,i]
        precision.append(tp/(tp + fp))
        recall.append(tp/(tp+fn))
        print("Class",i," - precision", precision[i], "Recall",recall[i] )
    
    prec_avg = sum(precision)/len(precision)
    rec_avg = sum(recall)/len(recall)
    return precision, recall

'''
Create a list with max and min values for each channel for the input data
data - input in form [batch size, 1, window size, channels]
values - input argument having the max and min values of all channels from the previous iteration.
         Compares these previous values to current min and max values and updates
output - returns a list with max and min values for all channels

'''
def max_min_values(data, values):
    temp_values = []
    data = data.numpy()
    data = data.reshape(data.shape[0],data.shape[2], data.shape[3])
    
    temp_values = []
    for attr in range(data.shape[2]):
        attribute = []
        temp_max = np.max(data[:,:,attr])
        temp_min = np.min(data[:,:,attr])
        if (values[attr][0] > temp_max):
            attribute.append(values[attr][0])
        else:
            attribute.append(temp_max)
        if(values[attr][1] < temp_min):
            attribute.append(values[attr][1])
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
def normalize(data, min_max):
    
    data = data.numpy()
    data = data.reshape(data.shape[0],data.shape[2], data.shape[3])
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            data[i,:,j] = (data[i,:,j] - min_max[j][1])/(min_max[j][0] - min_max[j][1])
    data = data.reshape(data.shape[0],1,data.shape[1], data.shape[2])
    data = torch.tensor(data)
    return data

'''
returns a list of F1 score for all classes
'''
def F1_score(precision, recall):
    prec = precision.numpy()
    rec = recall.numpy()
    score = []
    for i in range(len(prec)):
        score.append(2*prec[i]*rec[i]/(prec[i] + rec[i]))
    return score
    
    
if __name__ == '__main__':
    
    if torch.cuda.is_available():  
          dev = "cuda:1" 
    else:  
          dev = "cpu"  
          
    device = torch.device(dev)
    config = {
        "NB_sensor_channels":126,
        "sliding_window_length":200,
        "filter_size":5,
        "num_filters":64,
        "network":"cnn",
        "output":"softmax",
        "num_classes":8,
        "reshape_input":False
        }


    ws=200
    features = 126
    accumulation_steps = 5
    trueValue = np.array([], dtype=np.int64)
    prediction = np.array([], dtype=np.int64)
    correct = 0
    total_loss = 0.0
    total_correct = 0
    epochs = 10
    batch_size = 50
    l = []
    tot_loss = 0
    accuracy = []
    
    model = Network(config)
    model = model.float()
    model = model.to(device)
    #model.load_state_dict(torch.load())
    #print("model loaded")
    noise = np.random.normal(0,1,(batch_size,1,ws,features))
    #noise = np.random.normal(0,1,(batch_size,features,ws))
    noise = torch.tensor(noise)
    noise = noise.float()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    model_path = '/data/sawasthi/data/MoCAP_data/model/model.pth'
    #model_path = 'S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/'
    path = '/data/sawasthi/data/MoCAP_data/trainData/'
    #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/'
    #path = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Train_data/"
    train_dataset = CustomDataSet(path)
    dataLoader_train = DataLoader(train_dataset, shuffle=True,
                                  batch_size=batch_size,
                                   num_workers=0,
                                   pin_memory=True,
                                   drop_last=True)
  
    print("preparing data for normalisation")
    
    # Initialise values list to store max and min values across all channels
    value = []
    for k in range(999):
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
        
    # Test data    
    path = '/data/sawasthi/data/MoCAP_data/testData/'
    #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/'
    #path = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Test_data/"
    test_dataset = CustomDataSet(path)
    dataLoader_test = DataLoader(test_dataset, shuffle=False,
                                  batch_size=batch_size,
                                   num_workers=0,
                                   pin_memory=True,
                                   drop_last=True)
        
    for b, harwindow_batched in enumerate(dataLoader_test):
        data_x = harwindow_batched["data"]
        data_x.to(device)
        value = max_min_values(data_x,value)
         
    print('Start Training')
    correct = 0
    counter = 0 
    total_loss = 0
    n_classes = 8
    for e in range(epochs):
          print("next epoch")
          #loop per batch:
          for b, harwindow_batched in enumerate(dataLoader_train):
              
              train_batch_v = harwindow_batched["data"]
              train_batch_l = harwindow_batched["label"][:, 0]
              #train_batch_v.to(device)
              train_batch_l = train_batch_l.to(device)
              train_batch_v = normalize(train_batch_v, value)
              train_batch_v = train_batch_v.float()
              train_batch_v = train_batch_v + noise
              train_batch_v = train_batch_v.to(device)
              #print(train_batch_v.device)
              out = model(train_batch_v)
              
              train_batch_l = train_batch_l.long()
              #loss = criterion(out.view(-1, n_classes), train_y.view(-1))
              loss = criterion(out,train_batch_l)*(1/accumulation_steps)
              predicted_classes = torch.argmax(out, dim=1).type(dtype=torch.LongTensor)
              predicted_classes = predicted_classes.to(device)
              correct += torch.sum(train_batch_l == predicted_classes)
              counter += out.size(0)
              
              loss.backward()
              if (b + 1) % accumulation_steps == 0:   
                optimizer.step()
                # zero the parameter gradients
                optimizer.zero_grad()
              print(' loss: ', loss.item(), 'accuracy in percent',100.*correct.item()/counter)
                      
              #lo, correct = Training(train_batch_v, train_batch_l, noise, model_path, batch_size, tot_loss, accumulation_steps)
              total_loss += loss.item()
              total_correct += correct
          l.append(total_loss/((e+1)*(b + 1)))
          accuracy.append(100*correct/((e+1)*(b + 1)*batch_size))
    
    print('Finished Training')
    ep = list(range(1,e+2))   
    plt.subplot(1,2,1)
    plt.title('epoch vs loss')
    plt.plot(ep,l)
    plt.subplot(1,2,2)
    plt.title('epoch vs accuracy')
    plt.plot(ep,accuracy)
    plt.savefig('/data/sawasthi/data/MoCAP_data/results/result.png') 
    #plt.savefig('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/result.png') 
    #plt.savefig('S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/result.png')
    
    print('Start Testing')
    total = 0.0
    correct = 0.0
    with torch.no_grad():
            
        for b, harwindow_batched in enumerate(dataLoader_test):
            test_batch_v = harwindow_batched["data"]
            test_batch_l = harwindow_batched["label"][:, 0]
            test_batch_v = normalize(test_batch_v, value)
            test_batch_v = test_batch_v.float()
            test_batch_v = test_batch_v.to(device)
            test_batch_l = test_batch_l.to(device)
            
            out = model(test_batch_v)
            #print("Next Batch result")
            predicted_classes = torch.argmax(out, dim=1).type(dtype=torch.LongTensor)
            #predicted = Testing(test_batch_v, test_batch_l)
            trueValue = np.concatenate((trueValue, test_batch_l))
            prediction = np.concatenate((prediction,predicted_classes))
            total += test_batch_l.size(0) 
            test_batch_l = test_batch_l.long()
            predicted_classes = predicted_classes.to(device)
            correct += (predicted_classes == test_batch_l).sum().item()
            #counter = out.view(-1, n_classes).size(0)
        
    print('\nTest set:  Percent Accuracy: {:.4f}\n'.format(100. * correct / total))
        
    cm = confusion_matrix(trueValue, prediction)
    print(cm)
    #precision, recall = performance_metrics(cm)
    precision, recall = get_precision_recall(trueValue, prediction)
    F1 = F1_score(precision, recall)
    print("precision", precision)
    print("recall", recall)
    print("F1 score", F1)
    
    print('Finished Testing')
    #with open('S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/result.csv', 'w', newline='') as myfile:
    #with open('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/result.csv', 'w', newline='') as myfile:
    with open('/data/sawasthi/data/MoCAP_data/results/result.csv', 'w') as myfile:
         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
         wr.writerow(accuracy)
         wr.writerow(l)
             
