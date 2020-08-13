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


cuda = "True"
torch.manual_seed(1111)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
def getTrainData(path,batch_size):
        
    
    train_data = []
    #path = '/data/sawasthi/data/trainData/'
    #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/'
    #while folder_counter < 10:
        #some code to get path_to_imgs which is the location of the image folder
    train_dataset = CustomDataSet(path)
    #all_datasets.append(train_dataset)
    
        
    #final_dataset = torch.utils.data.ConcatDataset(train_dataset)
    train_loader = DataLoader(train_dataset, shuffle=False,
                                      batch_size=batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
    for idx, input_seq  in enumerate(train_loader):
        train_data.append(input_seq)
        
    return train_data



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

def Training(train_x, train_y, noise, model_path,batch_size, total_loss, accumulation_steps):
          
    #global i, total_loss, counter
    index = 0
    correct = 0
    counter = 0 
    model.train()
    total_loss = 0
    n_classes = 8
    
    train_x = train_x.float()
    train_x = train_x + noise
    #x = np.reshape(x,(batch_size,ws,features))
    #x = np.reshape(x,(batch_size,features,ws))
    #out = model(x.unsqueeze(1).contiguous())
    out = model(train_x)
    #out = model(x)
    train_y = train_y.long()
    loss = criterion(out.view(-1, n_classes), train_y.view(-1))
    
    pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
    correct += pred.eq(train_y.data.view_as(pred)).cpu().sum().item()
    counter += out.view(-1, n_classes).size(0)
    
    loss.backward()
    if (index + 1) % accumulation_steps == 0:   
      optimizer.step()
      # zero the parameter gradients
      optimizer.zero_grad()
    #optimizer.step()
    #total_loss += loss.item()
    
    print(' loss: ', loss.item(), 'accuracy in percent',100.*correct/counter)
    
    
   
    #torch.save(model.state_dict(), model_path)
    #print(index)
    
    counter+=1
    return loss.item(), correct

def Testing(test_x, test_y):
     with torch.no_grad():
        
        test_x = test_x.float()
        out = model(test_x)
        print("Next Batch result")
        _,predicted = torch.max(out, 1)
            
     return predicted

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


if __name__ == '__main__':
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
    model = Network(config)
    model = model.float()
    trueValue = np.array([], dtype=np.int64)
    prediction = np.array([], dtype=np.int64)
    correct = 0
    total_loss = 0.0
    total_correct = 0
    epochs =6
    batch_size = 200
    l = []
    tot_loss = 0
    temp = []
    accuracy = []
    
    #model.load_state_dict(torch.load())
    #print("model loaded")
    noise = np.random.normal(0,1,(batch_size,1,ws,features))
    #noise = np.random.normal(0,1,(batch_size,features,ws))
    noise = torch.tensor(noise)
    noise = noise.float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
              train_batch_v = train_batch_v.float()
              train_batch_v = train_batch_v + noise
              
              out = model(train_batch_v)
              
              train_batch_l = train_batch_l.long()
              #loss = criterion(out.view(-1, n_classes), train_y.view(-1))
              loss = criterion(out,train_batch_l)*(1/accumulation_steps)
              pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
              correct += pred.eq(train_batch_l.data.view_as(pred)).cpu().sum().item()
              counter += out.view(-1, n_classes).size(0)
              
              loss.backward()
              if (b + 1) % accumulation_steps == 0:   
                optimizer.step()
                # zero the parameter gradients
                optimizer.zero_grad()
              print(' loss: ', loss.item(), 'accuracy in percent',100.*correct/counter)
              
             
 
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
    path = '/data/sawasthi/data/MoCAP_data/testData/'
    #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/'
    #path = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Test_data/"
    test_dataset = CustomDataSet(path)
    dataLoader_test = DataLoader(test_dataset, shuffle=False,
                                  batch_size=batch_size,
                                   num_workers=0,
                                   pin_memory=True,
                                   drop_last=True)
    total = 0.0
    with torch.no_grad():
            
        for b, harwindow_batched in enumerate(dataLoader_test):
            test_batch_v = harwindow_batched["data"]
            test_batch_l = harwindow_batched["label"][:, 0]
            test_batch_v = test_batch_v.float()
            out = model(test_batch_v)
            #print("Next Batch result")
            _,predicted = torch.max(out, 1)
            #predicted = Testing(test_batch_v, test_batch_l)
            trueValue = np.concatenate((trueValue, test_batch_l))
            prediction = np.concatenate((prediction,predicted))
            total += test_batch_l.size(0) 
            test_batch_l = test_batch_l.long()
            correct += (predicted == test_batch_l).sum().item()
            #counter = out.view(-1, n_classes).size(0)
        
    print('\nTest set:  Percent Accuracy: {:.4f}\n'.format(100. * correct / total))
        
    cm = confusion_matrix(trueValue, prediction)
    print(cm)
    #precision, recall = performance_metrics(cm)
    precision, recall = get_precision_recall(trueValue, prediction)
    print(precision)
    print(recall)
    
    print('Finished Testing')
    #with open('S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/result.csv', 'w', newline='') as myfile:
    #with open('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/result.csv', 'w', newline='') as myfile:
    with open('/data/sawasthi/data/MoCAP_data/results/result.csv', 'w') as myfile:
         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
         wr.writerow(accuracy)
         wr.writerow(l)
             
