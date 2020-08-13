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
from Temp_Block import TemporalConvNet
from sklearn.metrics import confusion_matrix

cuda = "True"
torch.manual_seed(1111)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
def getTrainData(path):
        
    all_datasets = []
    train_data = []
    #path = '/data/sawasthi/data/trainData/'
    #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/'
    #while folder_counter < 10:
        #some code to get path_to_imgs which is the location of the image folder
    train_dataset = CustomDataSet(path)
    all_datasets.append(train_dataset)
    
        
    final_dataset = torch.utils.data.ConcatDataset(all_datasets)
    train_loader = DataLoader(final_dataset, shuffle=False,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
    for idx, input_seq  in enumerate(train_loader):
        train_data.append(input_seq)
        
    return train_data

def getTrainDataLabels(path):
        
    all_datasets = []
    train_data = []
    #path = '/data/sawasthi/data/testData/'
    #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/'
    #while folder_counter < 10:
        #some code to get path_to_imgs which is the location of the image folder
    train_dataset = CustomDataSetTest(path)
    all_datasets.append(train_dataset)
    
        
    final_dataset = torch.utils.data.ConcatDataset(all_datasets)
    train_loader = DataLoader(final_dataset, shuffle=False,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
    for idx, input_seq  in enumerate(train_loader):
        train_data.append(input_seq)
        
    return train_data

def normalize(data,ws):
    data_new = np.reshape(data,(30,200))
    data_new = data_new.cpu().detach().numpy()
    for i in range(len(data_new)):
        max = np.max(data_new[i,:])
        min = np.min(data_new[i,:])
        for j in range(ws-1):
            data_new[i,j] = (data_new[i,j] - min)/(max - min)
            
    data_new = torch.tensor(data_new)        
    return data_new

def Testing(test_x, test_y):
    trueValue = []
    prediction = []
    with torch.no_grad():
        for i in range(len(test_x)):
                
            x = test_x[i]
            y = test_y[i]
            x = torch.tensor(x)
            x = np.reshape(x,(30,200))
            x = x.float()
            out = model(x.unsqueeze(1).contiguous())
            _,predicted = torch.max(out, 1)
            loss = criterion(out.view(-1, n_classes), y.view(-1))
            pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
            trueValue.append(y.tolist())
            prediction.append(predicted.tolist())
            flat_list_pred = [item for sublist in prediction for item in sublist]
            flat_list_true = [item for sublist in trueValue for item in sublist]
            print("predicted list")
            unique(flat_list_pred)
            print("true list")
            unique(flat_list_true)
            correct = pred.eq(y.data.view_as(pred)).cpu().sum()
            counter = out.view(-1, n_classes).size(0)
            print('\nTest set: Average loss: {:.8f}  |  Accuracy: {:.4f}\n'.format(
                loss.item(), 100. * correct / counter))
        cm = confusion_matrix(trueValue, prediction)
        print(cm)
        precision, recall = performance_metrics(cm)
        
        return loss.item()
    
def unique(list1): 
      
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set)) 
    for x in unique_list: 
        print(x)
      
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

def Training(train_x, train_y, noise, model_path):
        
    global batch_size, seq_len, iters, epochs
    model.train()
    total_loss = 0

    for i in range(len(train_x)):
        x = train_x[i]
        x = normalize(x, ws)
        y = train_y[i]
        x = np.reshape(x,(30,200))
        optimizer.zero_grad()
        x = x.float()
        x = x + noise
        out = model(x.unsqueeze(1).contiguous())
        #out = model(x)
        loss = criterion(out.view(-1, n_classes), y.view(-1))
        #pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
        #correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        #counter += out.view(-1, n_classes).size(0)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 50 == 49:    # print every 2000 mini-batches
            print(' loss: ', (total_loss / 50))
            total_loss = 0.0
    #torch.save(model.state_dict(), model_path)
    print('Finished Training')
    
ws = 200
#with open('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/seq_S13_29_457.pkl', 'rb') as f:
 #   pk = pickle.load(f)
#train = pk['data']
#train = np.transpose(train)
#train = np.reshape(train,(30,200))

#trainset = torch.tensor(train)
#y = pk['label']
#y = y[0]
#y = torch.tensor(y, dtype=torch.long)

n_classes = 8  # Digits 0 - 9
n_train = 10000
n_test = 1000

#print(args)
#print("Preparing data...")
#train_x, train_y = data_generator(T, seq_len, n_train)
#test_x, test_y = data_generator(T, seq_len, n_test)

channel_sizes = 30
#channel_sizes = [args.nhid] * args.levels
kernel_size = 5
dropout = 0.2
model = TemporalConvNet(kernel_size, dropout)
model = model.float()
#model.load_state_dict(torch.load())
#print("model loaded")
#model.cuda()
#train_x = train_x.cuda()
#train_y = train_y.cuda()
#test_x = test_x.cuda()
#test_y = test_y.cuda()

#criterion = nn.CrossEntropyLoss()
#lr = args.lr
#optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model_path = '/data/sawasthi/data/model/model.pth'
#model_path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/'
path = '/data/sawasthi/data/trainData/'
#path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/'
train_x = getTrainData(path)
train_y = getTrainDataLabels(path)
#train_y = torch.tensor(train_y)
noise = np.random.normal(0,1,(30,200))
noise = torch.tensor(noise)
noise = noise.float()
Training(train_x, train_y, noise, model_path)
#path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/'
path = '/data/sawasthi/data/testData/'
test_x = getTrainData(path)
test_y = getTrainDataLabels(path)
Testing(test_x, test_y)
       
         
