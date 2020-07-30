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

cuda = "True"
torch.manual_seed(1111)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
def getTrainData():
        
    all_datasets = []
    train_data = []
    
    path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/'
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

def getTrainDataLabels():
        
    all_datasets = []
    train_data = []
    
    path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/'
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


ws = 200
with open('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/seq_000001.pkl', 'rb') as f:
    pk = pickle.load(f)
noise = np.random.normal(0,1,200)
train = pk['data']
train = np.transpose(train)
train = np.reshape(train,(30,200))
for i in range(len(train)-1):
    max = np.max(train[i,:])
    min = np.min(train[i,:])
    for j in range(ws-1):
        train[i,j] = (train[i,j] - min)/(max - min)

trainset = torch.tensor(train)
y = pk['label']
y = y[0]
y = torch.tensor(y, dtype=torch.long)

n_classes = 8  # Digits 0 - 9
n_train = 10000
n_test = 1000

#print(args)
#print("Preparing data...")
#train_x, train_y = data_generator(T, seq_len, n_train)
#test_x, test_y = data_generator(T, seq_len, n_test)

channel_sizes = 30
#channel_sizes = [args.nhid] * args.levels

model = TemporalConvNet(kernel_size, dropout)
model = model.float()

#model.cuda()
#train_x = train_x.cuda()
#train_y = train_y.cuda()
#test_x = test_x.cuda()
#test_y = test_y.cuda()

#criterion = nn.CrossEntropyLoss()
#lr = args.lr
#optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(test_x.unsqueeze(1).contiguous())
        loss = criterion(out.view(-1, n_classes), test_y.view(-1))
        pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
        correct = pred.eq(test_y.data.view_as(pred)).cpu().sum()
        counter = out.view(-1, n_classes).size(0)
        print('\nTest set: Average loss: {:.8f}  |  Accuracy: {:.4f}\n'.format(
            loss.item(), 100. * correct / counter))
        return loss.item()

train_x = getTrainData()
train_y = getTrainDataLabels()
train_y = torch.tensor(train_y)

global batch_size, seq_len, iters, epochs
model.train()
total_loss = 0
start_time = time.time()
correct = 0
counter = 0
for i in range(len(train_x)):
    x = train_x[i]
    y = train_y[i]
    x = np.reshape(x,(30,200))
    optimizer.zero_grad()
    x = x.float()
    
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
        print(' loss: ' (total_loss / 50))
        total_loss = 0.0
print('Finished Training')
   
     
