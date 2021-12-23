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
import os
import random
import platform
import pandas as pd
import logging
from logging import handlers

# not called anymore. This method normalizes each attribute of a 2D matrix separately


def setup_experiment_logger(logging_level=logging.ERROR, filename=None):
    # set up the logging
    logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
    if filename != None:
        logging.basicConfig(filename=filename,level=logging.DEBUG,
                            format=logging_format,
                            filemode='w')
    else:
        logging.basicConfig(level=logging_level,
                            format=logging_format,
                            filemode='w')
        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)   


    return


# not called anymore. This method normalizes each attribute of a 2D matrix separately

'''
Calculates precision and recall for all class using confusion matrix (cm)
returns list of precision and recall values
'''  
def get_precision_recall(targets, predictions):
        precision = torch.zeros((config['num_classes']))
        recall = torch.zeros((config['num_classes']))
        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)
        x = torch.ones(predictions.size())
        y = torch.zeros(predictions.size())

        #x = x.to('cuda', dtype=torch.long)
        #y = y.to('cuda', dtype=torch.long)

        for c in range(len(precision)):
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



def F1_score(targets, preds, precision, recall):
        # Accuracy
        targets = torch.tensor(targets)
        #predictions = torch.argmax(preds, dim=1)
        #precision, recall = get_precision_recall(targets, preds)
        proportions = torch.zeros(config['num_classes'])

        for c in range(config['num_classes']):
            proportions[c] = torch.sum(targets == c).item() / float(targets.size()[0])
        
        multi_pre_rec = precision * recall
        sum_pre_rec = precision + recall
        multi_pre_rec[torch.isnan(multi_pre_rec)] = 0
        sum_pre_rec[torch.isnan(sum_pre_rec)] = 0

        # F1 weighted
        weighted_f1 = proportions * (multi_pre_rec / sum_pre_rec)
        weighted_f1[torch.isnan(weighted_f1)] = 0
        F1_weighted = torch.sum(weighted_f1) * 2

        # F1 mean
        f1 = multi_pre_rec / sum_pre_rec
        f1[torch.isnan(f1)] = 0
        F1_mean = torch.sum(f1) * 2 / config['num_classes']

        return F1_weighted, F1_mean
    
def metrics(predictions, true):
    counter = 0.0
    correct = 0.0
    predicted_classes = torch.argmax(predictions, dim=1).type(dtype=torch.LongTensor)
    predicted_classes = predicted_classes.to(device)
    
    correct = torch.sum(true == predicted_classes)
    counter = true.size(0)
    accuracy = 100.*correct.item()/counter
    return accuracy, correct
    
def validation(dataLoader_validation,device):
    total = 0.0
    correct = 0.0
    trueValue = np.array([], dtype=np.int64)
    prediction = np.array([], dtype=np.int64)
    total_loss = 0.0
    with torch.no_grad():
            
        for b, harwindow_batched in enumerate(dataLoader_validation):
            test_batch_v = harwindow_batched["data"]
            test_batch_l = harwindow_batched["label"][:, 0]
            #test_batch_v = normalize(test_batch_v, value,"test")
            test_batch_v = test_batch_v.float()
            test_batch_v = test_batch_v.to(device)
            test_batch_l = test_batch_l.to(device)
            test_batch_l = test_batch_l.long()
            out = model(test_batch_v)
            loss = criterion(out,test_batch_l)
            #print("Next Batch result")
            predicted_classes = torch.argmax(out, dim=1).type(dtype=torch.LongTensor)
            #predicted = Testing(test_batch_v, test_batch_l)
            trueValue = np.concatenate((trueValue, test_batch_l.cpu()))
            prediction = np.concatenate((prediction,predicted_classes))
            total += test_batch_l.size(0) 
            test_batch_l = test_batch_l.long()
            predicted_classes = predicted_classes.to(device)
            correct += (predicted_classes == test_batch_l).sum().item()
            total_loss += loss.item()
            #counter = out.view(-1, n_classes).size(0)
        
    print('\nValidation set:  Percent Validation Accuracy: {:.4f}\n'.format(100. * correct / total))
    return (100. * correct / total, total_loss/(b+1))
     


def training(dataLoader_train, dataLoader_validation, device,flag):
    print('Start Training')
    correct = 0
    total_loss = 0
    total_correct = 0
    best_acc = 0.0
    validation_loss = []
    validation_acc = []
    accuracy = []
    l = []
    for e in range(epochs):
          
          model.train()
          logging.info('epoch {}'.format(e))
          #loop per batch:
          
          for b, harwindow_batched in enumerate(dataLoader_train):
             
              train_batch_v = harwindow_batched["data"]
              train_batch_l = harwindow_batched["label"][:, 0]
              train_batch_all = harwindow_batched["labels"][:,:,:]
              
              train_batch_v.to(device)
              train_batch_l = train_batch_l.to(device)
              
              train_batch_v = train_batch_v.float()
              train_batch_v = train_batch_v.to(device)
              noise = normal.sample((train_batch_v.size()))
              noise = noise.reshape(train_batch_v.size())
              noise = noise.to(device, dtype=torch.float)

              train_batch_v = train_batch_v + noise
              
              #print(train_batch_v.device)
              out = model(train_batch_v)
              train_batch_l = train_batch_l.long()
              #loss = criterion(out.view(-1, n_classes), train_y.view(-1))
              loss = criterion(out,train_batch_l)/accumulation_steps
              #predicted_classes = torch.argmax(out, dim=1).type(dtype=torch.LongTensor)
              #predicted_classes = predicted_classes.to(device)
              
              #correct += torch.sum(train_batch_l == predicted_classes)
              #counter += out.size(0)
             # a = list(model.parameters())[0].clone() 
              loss.backward()
              
              if (b + 1) % accumulation_steps == 0:   
                optimizer.step()
                # zero the parameter gradients
                optimizer.zero_grad()
              #c = list(model.parameters())[0].clone()
              #print(torch.equal(a.data, c.data))
              acc, correct = metrics(out, train_batch_l)
              #print(' loss: ', loss.item(), 'accuracy in percent',acc)
              #lo, correct = Training(train_batch_v, train_batch_l, noise, model_path, batch_size, tot_loss, accumulation_steps)
              total_loss += loss.item()
              total_correct += correct
          
          model.eval()
          
          val_acc, val_loss =  validation(dataLoader_validation,device)
          validation_loss.append(val_loss)
          validation_acc.append(val_acc)
          if (val_acc >= best_acc):
              torch.save(model, model_path)
              
              print("model saved on epoch", e)
              best_acc = val_acc
          
          
          l.append(total_loss/((e+1)*(b + 1)))
          accuracy.append(100*total_correct/((e+1)*(b + 1)*batch_size))
         
          '''
          for param_group in optimizer.param_groups:
              print(param_group['lr'])        
              param_group['lr'] = lr_factor*param_group['lr']
          #scheduler.step(val_loss)
          '''
    if (flag):
                  
        print('Finished Training')
        ep = list(range(1,e+2))   
        plt.subplot(1,2,1)
        plt.title('epoch vs loss')
        plt.plot(ep,l, 'r', label='training loss')
        plt.plot(ep,validation_loss, 'g',label='validation loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.title('epoch vs accuracy')
        plt.plot(ep,accuracy,'r',label='training accuracy')
        plt.plot(ep,validation_acc, 'g',label='validation accuracy')
        plt.legend()
        plt.savefig('/data/sawasthi/Lara_motionminer/results/result_10_50.png') 
        #plt.savefig('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/result.png') 
        #plt.savefig('S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/result.png'

def Testing(config):
    total = 0.0
    correct = 0.0
    trueValue = np.array([], dtype=np.int64)
    prediction = np.array([], dtype=np.int64)
    model = torch.load(model_path)
    print("best model loaded")
    model.eval()
    with torch.no_grad():
            
        for b, harwindow_batched in enumerate(dataLoader_test):
            test_batch_v = harwindow_batched["data"]
            test_batch_l = harwindow_batched["label"][:, 0]
           # test_batch_v = normalize(test_batch_v, value,"test")
            test_batch_v = test_batch_v.float()
            test_batch_v = test_batch_v.to(device)
            test_batch_l = test_batch_l.to(device)
            
            out = model(test_batch_v)
            #print("Next Batch result")
            predicted_classes = torch.argmax(out, dim=1).type(dtype=torch.LongTensor)
            #predicted = Testing(test_batch_v, test_batch_l)
            trueValue = np.concatenate((trueValue, test_batch_l.cpu()))
            prediction = np.concatenate((prediction,predicted_classes))
            total += test_batch_l.size(0) 
            test_batch_l = test_batch_l.long()
            predicted_classes = predicted_classes.to(device)
            correct += (predicted_classes == test_batch_l).sum().item()
            #counter = out.view(-1, n_classes).size(0)
    logging.info('Test set:  Percent Accuracy: {:.4f}\n'.format(100. * correct / total))    
    cm = confusion_matrix(trueValue, prediction)
    test_acc = 100. * correct / total        
    
    
    logging.info('confusion matrix {}'.format(cm))
    #precision, recall = performance_metrics(cm)
    precision, recall = get_precision_recall(trueValue, prediction)
    F1_weighted, F1_mean = F1_score(trueValue, prediction, precision, recall)
   
    logging.info('precision {}'.format(precision))
    
    logging.info('recall {}'.format(recall))
    
    logging.info('F1 weighted {}'.format(F1_weighted))
    
    logging.info('F1 mean {}'.format(F1_mean))
    
    logging.info('Finished Validation')
    return F1_weighted, test_acc
  


if __name__ == '__main__':
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)
    setup_experiment_logger(logging_level=logging.DEBUG, filename= "/data/sawasthi/Lara_motionminer/logger_10_50.txt")
    
    print(":Python Platform {}".format(platform.python_version()))
    
    
    if torch.cuda.is_available():  
          dev = "cuda:0" 
    else:  
          dev = "cpu"  
          
    device = torch.device(dev)
    config = {
        "NB_sensor_channels":27,
        "sliding_window_length":100,
        "filter_size":5,
        "num_filters":64,
        "network":"cnn",
        "output":"softmax",
        "num_classes":8,
        "reshape_input":False
        }


    ws=100
    
    accumulation_steps = 5
    correct = 0
    total_loss = 0.0
    total_correct = 0
    epochs = 40
    batch_size = 1000
   
    flag = True
    iterations = 5
    weighted_F1_array = []
    test_acc_array = []
    for iter in range(iterations):
        
        l = []
        tot_loss = 0
        accuracy = []
        learning_rate = 0.00001
        print("epoch: ",epochs,"batch_size: ", batch_size,"accumulation steps: ",accumulation_steps,"ws: ",ws, "learning_rate: ",learning_rate)
            
        #df = pd.read_csv('/data/sawasthi/Thesis--Create-Synthetic-IMU-data/MoCAP/norm_values.csv')
        #df = pd.read_csv('S:/MS A&R/4th Sem/Thesis/Github/Thesis- Create Synthetic IMU data/Lara_IMU/norm_IMU.csv')
        #value = df.values.tolist()
        #print(len(df),len(value), len(value[0]))
        model = Network(config)
        model = model.float()
        model = model.to(device)
        #model.load_state_dict(torch.load())
        #print("model loaded")   # 
        normal = torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([0.01]))
        #noise = noise.float()
        
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9,weight_decay=0.0005, momentum=0.9)
        #lmbda = lambda epoch: 0.95
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.95)
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        model_path = '/data/sawasthi/Lara_motionminer/model/Laramm_model_5_100.pth'
        #model_path = 'S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/model.pth'
        path = '/data/sawasthi/Lara_motionminer/trainData_5_100/'
        #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows2/'
        #path = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Train_data/"
        train_dataset = CustomDataSet(path)
        dataLoader_train = DataLoader(train_dataset, shuffle=True,
                                      batch_size=batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
      
       
        # Validation data    
        path = '/data/sawasthi/Lara_motionminer/validationData_5/'
        #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/'
        #path = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Test_data/"
        validation_dataset = CustomDataSet(path)
        dataLoader_validation = DataLoader(validation_dataset, shuffle=False,
                                      batch_size=batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
        
        # Test data    
        path = '/data/sawasthi/Lara_motionminer/testData_5/'
        #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/'
        #path = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Test_data/"
        test_dataset = CustomDataSet(path)
        dataLoader_test = DataLoader(test_dataset, shuffle=False,
                                      batch_size=batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
        
        
        training(dataLoader_train, dataLoader_validation,device,flag)
        print('Start Testing')
        WF, TA = Testing(config)
        flag = False
        #with open('S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/result.csv', 'w', newline='') as myfile:
        #with open('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/result.csv', 'w', newline='') as myfile:
        weighted_F1_array.append(WF)
        test_acc_array.append(TA)
        
    logging.info("Mean Weighted F1 score after 5 runs is {}".format(np.mean(weighted_F1_array)))
    sys.stdout.write("Mean Weighted F1 score after 5 runs is {}".format(np.mean(weighted_F1_array)))
    logging.info("Standard deviation of Weighted F1 score after 5 runs is {}".format(np.std(weighted_F1_array)))
    sys.stdout.write("Standard deviation of Weighted F1 score after 5 runs is {}".format(np.std(weighted_F1_array)))
    logging.info("Mean Test accuracy score after 5 runs is {}".format(np.mean(test_acc_array)))
    sys.stdout.write("Standard deviation of Test accuracy score after 5 runs is {}".format(np.std(test_acc_array)))
    logging.info('Standard deviation of Test accuracy score after 5 runs is {}'.format(np.std(test_acc_array)))
                 
