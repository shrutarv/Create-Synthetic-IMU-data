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
from torch.optim import lr_scheduler 
import pandas as pd
import os
import random
import platform
from metrics import Metrics


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

'''
Create a list with max and min values for each channel for the input data
data - input in form [batch size, 1, window size, channels]
values - input argument having the max and min values of all channels from the previous iteration.
         Compares these previous values to current min and max values and updates
output - returns a list with max and min values for all channels

'''  # Calculate max min and save it to save time.
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
def normalize(data, min_max, string):
    #print(len(min_max), len(min_max[0]))
    data = data.numpy()
    #print(data.shape)
    data = data.reshape(data.shape[0],data.shape[2], data.shape[3])
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            data[i,:,j] = (data[i,:,j] - min_max[j][1])/(min_max[j][0] - min_max[j][1])
    test = np.array(data[:,:,:])
    if (string=="train"):
        if(np.max(test)>1.001):
            print("Error",np.max(test))
        if(np.min(test)<-0.001):
            print("Error",np.min(test))
    if (string=="test"):
        test[test > 1] = 1
        test[test < 0] = 0
    data = data.reshape(data.shape[0],1,data.shape[1], data.shape[2])
    data = torch.tensor(data)
    return data

'''
returns a list of F1 score for all classes
'''
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
        weighted_f1[np.isnan(weighted_f1)] = 0
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

def validation(dataLoader_validation, device, mod):
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
            out = mod(test_batch_v)
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
          print("epoch ", e)
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
          
          val_acc, val_loss =  validation(dataLoader_validation,device, model)
          validation_loss.append(val_loss)
          validation_acc.append(val_acc)
          if (val_acc >= best_acc):
              torch.save({'state_dict': model.state_dict()}, model_path)
              torch.save(model, config['model_complete'])
              #torch.save({'state_dict': model.state_dict()}, model_path)
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
    if(flag):
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
        plt.savefig('/data/sawasthi/Penn/results/result_1.png') 
        #plt.savefig('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/result.png') 
        #plt.savefig('S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/result.png'

def testing(config):
    print('Start Testing')
    
    total = 0.0
    correct = 0.0
    trueValue = np.array([], dtype=np.int64)
    prediction = np.array([], dtype=np.int64)
    #torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
    mod = torch.load(config['model_complete'])
    mod.eval()
    mod.to(device)
    loss_test = 0.0
    with torch.no_grad():
            
        for b, harwindow_batched in enumerate(dataLoader_test):
            
            test_batch_v = harwindow_batched["data"]
            test_batch_l = harwindow_batched["label"][:, 0]
            #test_batch_v = normalize(test_batch_v, value,"test")
            test_batch_v = test_batch_v.float()
            test_batch_v = test_batch_v.to(device)
            test_batch_l = test_batch_l.to(device)
            
            predictions = mod(test_batch_v)
            test_batch_l = test_batch_l.long()
            loss = criterion(predictions, test_batch_l)
            loss_test = loss_test + loss.item()
            if b == 0:
                    predictions_test = predictions
                    if config['output'] == 'softmax':
                        test_labels = harwindow_batched["label"][:, 0]
                        test_labels = test_labels.reshape(-1)

                        test_labels_window = harwindow_batched["labels"][:, :, 0]
                    elif config['output'] == 'attribute':
                        #test_labels = harwindow_batched_test["label"][:, 1:]
                        test_labels = harwindow_batched["label"]

                        test_labels_window = harwindow_batched["labels"][:, :, 0]
                    elif config['output'] == 'identity':
                        test_labels = harwindow_batched["identity"]
                        test_labels = test_labels.reshape(-1)

                    #test_file_labels = harwindow_batched["label_file"]
                    #test_file_labels = test_file_labels.reshape(-1)
            else:
                predictions_test = torch.cat((predictions_test, predictions), dim=0)
                
                if config['output'] == 'softmax':
                    test_labels_batch = harwindow_batched["label"][:, 0]
                    test_labels_batch = test_labels_batch.reshape(-1)

                    test_labels_window_batch = harwindow_batched["labels"][:, :, 0]
                elif config['output'] == 'attribute':
                    #test_labels_batch = harwindow_batched_test["label"][:, 1:]
                    test_labels_batch = harwindow_batched["label"]

                    test_labels_window_batch = harwindow_batched["labels"][:, :, 0]
                elif config['output'] == 'identity':
                    test_labels_batch = harwindow_batched["identity"]
                    test_labels_batch = test_labels_batch.reshape(-1)

                #test_file_labels_batch = harwindow_batched["label_file"]
                #test_file_labels_batch = test_file_labels_batch.reshape(-1)

                test_labels = torch.cat((test_labels, test_labels_batch), dim=0)
                #test_file_labels = torch.cat((test_file_labels, test_file_labels_batch), dim=0)
                test_labels_window = torch.cat((test_labels_window, test_labels_window_batch), dim=0)
                # Shrutarv original code
                #print("Next Batch result")
                predicted_classes = torch.argmax(predictions, dim=1).type(dtype=torch.LongTensor)
                #predicted = Testing(test_batch_v, test_batch_l)
                trueValue = np.concatenate((trueValue, test_batch_l.cpu()))
                prediction = np.concatenate((prediction,predicted_classes))
                total += test_batch_l.size(0) 
                test_batch_l = test_batch_l.long()
                predicted_classes = predicted_classes.to(device)
                correct += (predicted_classes == test_batch_l).sum().item()
                #counter = out.view(-1, n_classes).size(0)
                
        print("number of windows",test_labels.size(0))        
        size_samples = (test_labels.size(0)-1)*config["step_size"] + config['sliding_window_length']
        print("total rows in test data",size_samples)
        accumulated_predictions = torch.zeros((config["num_classes"],
                                          size_samples)).to(device, dtype=torch.long)
        predicted_classes = torch.argmax(predictions_test, dim=1).to(device,dtype=torch.long)
        targets = torch.zeros((config["num_classes"],
                                          size_samples)).to(device, dtype=torch.long)
        test_labels = test_labels.to(device)
        expand_pred = torch.ones([1,config['sliding_window_length']]).squeeze().to(device,dtype=torch.long)
        index = 0
        prediction_unsegmented = []
        #labels_per_window = harwindow_batched["label"][:,0]
        for i in range(predicted_classes.size(0)):
            # ignore the windows which are less than of size=100
            if(index +config['sliding_window_length'])>size_samples:
                print("exit on"+i)
                break
            accumulated_predictions[predicted_classes[i].item(),index:(index +config['sliding_window_length'])] += expand_pred 
            targets[test_labels[i].item(),index:(index +config['sliding_window_length'])] += expand_pred
            #temp = np.ones(1,config['sliding_window_length'])
            index += config["step_size"]
        Final_pred = torch.argmax(accumulated_predictions, dim=0).to(device,dtype=torch.long)
        Final_pred = Final_pred.unsqueeze(1)
        #df = pd.read_csv('/home/sawasthi/Thesis--Create-Synthetic-IMU-data/JHMDB/test_data.csv')
        #df = pd.read_csv('S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/train_data.csv')
        #data = df.values
        #true_labels = torch.tensor(data[:,31])
        true_labels = torch.argmax(targets, dim=0).to(device,dtype=torch.long)
        true_labels = true_labels.unsqueeze(1)
        metrics_obj = Metrics(config, device)
        # unsegmented accuracy
        true_labels = true_labels.to(device, dtype=torch.float)
        print(true_labels.size(),Final_pred.size())
        results_test = metrics_obj.metric(true_labels, Final_pred, mode="classification")
        predictions_labels = results_test["classification"]['predicted_classes'].to("cpu", torch.double).numpy()
        print('Network_User: Testing:  after de- segmented acc {}, f1_weighted {}, f1_mean {}'.format(
                results_test["classification"]['acc'], results_test["classification"]['f1_weighted'],
                results_test["classification"]['f1_mean']))
        #test_file_labels = test_file_labels.to("cpu", dtype=torch.long)
        #test_labels_window = test_labels_window.to(self.device, dtype=torch.long)
        #segmented accuracy
        results_test_segment = metrics_obj.metric(test_labels, predicted_classes, mode="classification")
        #print statistics
        print('Network_User: Testing Segmentation:  acc {}, '
            'f1_weighted {}, f1_mean {}'.format(results_test_segment["classification"]['acc'],
                                                results_test_segment["classification"]['f1_weighted'],
                                                results_test_segment["classification"]['f1_mean']))
         
        print('\nTest set:  Percent Accuracy: {:.4f}\n'.format(100. * correct / total))
            
        cm = confusion_matrix(trueValue, prediction)
        test_acc = 100. * correct / total        
        
        print(cm)
        #precision, recall = performance_metrics(cm)
        precision, recall = get_precision_recall(trueValue, prediction)
        F1_weighted, F1_mean = F1_score(trueValue, prediction, precision, recall)
        print("precision", precision)
        print("recall", recall)
        print("F1 weighted", F1_weighted)
        print("F1 mean",F1_mean)
        
        print('Finished Testing')
        return F1_weighted, test_acc
        #with open('S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/result.csv', 'w', newline='') as myfile:
        #with open('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/result.csv', 'w', newline='') as myfile:
        #with open('/data/sawasthi/JHMDB/results/result_12.csv', 'w') as myfile:
         #    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
             #wr.writerow(accuracy)
             #wr.writerow(l)
                 
        
        
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

    print(":Python Platform {}".format(platform.python_version()))
    
   
    if torch.cuda.is_available():  
          dev = "cuda:0" 
    else:  
          dev = "cpu"  
          
    device = torch.device(dev)
    config = {
        "NB_sensor_channels":26,
        "sliding_window_length":50,
        "filter_size":5,
        "num_filters":64,
        "network":"cnn",
        "output":"softmax",
        "num_classes":15,
        "reshape_input":False,
        "step_size":1,
        'model_complete': '/data/sawasthi/Penn/model/model_pose_tf_2.pth'
        }

    iterations = 1
    weighted_F1_array = []
    test_acc_array = []
    flag = True
    for iter in range(iterations):
    
        #ws=50
        accumulation_steps = 5
        correct = 0
        total_loss = 0.0
        total_correct = 0
        epochs = 150
        batch_size = 500
        lr_factor = 1
        l = []
        tot_loss = 0
        accuracy = []
        learning_rate = 0.000001
        print("accumulation_steps ", accumulation_steps, "batch_size",  batch_size, "epochs", epochs, "accumulation_steps ", accumulation_steps,"sliding_window_length", config["sliding_window_length"])    
        #df = pd.read_csv('/data/sawasthi/Thesis--Create-Synthetic-IMU-data/MoCAP/norm_values.csv')
        #df = pd.read_csv('S:/MS A&R/4th Sem/Thesis/Github/Thesis- Create Synthetic IMU data/MoCAP/norm_values.csv')
        #value = df.values.tolist()
        #print(len(df),len(value), len(value[0]))
        model = Network(config)
        model = model.float()
        model = model.to(device)
        #model.load_state_dict(torch.load())
        #print("model loaded")   # 
        normal = torch.distributions.Normal(torch.tensor([0.0]),torch.tensor([0.001]))
        #noise = noise.float()
        
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9,weight_decay=0.0005, momentum=0.9)
        #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        model_path = '/data/sawasthi/Penn/model/model_pose_tf.pth'
        #model_path = 'S:/Datasets/Penn_Action/Penn_Action/model_test.pth'
        #model_path = 'S:/MS A&R/4th Sem/Thesis/PAMAP2_Dataset/'
       
        path = '/data/sawasthi/Penn/trainData_pose_tf/'
        #path = 'S:/Datasets/Penn_Action/Penn_Action/train_pkl/'
        #path = 'S:/MS A&R/4th Sem/Thesis/PAMAP2_Dataset/pkl files'
        #path = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Train_data/"
        train_dataset = CustomDataSet(path)
        dataLoader_train = DataLoader(train_dataset, shuffle=True,
                                      batch_size=batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
      
        
        # Validation data    
        path = '/data/sawasthi/Penn/validationData_pose/'
        #path = 'S:/Datasets/Penn_Action/Penn_Action/val_pkl/'
        #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/'
        #path = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Test_data/"
        validation_dataset = CustomDataSet(path)
        dataLoader_validation = DataLoader(validation_dataset, shuffle=False,
                                      batch_size=batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
        
       
        training(dataLoader_train, dataLoader_validation,device,flag)
         # Test data    
        print("Calculating accuracy for the trained model on validation set ")
        path = '/data/sawasthi/Penn/testData_pose/'
        #path = 'S:/Datasets/Penn_Action/Penn_Action/test_pkl/'
        #path = 'S:/Datasets/Penn_Action/Penn_Action/train_pkl/'
        test_dataset = CustomDataSet(path)
        dataLoader_test = DataLoader(test_dataset, shuffle=False,
                                      batch_size=batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
        flag = False
        WF, TA = testing(config)
        weighted_F1_array.append(WF)
        test_acc_array.append(TA)
        
        testing(config)
    print("Mean Weighted F1 score after 5 runs is",np.mean(weighted_F1_array))
    print("Standard deviation of Weighted F1 score after 5 runs is",np.std(weighted_F1_array))
    print("weighted F1 array",weighted_F1_array )
    print("Mean Test accuracy score after 5 runs is",np.mean(test_acc_array))
    print("Standard deviation of Test accuracy score after 5 runs is",np.std(test_acc_array))
     # Test data    
    '''
    path = '/data/sawasthi/Penn/testData_opp_tf/'
    #path = 'S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/Windows/'
    #path = "S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/Test_data/"
    test_dataset = CustomDataSet(path)
    print("Calculating accuracy for the trained model on test set ")
    dataLoader_test = DataLoader(test_dataset, shuffle=False,
                                  batch_size=batch_size,
                                   num_workers=0,
                                   pin_memory=True,
                                   drop_last=True)
    #testing(config)
    
    print('Finished Validation')
    #with open('S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/result.csv', 'w', newline='') as myfile:
    #with open('S:/MS A&R/4th Sem/Thesis/LaRa/IMU data/IMU data/result.csv', 'w', newline='') as myfile:
    with open('/data/sawasthi/data/CAD60/results/result.csv', 'w') as myfile:
         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
         wr.writerow(accuracy)
         wr.writerow(l)
        
    '''