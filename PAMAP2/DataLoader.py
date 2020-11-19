from torch.utils.data.dataset import Dataset
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch
import pickle
import numpy as np
import random

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        #self.transform = transform
        self.all_files = os.listdir(main_dir)
        
        #for file_name in self.all_files:
          #  if '.txt' in file_name: self.total_imgs.remove(file_name)
          #  if file_name == 'semantic': self.total_imgs.remove('semantic')

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file = os.path.join(self.main_dir, self.all_files[idx])
        #print(idx)
        f = open(file,'rb')
        #image = Image.open(img_loc).convert("RGB")
        #tensor_image = self.transform(image)
        data = pickle.load(f, encoding='latin1')
        f.close()
        X = data['data']
       
        y = data['label']
        Y = data['labels']
        
        window_data = {"data" : X, "label" : y, "labels" : Y}

        return window_data
        # train = np.transpose(dat)
#        train = np.reshape(dat,(dat.shape[1],dat.shape[2]))
        
        #train = np.transpose(dat)
        #train = np.reshape(train,(train.shape[0],(train.shape[1]))
        
 #       tensor_file = torch.tensor(train)
  #      return tensor_file
    
class CustomDataSetTest(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        #self.transform = transform
        self.all_files = os.listdir(main_dir)
        
        #for file_name in self.all_files:
          #  if '.txt' in file_name: self.total_imgs.remove(file_name)
          #  if file_name == 'semantic': self.total_imgs.remove('semantic')

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        #print(idx)
        file = os.path.join(self.main_dir, self.all_files[idx])
        f = open(file,'rb')
        #image = Image.open(img_loc).convert("RGB")
        #tensor_image = self.transform(image)
        pk = pickle.load(f)
        f.close()
        dat = pk['label']
        dat = dat[0]
        tensor_file = torch.tensor(dat, dtype=torch.long)
        
        return tensor_file