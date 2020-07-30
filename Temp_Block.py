# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:16:39 2020

@author: STUDENT
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels,stride=1, dilation=4, padding=0, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, 64, kernel_size = 5,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(3, stride=1)
        self.conv2 = weight_norm(nn.Conv1d(64, 64, kernel_size=5,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(3, stride=1)
        
        
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        #self.conv3.weight.data.normal_(0, 0.01)
        #self.conv4.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        #print(out.shape)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        #print(out.shape)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        #num_levels = len(num_channels)
        for i in range(2):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else 64
            #out_channels = num_channels[i]
            # padding=(kernel_size-1) * dilation_size
            layers += [TemporalBlock(in_channels, stride=1, dilation=dilation_size,
                                     padding = 0, dropout=dropout)]

        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(64*168*30, 128)
        
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        d = nn.Dropout(p=0.2) 
        x = self.net(x)
        x = x.view(1, -1)
        #print(x.shape)
        x = self.fc1(x)
        x = d(x)
        x = self.fc2(x)
        x = d(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    



class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1.transpose(1, 2))