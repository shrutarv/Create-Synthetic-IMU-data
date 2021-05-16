# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:44:34 2021

@author: STUDENT
"""
from Network import Network
import torch

config = {
        "NB_sensor_channels":27,
        "sliding_window_length":100,
        "filter_size":5,
        "num_filters":64,
        "network":"cnn",
        "output":"softmax",
        "num_classes":8,
        "reshape_input":False,
        "folder_exp_base_fine_tuning": '/data/sawasthi/JHMDB/model/model_acc_up4.pth'
        #"folder_exp_base_fine_tuning": 'S:/MS A&R/4th Sem/Thesis/LaRa/OMoCap data/model_full.pth'
        }
model = Network(config)
model.load_state_dict(torch.load('/data/sawasthi/Penn/model/model_tf.pth'))
#m = torch.load('S:/MS A&R/4th Sem/Thesis/J-HMDB/joint_positions/train/model.pth')
torch.save({'state_dict': model},
                       '/data/sawasthi/Penn/model/network.pt')     