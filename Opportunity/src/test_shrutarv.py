# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:44:34 2021

@author: STUDENT
"""
import torch

pretrained_dict = torch.load('/data/sawasthi/CAD60/model/model_tf.pth').state_dict()

torch.save({'state_dict': pretrained_dict},
                       '/data/sawasthi/CAD60/model/' + 'network.pt')     