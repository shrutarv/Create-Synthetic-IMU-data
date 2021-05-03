'''
Created on Oct 02, 2019

@author: fmoya
'''

import os
import sys
import numpy as np

import csv_reader
from sliding_window import sliding_window
import pickle


from attributes import Attributes

#from HARwindows import HARWindows

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import DataLoader

from scipy.stats import norm, mode
