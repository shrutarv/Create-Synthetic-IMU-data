# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:56:15 2020

@author: STUDENT
"""
from bvh import Bvh
with open('S:/MS A&R/4th Sem/Thesis/Berkley MHAD/SkeletalData-20200922T160342Z-001/SkeletalData/skl_s01_a01_r01.bvh') as f:
    mocap = Bvh(f.read())