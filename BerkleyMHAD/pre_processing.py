# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:56:15 2020

@author: STUDENT
"""
from bvh import Bvh
with open('tests/test_freebvh.bvh') as f:
    mocap = Bvh(f.read())