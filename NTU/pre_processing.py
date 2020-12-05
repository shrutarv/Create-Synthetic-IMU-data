# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:37:20 2020

@author: STUDENT
"""

'''
transform the skeleton data in NTU RGB+D dataset into the numpy arrays for a more efficient data loading
'''

import numpy as np
import os
import sys 
import csv

user_name = 'user'
save_npy_path = 'S:/MS A&R/4th Sem/Thesis/NTU/npy files/'
load_txt_path = 'S:/MS A&R/4th Sem/Thesis/nturgb+d_skeletons/'
missing_file_path = 'S:/MS A&R/4th Sem/Thesis/NTU/missing_files.txt'
step_ranges = list(range(0,100)) # just parse range, for the purpose of paralle running. 


toolbar_width = 50
def _print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def _end_toolbar():
    sys.stdout.write('\n')

def _load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True 
    return missing_files 

def _read_skeleton(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the 
    # abundant bodys. 
    # read all lines into the pool to speed up, less io operation. 
    nframe = int(datas[0][:-1])
    bodymat = dict()
    bodymat['file_name'] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat['nbodys'] = [] 
    bodymat['njoints'] = njoints 
    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])    
        if bodycount == 0:
            continue 
        # skip the empty frame 
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)
            
            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1
            
            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame,joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame,joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame,joint] = jointinfo[5:7]
    # prune the abundant bodys 
    for each in range(max_body):
        if each >= max(bodymat['nbodys']):
            if save_skelxyz:
                del bodymat['skel_body{}'.format(each)]
            if save_rgbxy:
                del bodymat['rgb_body{}'.format(each)]
            if save_depthxy:
                del bodymat['depth_body{}'.format(each)]
    return bodymat 

def choose_skel(d):
    x0 = 0
    y0 = 0
    z0 = 0
    x1 = 0
    y1 = 0
    z1 = 0   
    body0 = d['skel_body0']
    body1 = d['skel_body1']
    for i in range(pose.shape[1]):
        x0 = x0 + np.var(body0[:,i,0])
        y0 = y0 + np.var(body0[:,i,1])
        z0 = z0 + np.var(body0[:,i,2])
        x1 = x1 + np.var(body1[:,i,0])
        y1 = y1 + np.var(body1[:,i,1])
        z1 = z1 + np.var(body1[:,i,2])
        
    var0 = x0 + y0 + z0
    var1 = x1 + y1 + z1  
    if var0>var1:
        return "skel_body0"
    else:
        return "skel_body1"       


if __name__ == '__main__':
    missing_files = _load_missing_file(missing_file_path)
    datalist = os.listdir(load_txt_path)
    alread_exist = os.listdir(save_npy_path)
    alread_exist_dict = dict(zip(alread_exist, len(alread_exist) * [True]))
    pose3D = np.zeros((1,25,3)) 
    labels = np.zeros((1,1))
    for ind, each in enumerate(datalist):
        
        _print_toolbar(ind * 1.0 / len(datalist),
                       '({:>5}/{:<5})'.format(
                           ind + 1, len(datalist)
                       ))
        S = int(each[1:4])
        if each+'.skeleton.npy' in alread_exist_dict:
            print('file already existed !')
            continue
        if each[:20] in missing_files:
            print('file missing')
            continue 
        loadname = load_txt_path+each
        print(each)
        mat = _read_skeleton(loadname)
        
        name = mat['file_name']
        activity = int(name[-2:])
        subject = int(name[2:4])
        if (activity>49 and subject>12):
            continue
        else:
            if len(mat)>6:
                str = choose_skel(mat)
            else:
                str = 'skel_body0'
            pose = mat[str]
            if len(mat)>6:
                print("name",name)
            pose3D = np.concatenate((pose3D,pose), axis=0)
            label = np.full((pose.shape[0],1),activity - 1)
            labels = np.concatenate((labels,label))
            mat = np.array(mat)
            save_path = save_npy_path+'{}.npy'.format(each)
            #np.save(save_path, mat)
            # raise ValueError()
    _end_toolbar()
    p = pose3D.reshape(pose3D.shape[0],75)
    l = np.asarray(labels, dtype='float64')
    data = np.hstack((p,l))
    np.savetxt('S:/MS A&R/4th Sem/Thesis/NTU/data.csv', data, delimiter=',')
    

    
'''   
d = np.load('S:/MS A&R/4th Sem/Thesis/NTU/npy files/S001C001P001R001A024.skeleton.npy', allow_pickle=True)
d = d.tolist()
'''
