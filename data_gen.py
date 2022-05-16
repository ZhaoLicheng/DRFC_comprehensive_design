# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 20:45:58 2021

@author: zhaolicheng
"""

import numpy as np
import math
import os

"""
radar setting
"""

def data_gen_func(M=32, N=32, theta_target=None, theta_comm=None, comm_level=None):
    angle_interval = 1
    theta_main = np.arange(start=-89, stop=90, step=angle_interval, dtype=np.float32)
    I = len(theta_main)
    if theta_target is None:
        theta_target = np.array([0, 0, 0], dtype=np.float32)
    J = len(theta_target)
    if theta_comm is None:
        theta_comm = np.array([0, 0, 0], dtype=np.float32)
    if comm_level is None:
        comm_level = np.array([0, 0, 0], dtype=np.float32)
    L = len(theta_comm)
    
    
    def deg2rad(x):
        return math.pi / 180 * x
      
    am = np.exp(1j * math.pi * np.sin(deg2rad(theta_main[:, np.newaxis])) * np.arange(start=0, stop=M, step=1, dtype=np.float32)[np.newaxis, :])
    am_corr = np.exp(1j * math.pi * np.sin(deg2rad(theta_target[:, np.newaxis])) * np.arange(start=0, stop=M, step=1, dtype=np.float32)[np.newaxis, :])
    am_comm = np.exp(1j * math.pi * np.sin(deg2rad(theta_comm[:, np.newaxis])) * np.arange(start=0, stop=M, step=1, dtype=np.float32)[np.newaxis, :])
    
    def f_target(theta):
        for target in theta_target:
            if theta > target - 10 and theta < target + 10:
                return 1.0
        return 0.0
    
    pm = np.array([f_target(x) for x in theta_main])
    
    dirname = os.path.dirname(__file__)
    X0_val = np.genfromtxt(os.path.join(dirname,"X0/X0_M_" + str(M) + "_N_" + str(N) + ".csv"), 
                                        dtype=None, delimiter=',', encoding=None)
    X0_val = np.char.replace(X0_val,'i','j').astype(np.complex64)
    
    return theta_main, I, J, theta_comm, comm_level, L, am, am_corr, am_comm, pm, X0_val