# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 18:58:58 2021

@author: zhaolicheng
"""

import torch
import numpy as np

def obj_function(params, constant_dict):           
    omega = 0.1       
    rho = 0.1    
    tau = 0.1
          
    am_torch = constant_dict['am_torch']
    pm_torch = constant_dict['pm_torch']
    am_comm_torch = constant_dict['am_comm_torch']
    comm_level_torch = constant_dict['comm_level_torch']
    am_corr_torch = constant_dict['am_corr_torch']
    M = constant_dict['M']
    N = constant_dict['N']
    J = constant_dict['J']
    X0_val = constant_dict['X0_val']
      
    real = params[0]  
    imag = params[1]   
    X = torch.complex(real, imag)
    X = X / torch.abs(X) / np.sqrt(M * N)    
    
    mul = torch.matmul(torch.conj(am_torch), X)
    mul_abs = torch.abs(torch.sum(torch.conj(mul) * mul, axis=-1))   
    if 'params_theta_flag' not in constant_dict:
        mul_abs_select = torch.masked_select(mul_abs, torch.eq(pm_torch, torch.ones_like(pm_torch)))   
        alpha = torch.mean(mul_abs_select) 
    else:
        if constant_dict['params_theta_flag']:
            alpha = params[1]     
    diff_abs = torch.abs(alpha * pm_torch - mul_abs)  
    beampattern_loss = torch.mean(diff_abs ** 2)
     
    mul_corr = torch.matmul(torch.conj(am_corr_torch), X)
    corr = torch.matmul(mul_corr, torch.conj(mul_corr).t())
    cross_corr_abs = torch.abs(corr - torch.diag(torch.diag(corr)))
    cross_corr_loss = torch.sum(cross_corr_abs ** 2) / (J * (J - 1))
    
    similarity_loss = torch.mean(torch.abs(X - torch.from_numpy(X0_val)) ** 2)
    
    mul_comm = torch.matmul(torch.conj(am_comm_torch), X)
    mul_comm_abs = torch.abs(torch.sum(torch.conj(mul_comm) * mul_comm, axis=-1))   
    diff_comm_abs = torch.log10(mul_comm_abs) - torch.log10(comm_level_torch)
    communication_loss = torch.mean(diff_comm_abs ** 2)
    
    obj = beampattern_loss + omega * cross_corr_loss + rho * similarity_loss + tau * communication_loss
    
    return obj, X, alpha
    