# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:17:22 2022

@author: zhaolicheng
"""

import numpy as np
import torch
from torch.autograd import Variable

from obj_function import obj_function
from EnsembleOptmizer import EnsembleOptmizer

def algorithms(name = 'BCAS', settings={}, customized_init=None):
    
    optimizer_list = ['adam', 'gradient']
    predetermined_lr = None
    predetermined_w = None
    if name == 'BCAS':
        combination = 'hard'      
    elif name == 'CCDS':
        combination = 'soft'
        predetermined_lr = 'sqrt'
        predetermined_w = {'adam': 0.5, 'gradient': 0.5}
    elif name == 'PCAS':
        combination = 'soft'
    
    torch.manual_seed(settings['seed'])
    
    
    constant_dict = {}
    constant_dict['am_torch'] = torch.from_numpy(settings['am'])
    constant_dict['pm_torch'] = torch.from_numpy(settings['pm'])
    constant_dict['am_comm_torch'] = torch.from_numpy(settings['am_comm'])
    constant_dict['comm_level_torch'] = torch.from_numpy(settings['comm_level'])
    constant_dict['am_corr_torch'] = torch.from_numpy(settings['am_corr'])
    
    M = settings['M']
    constant_dict['M'] = settings['M']
    N = settings['N']
    constant_dict['N'] = settings['N']
    constant_dict['J'] = settings['J']
    constant_dict['X0_val'] = settings['X0_val']
    
    params = [] 
    real = Variable(torch.normal(mean=0, std=0.1, size=(M, N)), requires_grad = True)
    params.append(real) 
    imag = Variable(torch.normal(mean=0, std=0.1, size=(M, N)), requires_grad = True) 
    params.append(imag)  
  
        
    obj_func = lambda params: obj_function(params, constant_dict)
    optimizer = EnsembleOptmizer(params, optimizer_list=optimizer_list, 
                                     obj_func=obj_func, 
                                     amsgrad=True, combination=combination,
                                     predetermined_lr=predetermined_lr,
                                     predetermined_w=predetermined_w)
    
    obj_list = []
    
    for t in range(settings['iter_num']): 
        obj, X, alpha = obj_func(params)
        obj_val = obj.item()
        X_val = X.detach().numpy()
        alpha_val = alpha.detach().numpy()
        
        obj_list.append(obj_val)
    
        if (t + 1) % 100 == 0:
            print(name, 'iter num: ', t + 1)
            
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()
        
        if t > 1 and np.abs(obj_list[t-1] - obj_list[t]) / max(1.0, obj_list[t-1]) < settings['tol']:
            break
        
    dic = {'X': X_val, 'alpha': alpha_val, 'obj_list': obj_list}        
    return dic
    
    