# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:57:46 2021

@author: zhaolicheng
"""
import numpy as np
import torch 
from torch.optim.optimizer import Optimizer

class EnsembleOptmizer(Optimizer):
    def __init__(self, params, optimizer_list=['adam'], obj_func=None, lr_lst=[1e-3], betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, combination='soft', predetermined_lr=None,
                 predetermined_w=None):
        defaults = dict(betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(EnsembleOptmizer, self).__init__(params, defaults)
        self.optimizer_list = optimizer_list
        self.obj_func = obj_func
        self.combination = combination
        self.predetermined_lr = predetermined_lr
        self.predetermined_w = predetermined_w
        self.iter = 0

    @torch.no_grad()        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
         
        self.iter += 1
        
        params_current = []
        params_gradient = []
        params_delta = {x: [] for x in ['adam', 'gradient']}
        gd_ip = {}
        lr = {}   
        params_update = {}
        obj_descent = {}
        
        for group in self.param_groups:
            eps=group['eps']
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']         
            for p in group['params']:
                if p.grad is None:
                    print(p, "gradient is None!")
                    continue
                
                params_current.append(p)
                params_gradient.append(p.grad)
                state = self.state[p]                         
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)         
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)   
                    state['beta1'] = 1.0
                    state['beta2'] = 1.0
                    if amsgrad:
                        state['vmax'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                state['m'].mul_(beta1).add_(p.grad, alpha=1 - beta1)               
                state['v'].mul_(beta2).addcmul_(p.grad, p.grad.conj(), value=1 - beta2)
                state['beta1'] *= beta1
                state['beta2'] *= beta2
                if amsgrad:
                    torch.maximum(state['vmax'], state['v'], out=state['vmax'])
                    denom = state['vmax'].sqrt().add_(eps)
                else:
                    denom = state['v'].sqrt().add_(eps)                  
                delta_p = np.sqrt(1.0 - state['beta2']) / (1.0 - state['beta1']) * (state['m'] / denom)
                params_delta['adam'].append(delta_p)
                params_delta['gradient'].append(p.grad)
           
            
        obj_func = self.obj_func 
        current_obj = obj_func(params_current)[0].numpy()
                        
        if self.predetermined_lr is None:
                
            def gd_ip_load(key):
                gd_ip[key] = np.sum([torch.sum(g * d) for g, d in zip(params_gradient, params_delta[key])])
            
            for optimizer in self.optimizer_list:
                gd_ip_load(optimizer)    
            
            def line_search(key):
                lr[key] = 5e-3
                params_update[key] = []
                for p, delta_p in zip(params_current, params_delta[key]):
                   p_update = torch.add(p, delta_p, alpha=-lr[key])
                   params_update[key].append(p_update)
                update_obj = obj_func(params_update[key])[0].numpy()
                obj_descent[key] = current_obj - update_obj
                
                
                if obj_descent[key]  < 0.3 * lr[key] * gd_ip[key]:
                    while obj_descent[key]  < 0.3 * lr[key] * gd_ip[key] and lr[key] > 1e-9:
                        lr[key] *= 0.9              
                        params_update[key] = []
                        for p, delta_p in zip(params_current, params_delta[key]):
                           p_update = torch.add(p, delta_p, alpha=-lr[key])
                           params_update[key].append(p_update)
                        update_obj = obj_func(params_update[key])[0].numpy()
                        obj_descent[key] = current_obj - update_obj
                else:
                    while obj_descent[key]  >= 0.3 * lr[key] * gd_ip[key] and lr[key] < 1e9:
                        lr[key] *= 1.1
                        
                        params_update_store = [x for x in params_update[key]]              
                        params_update[key] = []               
                        for p, delta_p in zip(params_current, params_delta[key]):
                           p_update = torch.add(p, delta_p, alpha=-lr[key])
                           params_update[key].append(p_update)
                        update_obj = obj_func(params_update[key])[0].numpy()
                        obj_descent[key] = current_obj - update_obj
                    
                    params_update[key] = params_update_store
                    lr[key] /= 1.1
            
            for optimizer in self.optimizer_list:
                line_search(optimizer)
                
        elif self.predetermined_lr == 'sqrt':
            for key in self.optimizer_list:
                lr[key] = 1.0 / np.sqrt(self.iter)         
                params_update[key] = []
                for p, delta_p in zip(params_current, params_delta[key]):
                   p_update = torch.add(p, delta_p, alpha=-lr[key])
                   params_update[key].append(p_update)
                update_obj = obj_func(params_update[key])[0].numpy()
                obj_descent[key] = current_obj - update_obj
            
        comb_wts = None
        
        if self.combination == 'hard':
            obj_descent_max = 0
            key_max = ''
            for k in obj_descent:
                if obj_descent[k] > obj_descent_max:
                    obj_descent_max = obj_descent[k]
                    key_max = k
            
            if key_max == '':
                key_max = 'adam'
           
            comb_wts = {}
            for k in obj_descent: 
                if k == key_max:
                    comb_wts[k] = 1.0
                else:
                    comb_wts[k] = 0.0
            
            for p, p_update in zip(params_current, params_update[key_max]):
                p.data = p_update.data
                
        else:
            
            if self.predetermined_w is None:               
            
                obj_descent_array = np.array(list(obj_descent.values()))  
                obj_descent_array_abs = np.abs(obj_descent_array)
                rescale_factor = np.min(obj_descent_array_abs[obj_descent_array_abs>0])    
                obj_descent_array_rescale = obj_descent_array / rescale_factor
                obj_descent_array_rescale_max = np.max(obj_descent_array_rescale)
                obj_descent_array_rescale_minusmax_expsum = np.sum(np.exp(obj_descent_array_rescale - obj_descent_array_rescale_max))
                comb_wts = {}
                for k in obj_descent:
                    comb_wts[k] = np.exp(obj_descent[k] / rescale_factor - obj_descent_array_rescale_max) / obj_descent_array_rescale_minusmax_expsum
            else:
                comb_wts = {key: value for key, value in self.predetermined_w.items()}
                                             
            for i,p in enumerate(params_current):
                p_tmp = None
                for k in params_update:
                    if p_tmp is None:
                        p_tmp = comb_wts[k] * params_update[k][i] 
                    else:
                        p_tmp += comb_wts[k] * params_update[k][i]
                p.data = p_tmp.data
        
        return loss
    
    
        