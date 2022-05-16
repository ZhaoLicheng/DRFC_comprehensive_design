# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 22:08:02 2021

@author: zhaolicheng
"""
import numpy as np
import matplotlib.pyplot as plt 


def convergence_plot(obj_list_dict):
    plt.figure() 
    color_list = ['b', 'm', 'r']
    for i, key in enumerate(obj_list_dict):
        iteration = list(range(1,len(obj_list_dict[key]) + 1))
        plt.loglog(iteration, obj_list_dict[key], label=key, color=color_list[i])
        plt.axhline(y=obj_list_dict[key][-1], color=color_list[i], linestyle='--', dashes=(3, 2))
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.legend(loc='upper right')
    plt.grid()  
    plt.show()   

def beampattern_plot(basic_settings, dict_alg):
    am = basic_settings['am']
    pm = basic_settings['pm']
    theta_main = basic_settings['theta_main']
    theta_comm = basic_settings['theta_comm']
    comm_level = basic_settings['comm_level']


    theta_comm_index = [i for i in range(len(theta_main)) if theta_main[i] in theta_comm]

    designed_beampattern={}
    designed_beampattern_normalized={}
  
    desired_beampattern = np.reshape(pm, [-1])

    color_list = ['b', 'r', 'm']

    for key in ['PCAS','BCAS','CCDS']:
        mul = np.matmul(np.conj(am), dict_alg[key]['X'])
        designed_beampattern[key] = np.reshape(np.abs(np.sum(np.conj(mul) * mul, axis=-1)) , [-1])
        designed_beampattern_normalized[key] = np.reshape(np.abs(np.sum(np.conj(mul) * mul, axis=-1)) , [-1]) / dict_alg[key]['alpha']


    plt.figure()
    
    for i, key in enumerate(['BCAS', 'PCAS', 'CCDS',]):
        plt.plot(theta_main, 10 * np.log10(designed_beampattern[key]), label=key + ' (proposed)', color=color_list[i])
        
    plt.plot(theta_main, 10 * np.log10(3.2 * desired_beampattern + 1e-10), label='ideal beampattern', color='goldenrod')

    for theta in theta_comm:
        plt.axvline(x=theta, color='k', linestyle='--', dashes=(3, 2))
    
    for lvl in comm_level:
        plt.axhline(y=10 * np.log10(lvl), color='k', linestyle='--', dashes=(3, 2))
    
    for i, key in enumerate(['PCAS','BCAS','CCDS']):
        plt.scatter(theta_comm, [10 * np.log10(designed_beampattern[key][ind]) for ind in theta_comm_index], color=color_list[i], marker='x', s=150)

    plt.xlabel('Angle (degree)')
    plt.ylabel('Beampattern (dB)')
    plt.xticks(np.arange(-90, 91, step=15))
    plt.ylim(-58, 8)
    plt.legend(loc = 'lower left')
    plt.grid()     
    plt.show()

    # =======================================================================

    plt.figure()
    
    for i, key in enumerate(['BCAS', 'PCAS', 'CCDS',]):
        plt.plot(theta_main, 10 * np.log10(designed_beampattern_normalized[key]), label=key + ' (proposed)', color=color_list[i])
        
    plt.plot(theta_main, 10 * np.log10(desired_beampattern + 1e-10), label='ideal beampattern', color='goldenrod')
    
    for theta in theta_comm:
        plt.axvline(x=theta, color='k', linestyle='--', dashes=(3, 2))
  
    plt.xlabel('Angle (degree)')
    plt.ylabel('Beampattern (dB)')
    plt.xticks(np.arange(-90, 91, step=15))
    plt.ylim(-68, 8)
    plt.legend(loc = 'lower left')
    plt.grid()   
    plt.show()
 
    

            
            