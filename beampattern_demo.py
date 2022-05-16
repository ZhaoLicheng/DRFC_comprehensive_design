# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:21:59 2022

@author: zhaolicheng
"""
import numpy as np
from data_gen import data_gen_func
from algorithms import algorithms
from plot import beampattern_plot

M = 32
N = 64

theta_target=np.array([-30, 60], dtype=np.float32)
theta_comm=np.array([-80, 10, 35], dtype=np.float32)
comm_level=np.array([1e-2, 1e-4, 1e-4], dtype=np.float32)

theta_main, I, J, theta_comm, comm_level, L, am, am_corr,\
 am_comm, pm, X0_val = data_gen_func(M, N, theta_target=theta_target, theta_comm=theta_comm, comm_level=comm_level)


basic_settings = {'theta_main': theta_main, 'theta_comm': theta_comm,
                  'am': am, 'pm': pm, 'am_comm': am_comm, 
                  'comm_level': comm_level, 'am_corr': am_corr,
                  'M': M, 'N': N, 'J': J, 'L': L, 'X0_val': X0_val}

alg_settings = {'iter_num': int(5e3), 
                'seed': 2022, 'tol': 1e-7}

settings = {**basic_settings, **alg_settings}

dict_BCAS = algorithms(name='BCAS', settings=settings)
dict_CCDS = algorithms(name='CCDS', settings=settings)
dict_PCAS = algorithms(name='PCAS', settings=settings)

dict_alg = {'BCAS': dict_BCAS, 
            'CCDS': dict_CCDS,
            'PCAS': dict_PCAS}

beampattern_plot(basic_settings, dict_alg)
