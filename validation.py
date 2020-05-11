# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:12:06 2020

@author: TBE7
"""

# How to run
from main import main_func
import time
import pandas as pd
import numpy as np
t = pd.read_excel (r'Params.xlsx', header = None, sheet_name = 't')
t = t.to_numpy().reshape((61,))
data = pd.read_excel (r'Params.xlsx', header = None, sheet_name = 'data')
data = data.to_numpy()
u = np.ones((1, 61))
lengthu = 1
N = data.shape[0]
N1 = N+1
N2 = N**2


#%% running with data
tstart = time.time()
main_func(data, t, u, lengthu)
print('time elapsed in seconds is', time.time() - tstart)

#%% running without data
tstart = time.time()
main_func()
print('time elapsed in seconds is', time.time() - tstart)

#%% Parameters of the Vand de Vusse reactor
k1, k2, k3 =5/6, 5/3, 1/6 
f_ov_ss =4/7 
caf =  10 
Cas= 3 
Cbs = 1.117 
f_ov_ss =4/7 

Atrue = np.array([[-(f_ov_ss + k1 + 2*k3*3), 0], [k1, -(f_ov_ss + k2)] ] , dtype = np.float16)
Btrue = np.array([[caf  -Cas, -1, 0], [-Cbs, 0 , -1] ] , dtype = np.float16)


