# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:18:36 2020

@author: TBE7
"""

from scipy.integrate import solve_ivp
from functools import partial
from vdvnonlinear import vandevusse_Nonlinear
import numpy as np
import matplotlib.pyplot as plt

Cas= 3 
Cbs = 1.117 
f_ov_ss =4/7 
Caf_ss=10 
tspan = [0, 3.05] 
interval = 1 
Ca0 = 2 
Cb0 = 1.117 
y0=(Ca0, Cb0) # you can pass this as a list as well:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

f_OV = [1] 
vdvnonl = partial( vandevusse_Nonlinear, interval, tspan[-1], f_OV )
t_sp = np.arange( tspan[0], tspan[-1],  0.05 )
sol = solve_ivp(vdvnonl, tspan, y0, t_eval = t_sp, dense_output = False  )


lengthu = len(f_OV )
u = np.ones((1, len(t_sp)))
np.savez('file', t = sol.t, y = sol.y - np.array([[Cas],[Cbs]]), u = u, lenu = lengthu )


hf = plt.figure()
ha = hf.add_subplot(111)
ha.plot(sol.t, sol.y[1,:], 'c*-')
ha.set_xlabel(r'$t$', labelpad = 10)
ha.set_ylabel(r'$C_A$', labelpad = 10)


    

