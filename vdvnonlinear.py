# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:24:01 2020

@author: TBE7
"""

def vandevusse_Nonlinear(interval, tend, f_ov, t, z):

    k1, k2, k3 =5/6, 5/3, 1/6 
    f_ov_ss =4/7 
    caf =  10 
    
    ca, cb = z 
    
    for i in range( 1, interval + 1 ):
        if t>=((i-1)/interval)*tend and t<i/interval *tend:
            f_ov_u = f_ov[i-1] 
        if t>=tend:
            f_ov_u = f_ov[-1]  
    
    fv = f_ov_u + f_ov_ss 
    
    return [ fv*(caf-ca)-k1*ca-k3*ca**2, -fv*cb+k1*ca-k2*cb ]
