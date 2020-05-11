# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:29:26 2020

@author: TBE7
"""
cimport cython
from libc.math cimport sqrt
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double diffnorm(double [::1,:] a, double [::1,:] b, int N, int M):
    cdef short i, j
    cdef double s = 0.0
    for j in range(M):
        for i in range(N):
            s += ( a[i,j] - b[i,j]) *( a[i,j] - b[i,j] )
    return sqrt(s)        

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double [:,::1] creatematrix(double [:,::1] input1, double [::1,:] input2, double m, int N1) :
    cdef double [:,::1] output = np.zeros((N1, N1))
    cdef Py_ssize_t  i
    cdef Py_ssize_t  j 
    for i in range(N1):
        for j in range(N1):
            output[i,j] = m*(input1[i,j] + input2[i,j])
    return output