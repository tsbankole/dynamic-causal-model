# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:16:03 2020

@author: TBE7
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport floor
from scipy.linalg import expm
from scipy.interpolate import UnivariateSpline as spline
from scipy.interpolate import CubicSpline as cspline
from multiprocessing import Pool, cpu_count
from numpy.linalg import solve, inv
from cython.parallel import prange
cimport defrms
from defrms cimport diffnorm
from defrms cimport creatematrix
'''
cdef np.ndarray[DTYPE_t, ndim=2] 

#%% testing pyd imports
cdef double [:,::1] rrr = np.random.random((3,3))
rms(rrr, 3, 3)

'''

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef main_func_cython(data_c_in = None, t_in = None, u_in = None, lengthu_in = None):
    #%% Preliminaries
    cdef double [:] t
    cdef double [::1, :] data 
    cdef double [:, ::1] u 
    cdef int lengthu
    
    if data_c_in is None or t_in is None or u_in is None or lengthu_in is None:
        loadeddata = np.load('file.npz')
        t_in, data_c_in, u_in, lengthu_in = loadeddata['t'], loadeddata['y'], loadeddata['u'], loadeddata['lenu']
        t = t_in
        data = np.asfortranarray( data_c_in )
        u = u_in
        lengthu = lengthu_in        
    else:
        t = t_in
        data = np.asfortranarray( data_c_in )
        u = u_in
        lengthu = lengthu_in       
    
         
    cdef int N = data.shape[0]
    cdef int N1 = N+1;
    cdef int N2 = N**2;
    
    cdef Py_ssize_t i, j, k, m, g, row, pp, qq
    
    #define setA, setB
    cdef int [:] setA = np.array([i for i in range(N2)], dtype = np.int32)
    cdef int [:] setB = np.array([i for i in range(N2, 2*N2 + N)], dtype = np.int32)
    cdef int numsetA = len(setA)
    cdef int numsetB = len(setB)
    cdef short [:] ar_a = np.empty([numsetA], dtype = np.int16)
    cdef unsigned short [:] ac_a = np.empty([numsetA], dtype = np.uint16 )
    
    cdef short [:] bi_b = np.zeros([numsetB], dtype = np.int16)
    cdef short [:] ar_b = np.zeros([numsetB], dtype = np.int16)
    cdef short [:] ac_b = np.zeros([numsetB], dtype = np.int16)
    
    #%%  define the parameters
    cdef short count = -1
    cdef short count1 = numsetA - 1
    cdef int lenreading = data.shape[1] - 1
    cdef int lenreadpone = lenreading + 1
    cdef int Lenvec = numsetA+numsetB
    cdef int sz = lenreading*N;
    cdef int szvec = sz+Lenvec   
    cdef short location
    cdef double [:] vecparams = np.zeros([Lenvec,])
    cdef double [:] meanprior = np.copy( vecparams )
    cdef double [:] tmp = np.empty([Lenvec,]) 
    cdef double [::1,:] tmpoldsum = np.empty([N1,N1], order = 'F') 
    cdef double [:] tmpxq = np.empty([N1])
    cdef double [::1,:] storetmp = np.empty([N1, N1], dtype = np.float64, order = 'F') 
    cdef double [:] aa = np.empty(N1)
    
    #%% Initialize matrices A,B
    cdef double [:,::1] A = np.zeros([N1,N1], dtype = np.float64)
    cdef double [:,::1] Adel = np.zeros([N1,N1], dtype = np.float64)
    
    for i in range( numsetA ):
        ar_a[i] = <short> ( floor(setA[i]/N) + 1 )
        #ar_a[i] = np.floor(setA[i]/N) + 1
        ac_a[i] = setA[i] - ( ar_a[i] - 1)*N + 1
        A[ar_a[i],ac_a[i]] = 0.5 #np.random.random()
        vecparams[i] = A[ar_a[i] ,ac_a[i]]
    Adel[:] = A
    
    cdef double [:,:,::1] B = np.zeros([N1,N1, lengthu], dtype = np.float64)
    
    for i in range( numsetB ):
        location = setB[i] -N2
        bi_b[i] = <short> ( floor(location/(N2+N)) )
        ar_b[i] = <short> ( floor(( location )/N1 ) + 1 )
        #bi_b[i] = np.floor(location/(N2+N))
        #ar_b[i] = np.floor((location )/N1 ) + 1        
        ac_b[i] = location - N1*( ar_b[i] -1  )
        B[ar_b[i],ac_b[i],bi_b[i]] = 0.5 #np.random.random()
        vecparams[i + N2] = B[ar_b[i],ac_b[i],bi_b[i]]
        
    #%% Initialize oldsum 
    cdef double [::1,:,:] oldsum = np.zeros([N1,N1,lenreading], dtype = np.float64, order = 'F')
    cdef double [:] sliced = np.empty(lenreading, dtype = np.float64 )
    cdef double [:,::1] tmpAdeloldsum = np.empty([N1, N1], order = 'C')
    Uvec = np.transpose(np.asarray(u)[..., np.newaxis],[2, 1, 0])
    
    for j in range( lenreading ): 
        tmpoldsum = np.asfortranarray( np.sum(B * Uvec[:,j,:] ,2) )
        oldsum[...,j] = tmpoldsum
        
    cdef double [:,:] sum_uB = np.empty( (N1,N1) )
    
    #%% Monitors
    secderr = 1
    
    # %% Initialize xq, store
    data = np.asfortranarray( np.vstack( ( np.ones([1,lenreadpone]), data ) ) )
    cdef double [::1,:] xq = np.empty((N1, lenreadpone), order = 'F')
    cdef double [::1,:,:] store = np.empty([N1, N1, lenreading], dtype = np.float64, order = 'F') 
    xq[:] = data
    #store = np.empty([N1, N1, lenreading], dtype = np.float64) 
    
    for i in range( lenreading ):
        storetmp = expm(np.asarray( creatematrix(A, oldsum[...,i], t[i+1] -t[i], N1 ) ) )
        store[..., i] = storetmp
        
    for i in range( lenreading ):
        tmpxq = np.dot(store[...,i],xq[:,i]) #possible optimization
        xq[:,i+1] = tmpxq
    #print('first xq', np.linalg.norm(xq) )

    
    # %% M-step variables
    cdef double [:] lam = np.ones((sz,), dtype = np.float64)
    cdef double [:] der = np.copy(lam ).astype( dtype = np.float64 )
    cdef double [:, ::1] secder = np.zeros([sz,sz], np.float64)
    cdef double [:, ::1] Cthprior  = np.diag( np.full( (Lenvec,), 20, dtype = np.float64 ) )  # larger premultipliers imply less confidence in priors
    
    # %% Initialize variables for J matrix
    cdef double[::1,:] hwhole = np.zeros([sz,Lenvec], dtype = np.float64 , order = 'F')
    cdef double[::1,:] h = np.zeros([N1,lenreading], dtype = np.float64 , order = 'F')
    cdef double[::1,:] h1 = np.zeros([N1,lenreading], dtype = np.float64 , order = 'F')
    cdef double [:] hwholetmp = np.empty(sz)
    cdef double[:] DF = np.copy(hwhole[:,1])
    
    cdef short span = 2
    cdef double delta = 0.01
    cdef double delta1 = delta
    cdef short [:] ind = np.arange(-span, span + 1 , dtype = np.int16)
    cdef unsigned short szxrange = 2*span + 1
    cdef unsigned short [:] dex = np.empty(szxrange, dtype = np.uint16)
    for pp in range(szxrange):
        dex[pp] = ind[pp] + span
    cdef double [::1,:] tbspline = np.zeros([N1,szxrange], dtype = np.float64, order = 'F' )
    cdef double [:] xrange = np.zeros([szxrange], dtype = np.float64 )
    
    #%% Loop utilities
    
    cdef double tol = 1e-8, Arac= 0, delT = 0
    cdef int conv =1
    cdef int n =-1, n1 = 0, ar = 0 , ac = 0, bi = 0
    cdef int maxIter = 15
    cdef double [:] diff = np.full( [maxIter + 1,], np.Inf, dtype = np.float64 )
    
    #%% typing locals in loop
    cdef double B_ar_ac_bi
    cdef double [:] df = np.empty(N1)
    
    
    #%% Loop
    while conv > tol and n < maxIter:
        n  += 1
        n1 += 1
        
        Adel[:] = A
        
        for k in range(numsetA):
            ar = ar_a[k]
            ac = ac_a[k]
            count += 1
            Arac = A[ar][ac]
            for pp in range(szxrange):
                xrange[pp] = Arac + ind[pp]*delta
            for i in range(lenreading):
                delT = t[i+1] - t[i]
                for pp in range(N1):
                    for qq in range(N1):
                        tmpAdeloldsum[pp,qq] = delT * ( A[pp,qq] + oldsum[pp,qq,i] )
                for g in range(szxrange):
                    tmpAdeloldsum[ar,ac] = delT * ( xrange[g] + oldsum[ar,ac,i] )
                    aa = np.dot( expm(np.asarray( tmpAdeloldsum) ), data[:,i] ) #possible optimization
                    tbspline[:,g] = aa
                for row in range(1, N1 ):
                    h[row,i] = cspline(np.asarray( xrange ), np.asarray( tbspline[row,:])).derivative()(Arac)          
            hwholetmp = np.reshape(h[1:,], [sz,])
            for pp in range(sz):
                hwhole[pp,count] = hwholetmp[pp]
        count = -1
        
        for j in range(numsetB):
            ar = ar_b[j]
            ac = ac_b[j]
            bi = bi_b[j]
            count1 = count1 + 1
            B_ar_ac_bi = B[ar,ac,bi]
            for pp in range(5):
                xrange[pp] = B_ar_ac_bi + ind[pp]*delta1
            sliced[:] = oldsum[ar,ac,:]
            for i in range(lenreading):
                delT = t[i+1] - t[i]
                sum_uB[:] = oldsum[:,:,i]
                for pp in range(N1):
                    for qq in range(N1):
                        tmpAdeloldsum[pp,qq] = delT * ( A[pp,qq] + sum_uB[pp,qq] )                
                for g in range(szxrange):
                    tmpAdeloldsum[ar,ac] = delT * ( A[ar,ac] + (sliced[i] + u[bi,i]*ind[g]*delta1) )  
                    aa = np.dot( expm(np.asarray(tmpAdeloldsum)) , data[:,i] ) #possible optimization
                    tbspline[:,g] = aa
                for row in range(1, N1 ):
                    h1[row,i] =  cspline(xrange, tbspline[row,:]).derivative()(B_ar_ac_bi)
            hwholetmp = np.reshape(h1[1:,:],[sz,])
            for pp in range(sz):
                hwhole[pp,count1] = hwholetmp[pp]
        count1 = numsetA - 1
        
        
        #print(np.asarray( hwhole[::10, :]) )

        
    #%% ybar Jbar   
        for i in range(1,lenreadpone ):
            df = data[:,i]- np.dot(store[...,i-1], data[:,i-1] ) #possible optimization
            for pp in range(N):
                DF[i-1 + pp*lenreading] = df[pp+1]    # em for missing data could come here   
        
        #print(np.asarray(DF[::20]))
        
        for i in range(numsetA + numsetB):
            tmp[i] = meanprior[i] - vecparams[i]
            
        ybar = np.concatenate((DF, tmp))
        Jbar = np.vstack( (hwhole, np.eye(Lenvec, dtype = np.float64 )) )
    
    #%% M step
        zr = np.zeros((sz, Lenvec), dtype = np.float64 )
        Cebartop  = np.hstack( (np.diag(lam), zr) )
        Cebardown = np.hstack( (zr.T, Cthprior ))
        Cebar = np.vstack( (Cebartop, Cebardown) )
        
        tmpsolve = solve(Cebar, Jbar)
        
        P = inv(Cebar) - np.dot( np.dot(tmpsolve, solve( np.dot(Jbar.T,tmpsolve), Jbar.T ) ) , inv( Cebar ) ) #possible optimization
    #%% continue M step
        ybart, zrvec = ybar.T, np.zeros((szvec,)) 
        for m in range(sz):
            randvar = zrvec
            randvar[m] = np.sum(P[m,:]*ybart)
            der[m] = -0.5*P[m,m] + 0.5* np.dot(np.dot(ybart, P.T ) , randvar ) #possible optimization
            randvar[m] = 0
        secder = -0.5*P*P.T
        secder = secder[:sz, :sz]
        
        secderr = 1/np.linalg.cond(secder)
        if secderr <1e-13:
            updatelam = 0
        else:
            updatelam = solve(secder,der)
        
        lam = lam - 0.6*updatelam;
        
        #%% Update parameters
        updateparams = solve( np.dot(Jbar.T, tmpsolve ), np.dot(Jbar.T, solve(Cebar,ybar))  ) #possible optimization

        oldparams = vecparams
        vecparams = oldparams + updateparams;
        
    #%% Update A,B
        for i in range(numsetA):
            A[ar_a[i], ac_a[i]] = vecparams[i]
            
        for i in range(numsetB):
            B[ar_b[i],ac_b[i],bi_b[i]] = vecparams[i + N2]
    
        for j in range( lenreading ): 
            tmpoldsum = np.asfortranarray( np.sum(B * Uvec[:,j,:] ,2) )
            for pp in range(N1):
                for qq in range(N1):
                    oldsum[pp,qq,j] = tmpoldsum[pp,qq]
    
        for i in range( lenreading ):
            storetmp = expm(np.asarray( creatematrix(A, oldsum[...,i], t[i+1] -t[i], N1 ) ) )
            store[..., i] = storetmp
            
        for i in range( lenreading ):
            tmpxq = np.dot(store[...,i],xq[:,i]) #possible optimization
            xq[:,i+1] = tmpxq
            
        diff[n1] = diffnorm( data[1:,1:], xq[1:,1:], N, lenreading ) 
        
        #print(np.linalg.norm(xq) )
        #print(diff[n1] )

        while diff[n1] > diff[n]:
            updateparams *= 0.6
            vecparams = oldparams + updateparams
            #updateMatrices() 
            #updateMatrices(A,B,ar_a,ac_a, numsetA, ar_b, ac_b, bi_b, numsetB, lenreading, N, N2, vecparams, diff, oldsum, xq, store, data)                       
            for i in range(numsetA):
                A[ar_a[i], ac_a[i]] = vecparams[i]
                
            for i in range(numsetB):
                B[ar_b[i],ac_b[i],bi_b[i]] = vecparams[i + N2]
        
            for j in range( lenreading ): 
                tmpoldsum = np.asfortranarray( np.sum(B * Uvec[:,j,:] ,2) )
                for pp in range(N1):
                    for qq in range(N1):
                        oldsum[pp,qq,j] = tmpoldsum[pp,qq]
        
            for i in range( lenreading ):
                storetmp = expm(np.asarray( creatematrix(A, oldsum[...,i], t[i+1] -t[i], N1 ) ) )
                store[..., i] = storetmp
                
            for i in range( lenreading ):
                tmpxq = np.dot(store[...,i],xq[:,i]) #possible optimization
                xq[:,i+1] = tmpxq
                
            diff[n1] = diffnorm( data[1:,1:], xq[1:,1:], N, lenreading ) 
    
    print(np.asarray(A[1:,1:]))
    print(np.asarray(B[1:,:,0]))