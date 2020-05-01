# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:16:03 2020

@author: TBE7
"""

import numpy as np
from scipy.linalg import expm
from scipy.interpolate import UnivariateSpline as spline
from scipy.interpolate import CubicSpline as cspline
from multiprocessing import Pool, cpu_count
from numpy.linalg import solve, inv
from defrms import rms
#%% Preliminaries
loadeddata = np.load('file.npz')
t, data, u, lengthu = loadeddata['t'], loadeddata['y'], loadeddata['u'], loadeddata['lenu'] 
N = data.shape[0]
N1 = N+1;
N2 = N**2;

#define setA, setB
setA, setB  = tuple(i for i in range(N2) ), tuple(i for i in range(N2, 2*N2 + N))
numsetA = len(setA)
ar_a = np.zeros([numsetA], dtype = np.int8)
ac_a = np.zeros_like( ar_a )
numsetB = len(setB);
bi_b = np.zeros([numsetB],dtype = np.int8);
ar_b = np.zeros([numsetB],dtype = np.int8);
ac_b = np.zeros([numsetB],dtype = np.int8);

#%%  define the parameters
count = -1;
count1 = numsetA - 1;
lenreading = data.shape[1] - 1
lenreadpone = lenreading+1;
Lenvec = numsetA+numsetB
sz = lenreading*N;
szvec = sz+Lenvec;
zn = np.zeros([1,N])
znt = np.transpose(zn)
vecparams = np.zeros([Lenvec,])
meanprior = np.copy( vecparams )

#%%
A = np.zeros([N1,N1], dtype = np.float64)
for i in range( numsetA ):
    ar_a[i] = np.floor(setA[i]/N) + 1
    ac_a[i] = setA[i] - ( ar_a[i] - 1)*N + 1
    A[ar_a[i],ac_a[i]] = 0.5 #np.random.random()
    vecparams[i] = A[ar_a[i] ,ac_a[i]]
Adel = np.copy(A)

B = np.zeros([N1,N1, lengthu], dtype = np.float64)

for i in range( numsetB ):
    location = setB[i] -N2
    bi_b[i] = np.floor(location/(N2+N))
    ar_b[i] = np.floor(( location )/N1 ) + 1
    ac_b[i] = location - N1*( ar_b[i] -1  )
    B[ar_b[i],ac_b[i],bi_b[i]] = 0.5 #np.random.random()
    vecparams[i + N2] = B[ar_b[i],ac_b[i],bi_b[i]]

#%% initialize oldsum
oldsum = np.zeros([N1,N1,lenreading], dtype = np.float64)
Uvec = np.transpose(u[..., np.newaxis],[2, 1, 0])
for j in range( lenreading ): 
    oldsum[...,j] = np.sum(B * Uvec[:,j,:] ,2)
sum_uB = oldsum[...,0]

#%% Monitors
Trend = np.zeros([Lenvec,30], dtype = np.float64)
firstorder = np.zeros([10,1]) #new line to monitor first order optimality
secderr = 1;


# %% Initialize xq
data = np.vstack( ( np.ones([1,lenreadpone]), data ) );
xq = np.copy( data ).astype(np.float64)

def expmoldsum(i, partialoldsum):
    return expm(( t[i+1] -t[i])*(A + partialoldsum ) );

store = np.zeros([N1, N1, lenreading], dtype = np.float64)

''' 
#parallel execution
pool = Pool() #defaults to number of available CPU's
chunksize = cpu_count()
gen = ((i, oldsum[...,i] ) for i in range(lenreading))
for ind, res in enumerate(pool.imap(expmoldsum, gen, lenreading//(chunksize) ) ):
    store[...,ind] = res   
''' 

for i in range( lenreading ):
    store[..., i] = expmoldsum(i, oldsum[...,i])
    
for i in range( lenreading ):
    xq[:,i+1] = np.dot(store[...,i],xq[:,i])   


# %% M-step variables
lam = np.ones((sz,), dtype = np.float64)
der = np.copy(lam ).astype( dtype = np.float64 )
secder = np.zeros([sz,sz], np.float64)
Cthprior  = np.diag( np.full( (Lenvec,), 20, dtype = np.float64 ) )  # larger premultipliers imply less confidence in priors

# %% Initialize variables for J matrix
hwhole = np.zeros([sz,Lenvec], dtype = np.float64 )
h = np.zeros([N1,lenreading], dtype = np.float64 )
h1 = np.zeros([N1,lenreading], dtype = np.float64 )
DF = np.copy(hwhole[:,1])

span = 2
ind = np.arange(-span, span + 1 )
dex = ind+span
tbsplinez = np.zeros([N1,2*span+1], dtype = np.float64 )
xrange = np.zeros([1,2*span+1], dtype = np.float64 )

#%% Loop utilities
#inner
delta = 0.01
delta1 = delta
tol = 1e-8
conv =1
n = -1
diff = np.full( [30,1], np.Inf )


#%% Loop
while conv > tol and n < 15:
    n += 1
    n1 = n + 1
    for k in range(numsetA):
        ar = ar_a[k]
        ac = ac_a[k]
        count += 1
        Arac = A[ar][ac]
        for i in range(lenreading):
            Adel = np.copy(A)
            tbspline = np.copy(tbsplinez)
            xrange = Arac + ind*delta
            for g in dex:
                Adel[ar][ac] = xrange[g]
                tbspline[:,g] = np.dot( expm((t[i+1] - t[i])*(Adel + oldsum[...,i])), data[:,i] )
            for row in range(1, tbspline.shape[0] ):
                #h[row,i] = spline(xrange, tbspline[row,:], k=4, s=0).derivative()(Arac)
                h[row,i] = cspline(xrange, tbspline[row,:]).derivative()(Arac)
        hwhole[:,count] = np.reshape(h[1:,], [sz,])
    count = -1
    
    
    for j in range(numsetB):
        ar = ar_b[j]
        ac = ac_b[j]
        bi = bi_b[j]
        count1 = count1 + 1
        B_ar_ac_bi = B[ar,ac,bi]
        sliced = oldsum[ar,ac,:]
        for i in range(lenreading):
            sum_uB = oldsum[:,:,i]
            tbspline = np.copy(tbsplinez)
            xrange = B_ar_ac_bi + ind*delta1
            for g in dex:
                sum_uB[ar,ac] = sliced[i] + u[bi,i]*ind[g]*delta1;
                tbspline[:,g] = np.dot( expm((t[i+1] - t[i])*(A + sum_uB)) , data[:,i] )
            for row in range(1, tbspline.shape[0]):
                h1[row,i] =  cspline(xrange, tbspline[row,:]).derivative()(B_ar_ac_bi)
        hwhole[:,count1] = np.reshape(h1[1:,:],[sz,])
    count1 = numsetA - 1
#%% ybar Jbar   
    for i in range(1,lenreadpone ):
        df = data[:,i]- np.dot(store[...,i-1], data[:,i-1] ) # data row is y, col is time
        DF[i-1: (N-1)*lenreading +i: lenreading] = df[1:]    # em for missing data could come here

    
    ybar = np.concatenate((DF, meanprior - vecparams))
    Jbar = np.vstack( (hwhole, np.eye(Lenvec, dtype = np.float64 )) )

#%% M step
    zr = np.zeros((sz, Lenvec), dtype = np.float64 )
    Cebartop  = np.hstack( (np.diag(lam), zr) )
    Cebardown = np.hstack( (zr.T, Cthprior ))
    Cebar = np.vstack( (Cebartop, Cebardown) )
    
    tmp = solve(Cebar, Jbar)
    P = inv(Cebar) - np.dot( np.dot(tmp, solve( np.dot(Jbar.T,tmp), Jbar.T ) ) , inv( Cebar ) )

#%% continue M step
    ybart, zrvec = ybar.T, np.zeros((szvec,)) 
    for m in range(sz):
        randvar = zrvec
        randvar[m] = np.sum(P[m,:]*ybart)
        der[m] = -0.5*P[m,m] + 0.5* np.dot(np.dot(ybart, P.T ) , randvar )
        randvar[m] = 0
    secder = -0.5*P*P.T
    secder = secder[:sz, :sz]
    
    
    secderr = 1/np.linalg.cond(secder)
    if secderr <1e-13:
        updatelam = 0
    else:
        updatelam = solve(secder,der)
    
    conv2 = rms(updatelam);
    lam = lam - 0.6*updatelam;
    #%% stability monitors
    
    #%% Update parameters
    
    updateparams = solve( np.dot(Jbar.T, tmp ), np.dot(Jbar.T, solve(Cebar,ybar))  )
    oldparams = vecparams
    vecparams = oldparams + updateparams;

#%% Update A,B
    for i in range(numsetA):
        A[ar_a[i], ac_a[i]] = vecparams[j]
    for i in range(numsetB):
        B[ar_b[i],ac_b[i],bi_b[i]] = vecparams[i + N2]

    for j in range( lenreading ): 
        oldsum[...,j] = np.sum(B * Uvec[:,j,:] ,2)

    for i in range( lenreading ):
        store[..., i] = expmoldsum(i, oldsum[...,i])
        
    for i in range( lenreading ):
        xq[:,i+1] = np.dot(store[...,i],xq[:,i]) 
        
    diff[n1] = np.linalg.norm(data[1:,1:] - xq[1:,1:])
    while diff[n1] > diff[n]:
        print('step halving')
        updateparams *= 0.6
        vecparams = oldparams + updateparams
            
        for i in range(numsetA):
            A[ar_a[i], ac_a[i]] = vecparams[i]
        for i in range(numsetB):
            B[ar_b[i],ac_b[i],bi_b[i]] = vecparams[i + N2]
    
        for j in range( lenreading ): 
            oldsum[...,j] = np.sum(B * Uvec[:,j,:] ,2)
    
        for i in range( lenreading ):
            store[..., i] = expmoldsum(i, oldsum[...,i])
            
        for i in range( lenreading ):
            xq[:,i+1] = np.dot(store[...,i],xq[:,i]) 
            
        diff[n1] = np.linalg.norm(data[1:,1:] - xq[1:,1:])







'''
Number = sol.y.shape[0]
lenreading = sol.y.shape[0] - 1
Lenvec = Number*(Number + lengthu*(Number+1))

N1 = Number + 1
sz = lenreading * Number
vecparams = np.zeros( [Lenvec, 1] )

Cthprior = np.diag(100* np.random.rand(Lenvec, 1))

A = np.random.rand(Number, Number)
zerostoprow = np.zeros([1,])
y = np.ones([2,2,3])
for i in range(12): y.itemset(i, i)
'''


'''
import numpy as np
setB = [i for i in range(16, 36)]
N1 = 5
N = 4
N2 = 16
numsetB = len(setB)
bi_b = np.zeros(N1*N, dtype = np.int8)
ar_b = np.zeros_like(bi_b)
ac_b = np.zeros_like(bi_b)
vecparams = np.zeros(36)

B = np.zeros([N1, N1, 1])

for i in range(25):
    B.itemset(i,i+ 16)

for i in range( numsetB ):
    location = setB[i] -N2+1
    bi_b[i] = np.ceil(location/(N2+N))-1
    pagefel = N2 + N1*N*(bi_b[i])
    ar_b[i] = np.floor(( setB[i] - pagefel )/N1 )
    ac_b[i] = (setB[i] - pagefel ) - N1*( ar_b[i] )
    vecparams[i + N2] = B[ar_b[i],ac_b[i],bi_b[i]]
            
'''