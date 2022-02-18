# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 07:04:55 2019

@author: yl918888
"""

%reset -f
%clear
#%%----------------------------------------------------------------------------
import numpy as np

from optimization.parameter.mcs.jones.defaults import defaults
from optimization.parameter.mcs.mcs import mcs
from optimization.parameter.mcs.jones.functions import feval
#%%----------------------------------------------------------------------------
fcn = 'hm6' # gpr, bra, cam, hm3, s10, sh5, sh7, hm6

u,v,nglob,fglob,xglob = defaults(fcn)
#feval(fcn, [2,5])
# function paramters
n = len(u);		         # problem dimension
prt = 1 # print level
smax = 5*n+10 # number of levels used
nf = 50*pow(n,2) #limit on number of f-calls
stop = [3*n]  # m, integer defining stopping test
stop.append(float("-inf"))  # freach, function value to reach

m = 1
if m == 0:
    stop[0] = 1e-4	 # run until this relative error is achieved
    stop[1] = fglob	 # known global optimum value
    stop.append(1e-10) # stopping tolerance for tiny fglob

iinit = 0 # 0: simple initialization list
local = 50 	# local = 0: no local search
eps = 2.220446049250313e-16
gamma = eps		         # acceptable relative accuracy for local search
hess = np.ones((n,n)) 	     # sparsity pattern of Hessian

#%%call mcs algorithm
xbest,fbest,xmin,fmi,ncall,ncloc,flag = mcs(fcn,u,v,smax,nf,stop,iinit,local,gamma,hess)

print('The MCS Algorithms Results:')
print('fglob',fglob)
print('fbest',fbest)
print('xglob',xglob)
print('xbest',xbest)
print('\n')
