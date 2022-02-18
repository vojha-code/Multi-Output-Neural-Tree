# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:38:09 2019

@author: yl918888
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:02:30 2019

@author: yl918888
"""

from src.optimization.parameter.mh.mh_cost_function import costFunction
from src.optimization.structure.misc import sort_by_values

import numpy as np
import random
import copy
import math

class SimulatedAnnealing():   
    mEvaluateTree = None # Tree evaluation paramters
    mParams = None #  set of paramters
    mTree = None #  set of paramters
    mParameterToFetch = 'weights_and_bias'
    performance_record = None

    
    def __init__(self, pEvaluateTree, pParams, pTree):
        self.mEvaluateTree = pEvaluateTree
        self.mParams = pParams
        self.mTree = pTree
        self.performance_record = []
        if(pParams.n_fun_type == 'Gaussian'):
            self.mParameterToFetch = 'all'
        else:
            self.mParameterToFetch = 'weights_and_bias'
        
    
    #------------------------------------------------------------------------------------------    
    def optimize(self):
        '''
            Run Simulated Annealing optimization
            
        '''
        print('The Simulated Annealing: ', self.mParams.n_algo_param)
        self.mEvaluateTree.set_dataset_to_evaluate('train')       
        
        currParameter = self.mTree.getTreeParameters(self.mParams.n_max_target_attr, self.mParameterToFetch)
        print('Best tree parameter length', len(currParameter) ,' to start : ', self.fobj(currParameter))
        self.performance_record.append(self.fobj(currParameter, False))
        
        min_x = self.mParams.n_weight_range[0]
        max_x = self.mParams.n_weight_range[1]
        # SA Parameter
        # Number of cycles
        n = self.mParams.n_param_opt_max_itr
        # Number of trials per cycle
        m = self.mParams.n_mh_pop_size
        # Number of accepted solutions
        na = 0.0
        # Probability of accepting worse solution at the start
        p1 = 0.7
        # Probability of accepting worse solution at the end
        pLast = 0.001
        # Initial temperature
        t1 = -1.0/math.log(p1)
        # Final temperature
        tLast = -1.0/math.log(pLast)
        # Fractional reduction every cycle
        frac = (tLast/t1)**(1.0/(n-1.0))
        
        # Initialize x
        x_start = copy.deepcopy(currParameter)
        # Current best results so far
        xc = x_start
        fc =  self.fobj(x_start)
        # Record the best
        xbest = x_start
        fbest = fc
        
        na = na + 1.0
        # Current temperature
        t = t1
        # DeltaE Average
        DeltaE_avg = 0.0        
        for i in range(n):
            #print('Cycle: ' + str(i) + ' with Temperature: ' + str(t))
            for j in range(m):
                # Generate new trial points
                xi = [x + random.random() - 0.5 for x in xc]
                # Clip to upper and lower bounds
                xi = self.fix_boundry(xi, min_x, max_x)
                fxi = self.fobj(xi)
                DeltaE = abs(fxi - fc)
                if (fxi > fc):
                    # Initialize DeltaE_avg if a worse solution was found
                    #   on the first iteration
                    if (i==0 and j==0): DeltaE_avg = DeltaE
                    # objective function is worse
                    # generate probability of acceptance
                    p = math.exp(-DeltaE/(DeltaE_avg * t))
                    # determine whether to accept worse point
                    if (random.random()<p):
                        # accept the worse solution
                        accept = True
                    else:
                        # don't accept the worse solution
                        accept = False
                else:
                    # objective function is lower, automatically accept
                    accept = True
                if (accept==True):
                    # update currently accepted solution
                    xc = xi
                    fc = fxi
                    if (fc < fbest):
                        xbest = copy.deepcopy(xc)
                        fbest = fc
                    # increment number of accepted solutions
                    na = na + 1.0
                    # update DeltaE_avg
                    DeltaE_avg = (DeltaE_avg * (na - 1.0) +  DeltaE) / na
            
            # Lower the temperature for next cycle
            t = frac * t                        
            self.performance_record.append(self.fobj(xbest, False))
            if(np.mod(i, 10) == 0):
                print('MH Itr: ',i,' best: ', fbest)
            #iteration finished
            
        self.mTree.setTreeParameters(xbest, self.mParams.n_max_target_attr, self.mParameterToFetch)
        return self.mTree, self.performance_record 
 

    def fobj(self,pVector, only_error = True):
        return costFunction(pVector, self.mTree, self.mEvaluateTree, self.mParams.n_max_target_attr, self.mParameterToFetch, only_error)

    def fix_boundry(self, alist, min_x = 0.0, max_x = 1.0):
        for i in range(len(alist)):
            if alist[i] < min_x:
                alist[i] = min_x
            if alist[i] > max_x:
                alist[i] = max_x
        return alist
    
                