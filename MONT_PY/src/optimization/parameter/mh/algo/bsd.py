# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:41:19 2019

@author: yl918888
"""

from src.optimization.parameter.mh.mh_cost_function import costFunction
from src.optimization.structure.misc import sort_by_values

import numpy as np

class BernstainSearchDE():   
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
            Run Bernstain Search Deferential Evolution optimization
            
        '''
        print('The Bernstain Search Deferential Evolution: ', self.mParams.n_algo_param)
        self.mEvaluateTree.set_dataset_to_evaluate('train')       
        
        currParameter = self.mTree.getTreeParameters(self.mParams.n_max_target_attr, self.mParameterToFetch)
        print('Best tree parameter length', len(currParameter) ,' to start : ', self.fobj(currParameter))
        self.performance_record.append(self.fobj(currParameter, False))
        
        min_x = self.mParams.n_weight_range[0]
        max_x = self.mParams.n_weight_range[1]

        N = self.mParams.n_mh_pop_size
        #dimension of the solution
        D = len(currParameter)
        # random initial population
        pop = np.random.rand(N, D)
        #min_b, max_b = np.asarray(bounds).T
        #diff = np.fabs(min_b - max_b)
        #pop_denorm = min_b + pop * diff
        fitness = np.asarray([self.fobj(ind) for ind in pop])
        
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        for itr in range(self.mParams.n_param_opt_max_itr):            
            M = np.zeros((N, D))
            for i in range(N):
                alpha = self.getAlpha()
                select = int(np.ceil(alpha * D))
                u = np.random.choice([x for x in range(D)], select, replace = False)
                for k in u:
                    M[i,k] = 1
            
            if pow(np.random.rand(),3) < np.random.rand():
                F = pow(np.random.rand(1, D),3) * abs(pow(np.random.randn(1, D ),3))
            else:
                F = pow(np.random.randn(N, 1),3)
            
            sequence = [x for x in range(N)]
            while True:
                L1 = np.random.choice(sequence, N, replace = False)
                L2 = np.random.choice(sequence, N, replace = False)
                if sum(L1 == sequence) == 0 and sum(L1 == L2) ==0 and sum(L2 == sequence) ==0:
                    #print('break')
                    break
            
            w1 = np.random.rand(N, D)
            E = w1* pop[L1,:] + (1 - w1)* pop[L2,:]
            
            w2 = 1 - pow(np.random.rand(N, 1),3)
            v_trial = pop +  F * M * (w2 * E + ( 1 - w2 ) * best - pop)            
            for i in range(len(v_trial)):
                v_trial[i] = self.fix_boundry(v_trial[i], min_x, max_x)        
                #--- GREEDY SELECTION (step #3.C) -------------+
                f = self.fobj(v_trial[i])
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = v_trial[i]
                    if f < fitness[best_idx]:
                        best_idx = i
                        best = v_trial[i]
                        #print('MH Itr:',i,' has a new best : ', f)
                        
            self.performance_record.append(self.fobj(best, False))
            if(np.mod(itr, 10) == 0):
                print('MH Itr: ',itr,' best: ', fitness[best_idx])
            #iteration finished

        self.mTree.setTreeParameters(best, self.mParams.n_max_target_attr, self.mParameterToFetch)
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
    
    def getAlpha(self):
        beta = np.random.rand()
        kappa = np.ceil(3 * (pow(np.random.rand(),3)))
        if kappa == 1:
            #case 1
            return pow(beta,2)
        if kappa == 2:
            #case 2
            return 2*(1-beta)*beta
        if kappa == 3:
            #case 3
            return pow((1-beta),2)