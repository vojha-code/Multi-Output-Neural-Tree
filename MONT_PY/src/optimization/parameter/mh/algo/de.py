# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:02:30 2019

@author: yl918888
"""

from src.optimization.parameter.mh.mh_cost_function import costFunction
from src.optimization.structure.misc import sort_by_values

import numpy as np
import random

class DifferentialEvolution():   
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
            Run Defrential evoltion optimization
        '''
        print('The Deferential Evolution:', self.mParams.n_algo_param)
        self.mEvaluateTree.set_dataset_to_evaluate('train')       
        
        currParameter = self.mTree.getTreeParameters(self.mParams.n_max_target_attr, self.mParameterToFetch)
        print('Best tree parameter length', len(currParameter) ,' to start : ', self.fobj(currParameter))
        self.performance_record.append(self.fobj(currParameter, False))
        
        min_x = self.mParams.n_weight_range[0] + 0.1
        max_x = self.mParams.n_weight_range[1]
        # DE Parameter
        mut = 0.8
        crossp = 0.7
        
        popsize = self.mParams.n_mh_pop_size
        #dimension of the solution
        dimensions = len(currParameter)
        # random initial population
        pop = np.random.rand(popsize, dimensions)
        pop[0] = currParameter
        #min_b, max_b = np.asarray(bounds).T
        #diff = np.fabs(min_b - max_b)
        #pop_denorm = min_b + pop * diff
        fitness = np.asarray([self.fobj(ind) for ind in pop])
        
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        for i in range(self.mParams.n_param_opt_max_itr):
            for j in range(popsize):
                idxs = [idx for idx in range(popsize) if idx != j]
                x_1, x_2, x_3 = pop[np.random.choice(idxs, 3, replace = False)]
                x_t = pop[j]
                 #--- MUTATION (step #3.A) ---------------------+
                # subtract x3 from x2, and create a new vector (x_diff)
                x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
                # multiply x_diff by the mutation factor (F) and add to x_1
                v_donor = [x_1_i + mut * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
                v_donor = self.fix_boundry(v_donor, min_x, max_x)
                #--- RECOMBINATION (step #3.B) ----------------+
                v_trial = []
                for k in range(len(x_t)):
                    crossover = random.random()
                    if crossover <= crossp:
                        v_trial.append(v_donor[k])
                    else:
                        v_trial.append(x_t[k])
                #--- GREEDY SELECTION (step #3.C) -------------+
                f = self.fobj(v_trial)
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = v_trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = v_trial
                        #print('MH Itr:',i,' has a new best : ', f)
                        
            self.performance_record.append(self.fobj(best, False))
            if(np.mod(i, 10) == 0):
                print('MH Itr: ',i,' best: ', fitness[best_idx])
            #iteration finished
            
        self.mTree.setTreeParameters(best, self.mParams.n_max_target_attr, self.mParameterToFetch)
        #currParameter = self.mTree.getTreeParameters(self.mParams.n_max_target_attr, self.mParameterToFetch)
        #print('Check sum: ',np.sum(np.subtract(currParameter,best)))
        #print('LAST Best tree parameter length', len(currParameter) ,' to Finish : ', self.fobj(currParameter))
            
        return self.mTree, self.performance_record 
 

    def fobj(self,pVector, only_error = True):
        return costFunction(pVector, self.mTree, self.mEvaluateTree, self.mParams.n_max_target_attr, self.mParameterToFetch, only_error)

    def fix_boundry(self, alist, min_x = 0.1, max_x = 1.0):
        for i in range(len(alist)):
            if alist[i] < min_x:
                alist[i] = min_x
            if alist[i] > max_x:
                alist[i] = max_x
        return alist
    
                