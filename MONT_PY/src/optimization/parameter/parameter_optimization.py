# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:00:11 2019

@author: yl918888
"""

from src.optimization.parameter.mh.meta_heuristics import Metaheuristics


class ParameterOptimization():
    '''
        Class calling functions for the stucture optimization algorithms
    '''
    mEvaluateTree = None # Tree evaluation paramters
    mParams = None #  set of paramters
    mTree = None #  set of paramters
    
    def __init__(self, pEvaluateTree, pParams, pTree):
        '''
        
        '''
        self.mEvaluateTree = pEvaluateTree
        self.mParams = pParams
        self.mTree = pTree
        
    def optimize(self):
        '''
            use a parametr optimization alogrithm to optimize tree's parameter
        '''       
        if self.mParams.n_param_optimizer == 'mh':
            mh = Metaheuristics(self.mEvaluateTree,self.mParams,self.mTree)
            return mh.start()
