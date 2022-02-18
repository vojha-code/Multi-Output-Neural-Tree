# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:17:28 2019

@author: yl918888
"""



from src.optimization.structure.gp.gp import GP
from src.optimization.structure.nsgpii.nsgp_ii import NSGPII
from src.optimization.structure.nsgpiii.nsgp_iii import NSGPIII

class StructureOptimization():
    '''
        Class calling functions for the stucture optimization algorithms
    '''
    mEvaluateTree = None # Tree evaluation paramters
    mParams = None #  set of paramters
    def __init__(self, pEvaluateTree, pParams):
        '''
        
        '''
        self.mEvaluateTree = pEvaluateTree
        self.mParams = pParams
        
    def optimize(self, directory, trail):
        '''
            Call an optimization anlogrithm
        '''
        if self.mParams.n_algo_structure == 'gp':
            print('Evaluating tree using GP....')
            gp = GP(self.mEvaluateTree, self.mParams)
            return gp.start(directory, trail)
        
        if self.mParams.n_algo_structure == 'nsgp_2':
            print('Evaluating tree using NSGP-II....')
            nsgpii = NSGPII(self.mEvaluateTree, self.mParams)
            return nsgpii.start(directory, trail)
        
        if self.mParams.n_algo_structure == 'nsgp_3':
            print('Evaluating tree using NSGP-III....')
            nsgpiii = NSGPIII(self.mEvaluateTree, self.mParams)
            return nsgpiii.start(directory, trail)
            #return None
        
        