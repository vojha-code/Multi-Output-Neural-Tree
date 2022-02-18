# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:02:10 2019

@author: yl918888
"""
    
import numpy as np
import copy
from src.optimization.parameter.mcs.mcs_opt import TheMCS
from src.optimization.parameter.mh.algo.sa import SimulatedAnnealing
from src.optimization.parameter.mh.algo.de import DifferentialEvolution
from src.optimization.parameter.mh.algo.bsd import BernstainSearchDE
from src.optimization.parameter.mh.algo.pso import PSO

class Metaheuristics():
    
    mEvaluateTree = None # Tree evaluation paramters
    mParams = None #  set of paramters
    mTree = None #  set of paramters
    
    def __init__(self, pEvaluateTree, pParams, pTree):
        self.mEvaluateTree = pEvaluateTree
        self.mParams = pParams
        self.mTree = pTree
        
        
    def start(self):
        '''
             Optimizae a vector using 
        '''
        if self.mParams.n_algo_param ==  'mcs': #'mcs'
            mcs = TheMCS(self.mEvaluateTree,self.mParams,self.mTree)
            return mcs.optimize()
        
        if self.mParams.n_algo_param == 'sa': # 'sa'
            sa = SimulatedAnnealing(self.mEvaluateTree,self.mParams,self.mTree)
            return sa.optimize()
        
        if self.mParams.n_algo_param == 'de': 
            de = DifferentialEvolution(self.mEvaluateTree,self.mParams,self.mTree)
            return de.optimize()
        
        if self.mParams.n_algo_param == 'bsde': #'bsde'
            bsde = BernstainSearchDE(self.mEvaluateTree,self.mParams,self.mTree)
            return bsde.optimize()   
        
        if self.mParams.n_algo_param == 'pso': #'pso'
            pso = PSO(self.mEvaluateTree,self.mParams,self.mTree)
            return pso.optimize()


            
            
            

        