# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:08:20 2019

@author: yl918888
"""

def costFunction(pEvaluation, pSet, pTree, pObjs = 2, only_error = True):
    '''
        Compute a vector of objective functions
        args:
            param:  pEvaluation  data to evaluate
            param:  pSet type of dataset
            param:  pTree the tree on which to evaluate 
            param:  pObjs defult is 2 for nsga-2 and nsga-3 
    '''
    if only_error:
        if pObjs == 1:
            pEvaluation.set_dataset_to_evaluate(pSet)
            _ = pEvaluation.getTreePredictedOutputs(pTree) #  collected output prediction note used here
            return [pEvaluation.getTreeFitness()[0]]
    
        if pObjs == 2:        
            pEvaluation.set_dataset_to_evaluate(pSet)
            _ = pEvaluation.getTreePredictedOutputs(pTree) #  collected output prediction note used here
            return [pEvaluation.getTreeFitness()[0], pTree.getTreeSize()]
    else:
        if pObjs == 1:
            pEvaluation.set_dataset_to_evaluate(pSet)
            _ = pEvaluation.getTreePredictedOutputs(pTree) #  collected output prediction note used here
            return pEvaluation.getTreeFitness()
    
        if pObjs == 2:        
            pEvaluation.set_dataset_to_evaluate(pSet)
            _ = pEvaluation.getTreePredictedOutputs(pTree) #  collected output prediction note used here
            return pEvaluation.getTreeFitness() +  [pTree.getTreeSize()]
