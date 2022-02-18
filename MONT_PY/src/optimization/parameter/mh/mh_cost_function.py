# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:12:28 2019

@author: yl918888
"""

#------------------------------------------------------------------------------------------    
def costFunction(pParameter, pTree, pData, p_max_target_attr, eval_parama = 'all', only_error = True):
    '''
    computes parameter fitness.
    args:
        pParameter:         paramter vector to be set as tree parameter
        pTree:              tree to be evaluated
        pData:              training/test data to be evalauated
        p_max_target_attr:  max number of target attribute to examine number of child of tree trrot
        eval_parama:        'all' , 'weights' 'weights_and_bias' 'bias' 
    
    return: (float) error induced on pTree 
    '''
    if only_error:
        pTree.setTreeParameters(pParameter, p_max_target_attr, eval_parama)
        _ = pData.getTreePredictedOutputs(pTree)
        return pData.getTreeFitness()[0]
    else:
        pTree.setTreeParameters(pParameter, p_max_target_attr, eval_parama)
        _ = pData.getTreePredictedOutputs(pTree)
        return pData.getTreeFitness() +  [pTree.getTreeSize()]
