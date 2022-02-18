# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:29:05 2019

@author: yl918888
"""

import numpy as np

def normalizePopulation(pPopulation, pParams):
    '''
        Normalkization of the population
        args:
            param:    pPopulation       population  informations
            param:    pParams           set of referncepoints parameters
        return:  normailizaed population
    '''
    # Computer ideal points, i,e. Zmin_i for i = 1 to M objectives
    pParams.zmin = updateIdealPoint(pPopulation, pParams.zmin)
    
    # Transplate objectives, i.e. objective_values_i - zmin_i , for i = 1 to M objectives
    nfp = [] # list of list 
    for pop in pPopulation:
        nfp.append(np.subtract(pop.mCost, pParams.zmin).tolist())
    
    # Computer extream points, i.e., Zmax_i for i = 1 to M objectives, 
    pParams = performScalarizing(nfp, pParams)
    
    # Computer intercepts a_i
    nInterceptFound = True
    try:
        nZnadir_point = findHyperplaneIntercepts(pParams.zmax, pParams.zmin)
    except:
        nInterceptFound = False
        
    # if no nZnadir_point found , fall back to worst objective  
    if not nInterceptFound:
        nZnadir_point = maxInObjectives(pPopulation)
        
    # nZnadir_point must be significantly larger in each objective 
    epsilin_ndir = 10 ** -6
    for i in range(len(nZnadir_point)):
        if nZnadir_point[i] - pParams.zmin[i] < epsilin_ndir:
            nZnadir_point[i] = maxInObjectives(pPopulation)[i]
            
    # Normalize objectives by deviding translated objectives by intercept a_i
    for i in range(len(pPopulation)):
        pPopulation[i].mNormalizedCost = np.divide(np.asanyarray(nfp[i]), nZnadir_point).tolist()
    
    return pPopulation, pParams
    
def updateIdealPoint(pPopulation, pPrevZmin):
    '''
        Updateing ideal points for the refernce - which is the minimum cost of all 
        args:
            param:    pPopulation       population  informations
            param:    pPrevZmin         previous referncepoint mean
        return:  reference point mean
    '''
    if len(pPrevZmin) == 0:
        pPrevZmin = [np.Inf for j in range(len(pPopulation[0].mCost))] # assign inf to each obj functions
        
    nZmin = pPrevZmin
    for pop in pPopulation:
        nZmin = np.minimum(nZmin, pop.mCost).tolist()
        
    return nZmin

def performScalarizing(pfp, pParams):
    '''
    perfromaing scalrization
    Args: 
        pfp: translated objective values
        pParam:  contains zmean and slo used to retun zmax (extrem points)
        return pParam (Zmax and smin)
    '''
    nObj =  len(pfp[0]) # number of objectives
    
    if len(pParams.smin) != 0:
        zmax  = pParams.zmax
        smin  = pParams.smin
    else:
        zmax  = np.zeros((nObj, nObj)).tolist()
        smin  = [np.Inf for j in range(nObj)]
        
    for j in range(nObj):
        w = getScalarizingVector(nObj, j)
        s = []
        for fpVector in pfp:
            s.append(np.max(np.divide(fpVector, w)))    
          
        sminj = min(s)
        indxOfsmin = s.index(sminj)
        
        if sminj < smin[j]:
            zmax[j] = pfp[indxOfsmin]
            smin[j] = sminj
              
    pParams.zmax = zmax
    pParams.smin = smin
    
    return pParams
    

def getScalarizingVector(p_obj, p_index_j):
    '''
        Scalrized vectotr
        args:
            param:    p_obj          number of objectives
            param:    p_index_j      the index to be scalarized
        return sclarized vector
    '''
    epsilon = 10 ** -10
    w = (epsilon*np.ones(p_obj)).tolist()
    w[p_index_j] = 1

    return w

def findHyperplaneIntercepts(zmax, zmin):
    '''
        Computing hyperplane intercepts
    '''
    zmax = np.asmatrix(zmax)
    ones = np.ones(zmax.shape[1])
    zmaxInvs = np.linalg.inv(zmax)
    plane = np.dot(ones,zmaxInvs)
    intercept = np.asarray(np.divide(1,plane)).flatten()
    znadir = np.add(intercept,zmin)
    return znadir


def maxInObjectives(pPopulation):
    '''
        Updateing ideal points for the refernce - which is the minimum cost of all 
        args:
            param:    pPopulation       population  informations
            param:    pPrevZmin         previous referncepoint mean
        return:  maximum in objective
    '''
    nMaxInObj = [0 for j in range(len(pPopulation[0].mCost))] # assign inf to each obj functions
    for pop in pPopulation:
        nMaxInObj = np.maximum(nMaxInObj, pop.mCost).tolist()
    return nMaxInObj