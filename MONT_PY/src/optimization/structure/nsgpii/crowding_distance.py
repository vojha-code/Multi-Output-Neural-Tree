# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:48:27 2019

@author: yl918888
"""
import numpy as np
from src.optimization.structure.misc import sort_by_values

#Function to calculate crowding distance
def crowding_distance(P, front):
    '''
        Recieves a population and its a set of nondominated solutions (form front 0 to ...)
        args:
            param:    P              population
            param:    front = I     front  (index of individuals in P) a nondominated sorted front
        
        return: distance
    '''
    #P = population
    #l = len(front) # l = |I| , i.e., inds_in_this_font
    distance = [0 for i in range(0,len(front))] # for each i, set I[i]distance = 0
    no_obj = len(P[0].mCost)

    # Initilize a infinit -  here we have very large number
    distance[0] = 99999999999
    distance[len(front) - 1] = 99999999999
    #for each objective m
    for m in range(no_obj):
        #fetch values of the objetcive functions
        values = [P[i].mCost[m] for i in range(len(P))]
        #sort for values of the objecive m
        sortedFont = sort_by_values(front, values[:])
        #fmin = min(values)
        #fmax = max(values)
        fmin = values[sortedFont[0]]
        fmax = values[sortedFont[len(front)-1]]
        # for i = 2 to l -1 -  here indext start fron 1
        for k in range(1,len(front)-1):
            if fmax - fmin == 0:
                distance[k] = 99999999999
            else:
                distance[k] = distance[k] + (values[sortedFont[k+1]] - values[sortedFont[k-1]])/(fmax - fmin)
            
    return distance

