# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:49:00 2019

@author: yl918888
"""
import math
import os
import numpy as np
import matplotlib.pyplot as plt    

#Function to sort by values
def sort_by_values(aList, values):
    '''
        Sorting each individuals by their objective values
        Acending order sorting (lowesr value first) and gradualy increasing
        reutrns a list of index
    '''
    sorted_list = []
    while(len(sorted_list)!=len(aList)):
        if values.index(min(values)) in aList:
            sorted_list.append(values.index(min(values)))
        values[values.index(min(values))] = math.inf # Initilize min to infinit to avoid computing it
    return sorted_list

#Function to find index of list
def index_of(val, inList):
    for i in range(0,len(inList)):
        if inList[i] == val:
            return i
    return -1


def unique_elements(plist, printList = False): 
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in plist: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    
    if printList:
        for x in unique_list: 
            print ('Uniques list:',x)
    return unique_list

def index_in_list(plistElemennt, pList):
    index_list = []
    for x in plistElemennt:
        index_list.append(pList.index(x))
    
    return index_list

def preserve_diversity(pPopulation, pop_size):
    '''
        Preserve Elit and Diverser Individuals
        args:
            param:  itrPopulation       populaiton of individuslas, 
            param:  pop_size            number of individuals 
        return: a sorted and diverse population based on tree fitness and error
    '''
    nSortedPopulation = [pPopulation[i] for i in sort_by_values([i for i in range(0,len(pPopulation))], [pop.mCost[0] for pop  in pPopulation])]
    nSortedPopulationFitness = [[round(pop.mCost[0], 2), pop.mTree.getTreeSize()] for pop in nSortedPopulation]
    return [nSortedPopulation[i] for i in index_in_list(unique_elements(nSortedPopulationFitness),nSortedPopulationFitness)]      



    