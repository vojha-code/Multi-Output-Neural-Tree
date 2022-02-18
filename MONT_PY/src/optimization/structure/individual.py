# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:42:21 2019

@author: yl918888
"""

class Individual():
    '''
        An individual in NSGA population
    '''
    mTree = None
    mCost = None
    mRank = None 
    mDominationSet = None
    mDominatedCount = None
    mNormalizedCost = None
    mAssociatedRef = None 
    mDistanceToAssociatedRef = None
    
    def __init__(self):
        self.mTree = None 
        self.mCost = [] 
        self.mRank = [] 
        self.mDominationSet = [] 
        self.mDominatedCount = [] 
        self.mNormalizedCost = [] 
        self.mAssociatedRef = [] 
        self.mDistanceToAssociatedRef = []   