# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:42:43 2019

@author: yl918888
"""

class Params():
    '''
    '''
    nPop = None #  numbe rof population
    Zr = None #  Refernce points
    nZr = None #  size of reference points
    zmin = None # 
    zmax = None
    smin = None
    
    def __init__(self, pPop, pZr, pnZr, pZmin, pZmax, pSmin):
        self.nPop = pPop
        self.Zr = pZr
        self.nZr = pnZr
        self.zmin = pZmin
        self.zmax = pZmax
        self.smin = pSmin