# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:12:55 2019

@author: yl918888
"""

import math
import numpy as np

def generateReferencePoints(M, p):
    Zr = getFixedRowSumIntegerMatrix(M, p)
    Zr = Zr.transpose() 
    return np.divide(Zr,p)


def getFixedRowSumIntegerMatrix(M, RowSum):
    if M < 1:
        print('M cannot be less than 1.')
    
    if math.floor(M) != M:
        print('M must be an integer.')
    
    if M == 1:
        return [RowSum]
    
    for i in range(0,RowSum+1):
        B = getFixedRowSumIntegerMatrix(M - 1, RowSum - i)
        B = np.asmatrix(B)
        #print(len(B))
        itrMat = i*np.ones(len(B))
        itrMat = itrMat.astype(int)        
        itrMat = np.asmatrix(itrMat)
        itrMat = itrMat.transpose()
        itrMat = np.append(itrMat, B, axis = 1)
        #print(itrMat)
        if i == 0:
            A = itrMat #  To make sure A is a matrix
        else:
            A = np.append(A, itrMat, axis = 0) # Appending each itrMat as an row to itself
            
    return A