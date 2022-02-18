# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:14:32 2019

@author: yl918888
"""

def bound_check(bound, parameter):
    '''
     Check min and max of an element in the parameter
     args:
         param:     bound           is a list of two items: [min, max]
         param:     parameter       is vector of real values
     return:  a parameter set to boundary of the papremater
    '''
    if len(bound) >= 2:
        minimum = bound[0] 
        maximum = bound[1] 
        
        for elm in range(len(parameter)):
            if parameter[elm] < minimum:
                # if the element of paramter is less then min value then set it ot min value
                parameter[elm]  = minimum
            if parameter[elm] > maximum:
                # if the element of paramter is greter then max value then set it ot max value
                parameter[elm]  = maximum
    else:
        print('First argument of bound_check takes a list of two items [min, max]')
    
    # return the parameter with or without check
    return parameter

