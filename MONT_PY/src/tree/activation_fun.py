'''
Created on Sat Apr 20 12:42:42 2019

Training steps of neural tree
Author: v ojha
Affiliation: Uni of Reading

'''
#from scipy.stats import norm
import math
import numpy as np
import random

class ActivationFunction:
    '''
        This calss implements activation function for the nodes
    '''
    m_function_param = [] # List of paramters of a function
    m_net_wighted_sum = -99 # Set an arbitarary weighed sum at the node
    m_bias = -99 # Set an arbitarary weighed sum at the node
    
    def __init__(self, p_function_param, p_net_weighted_sum, p_bias):
        '''
            Constractor that sets the param values
            param: p_function_para is the parameter of the desfined function
            param: p_net_weighted_sum is the value on which activatin/quasing/transfer function is to be applied.  
        '''
        self.m_function_param = p_function_param
        self.m_net_wighted_sum = p_net_weighted_sum
        self.m_bias = p_bias
        
    def value(self):
        '''
           Implementation of of different functions       
        '''  
        #check what function is at the node
        if (self.m_function_param.__contains__('Gaussian')):
            #print('Using Gaussian')
            return self.Gaussian()
        if (self.m_function_param.__contains__('tanh')): 
            #print('Using tanh')
            return self.tanh()
        if (self.m_function_param.__contains__('sigmoid')): 
            #print('Using sigmoid')
            return self.sigmoid()
        if (self.m_function_param.__contains__('ReLU')): 
            #print('Using ReLU')
            return self.ReLU()
        if (self.m_function_param.__contains__('softmax')): 
            #print('Using softmax')
            return self.softmax()
        
        
    def Gaussian(self):
        '''
            Implementation of Gausian function
            Guassian function is from: 
                https://docs.scipy.org/doc/scipy/reference/stats.html
            Node that Guassiab function has no use of bias
        '''
        #Define the consants
        x = self.m_net_wighted_sum
        mu = self.m_function_param[0]
        sigma = self.m_function_param[1]  
        if sigma == 0:
            sigma = random.uniform(0.1,1.0) # this was the error
            self.m_function_param[1]  =  sigma
        # See defination of Gaussian here: https://en.wikipedia.org/wiki/Gaussian_function
        x = float(x - mu ) / sigma # e -(1/2)((x-mu)/sigma)^2
        return math.exp(-x*x/2.0) / math.sqrt(2.0*math.pi) / sigma
    
    def tanh(self):
        '''
            Implementation of tanh function
            tanh function is from: 
                http://mathworld.wolfram.com/HyperbolicTangent.html
        '''
        #Define the consants
        wx = self.m_net_wighted_sum
        b = self.m_bias
        x = wx + b
        return np.tanh(x)
        #return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
    
    def sigmoid(self):
        '''
            Implementation of sigmoid function
            sigmoid function is from: 
                https://en.wikipedia.org/wiki/Sigmoid_function
        '''
        #Define the consants
        wx = self.m_net_wighted_sum
        b = self.m_bias
        x = wx + b
        
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1.0 / (1.0 + math.exp(-x))
        
    
    
    def ReLU(self):
        '''
            Implementation of ReLU function
            ReLU function is from: 
                https://en.wikipedia.org/wiki/Sigmoid_function
        '''
        #Define the consants
        wx = self.m_net_wighted_sum
        b = self.m_bias
        x = wx + b
        return max(0,x)
    
    def softmax(self):
        '''
            Implementation of ReLU function
            ReLU function is from: 
                https://en.wikipedia.org/wiki/Sigmoid_function
        '''
        #Define the consants
        wx = self.m_net_wighted_sum
        b = self.m_bias
        x = wx + b
        # Softmans is not used in this case as it dependts on other output nodes - 
        # this can only be used at output layrof a calssifcation pronblem sor its impley return x
        return x
    
    
        