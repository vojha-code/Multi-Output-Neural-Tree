# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:02:40 2019

@author: yl918888
"""

from src.optimization.parameter.mh.mh_cost_function import costFunction
import random
import math
import numpy as np

class Particle():
    '''
        An individual in NSGA population
    '''
    mPosition = [] #  particle position
    mVelocity = [] # particle velocity
    mPositionBest = [] # particle best position

    mCostPosition = -1 # particle cost
    mCostPositionBest = -1 #  best cost 
    
    def __init__(self, min_x, max_x, currParameter):
        # intialize positition from using the randum pertubation to the best particle
        self.mPosition = [min_x + (max_x - min_x)*random.random() for x in currParameter]
        #self.mPosition = [min_x + (max_x - min_x)*random.random() + x for x in currParameter]
        self.mVelocity = [random.uniform(-1,1) for i in range(0, len(currParameter))]

    def evaluate(self, pCostPotition):
        self.mCostPosition = pCostPotition
        # determine if it is trhe best particle
        # If fun(x) < fun(p), then set p = x. 
        # This step ensures p has the best position the particle has seen.
        if self.mCostPosition < self.mCostPositionBest or self.mCostPositionBest == -1:
            self.mPositionBest = self.mPosition
            self.mCostPositionBest = self.mCostPosition
    
    # update new particle velocity
    def update_velocity(self, pPositionGroupBest, w = 0.729, c1 = 1.49445, c2 = 1.49445):
        '''
        Updateing particle velocity
        args:
            Group best position:  a vector of best position
            w=0.5       # constant inertia weight (how much to weigh the previous velocity)
            c1=1        # cognative constant
            c2=2        # social constant
        '''
        # v = W*v + c1*u1.*(p-x) + c2*u2.*(g-x).
        for i in range(0, len(self.mPosition)):
            u1 = random.uniform(0,1)
            u2 = random.uniform(0,1)

            vel_cognitive = c1*u1*(self.mPositionBest[i] - self.mPosition[i]) # self best pos - curr pos
            vel_social    = c2*u2*(pPositionGroupBest[i] - self.mPosition[i]) # self best pos group - curr pos
            self.mVelocity[i] = w*self.mVelocity[i] + vel_cognitive + vel_social # inartia +  congnative + social
    
    # update the particle position based off new velocity updates   
    def update_position(self, min_x = 0.0, max_x = 1.0):
        for i in range(0, len(self.mPosition)):
            # Update the position x = x + v.
            self.mPosition[i] = self.mPosition[i] + self.mVelocity[i]

            # Enforce the bounds. 
            # If any component of x is outside a bound, set it equal to that bound. 
            # For those components that were just set to a bound, 
            # if the velocity v of that component points outside the bound, set that velocity component to zero.
            if self.mPosition[i] < min_x:
                self.mPosition[i] = min_x
                if self.mVelocity[i] < min_x:
                    self.mVelocity[i] = 0.0
                
            # adjust maximum position if necessary
            if self.mPosition[i] > max_x:
                self.mPosition[i] = max_x
                if self.mVelocity[i] > max_x:
                    self.mVelocity[i] = 0.0
    

class PSO():   
    mEvaluateTree = None # Tree evaluation paramters
    mParams = None #  set of paramters
    mTree = None #  set of paramters
    mParameterToFetch = 'weights_and_bias'
    
    mSwarm = None
    mBestParticle = None

    dimension = None
    min_x = 0.0
    max_x = 1.0

    performance_record = None 
    
    def __init__(self, pEvaluateTree, pParams, pTree):
        self.mEvaluateTree = pEvaluateTree
        self.mParams = pParams
        self.mTree = pTree
        self.performance_record = []
        
        if(pParams.n_fun_type == 'Gaussian'):
            self.mParameterToFetch = 'all'
        else:
            self.mParameterToFetch = 'weights_and_bias'
            
            
    #------------------------------------------------------------------------------------------    
    def optimize(self):
        '''
            Run Particle Swarm Optimization optimization
        '''
        print('Particle Swarm Optimization:',self.mParams.n_algo_param)
        self.mEvaluateTree.set_dataset_to_evaluate('train')       
        currParameter = self.mTree.getTreeParameters(self.mParams.n_max_target_attr, self.mParameterToFetch)
        self.dimension = len(currParameter)
        print('Best tree parameter length', self.dimension ,' to start : ', costFunction(currParameter, self.mTree, self.mEvaluateTree, self.mParams.n_max_target_attr, self.mParameterToFetch))
                
        self.min_x = self.mParams.n_weight_range[0] - 0.5
        self.max_x = self.mParams.n_weight_range[1] - 0.5
        
        nSwarm = [Particle(self.min_x, self.max_x, currParameter) for i in range(self.mParams.n_mh_pop_size)]
        # Set one particle from best
        nSwarm[0].mPosition = currParameter
        self.evaluate(nSwarm[0])
        
        self.performance_record.append(costFunction(currParameter, self.mTree, self.mEvaluateTree, self.mParams.n_max_target_attr, self.mParameterToFetch, False))
        
        nPositionGroupBest = []   # best position for group
        nCostGroupBest = -1       # best error for group
        
        #pso paramters
        w = 1.0
        wdamp = 0.99
        c1 = 1.49445
        c2 = 1.49445
        # begin optimization loop
        itr = 0
        while itr < self.mParams.n_param_opt_max_itr:
            # evaluate swarm
            for particle in nSwarm:
                self.evaluate(particle)
            
                # determine if current particle is the best (Group)
                # If fun(x) < fun(g), then set g = x and fun(g) = fun(x). 
                # This step ensures b has the best objective function in the swarm, and d has the best location.
                if particle.mCostPosition < nCostGroupBest or nCostGroupBest == -1:
                    nPositionGroupBest = particle.mPosition # set Group best position
                    nCostGroupBest = float(particle.mCostPosition) # set Group best error
                    #print('Itr ', itr, ' new Best : ',nCostGroupBest)
                    
            # cycle through swarm and update velocities and position
            for particle in nSwarm:
                particle.update_velocity(nPositionGroupBest, w, c1, c2)
                particle.update_position(self.min_x, self.max_x)
            
            # inartia dampening
            w = w*wdamp
            
            self.performance_record.append(costFunction(nPositionGroupBest, self.mTree, self.mEvaluateTree, self.mParams.n_max_target_attr, self.mParameterToFetch, False))
            if(np.mod(itr, 10) == 0):
                print('MH Itr: ', itr, ' best: ', nCostGroupBest)
                
            itr += 1
        
        
        self.mTree.setTreeParameters(nPositionGroupBest, self.mParams.n_max_target_attr, self.mParameterToFetch)
        return self.mTree, self.performance_record 
        
        
        
    #----------------------------------------------------------------------------------------------------------------------------
    def evaluate(self, particle):
        particle.evaluate(costFunction(particle.mPosition, self.mTree, self.mEvaluateTree, self.mParams.n_max_target_attr, self.mParameterToFetch))
        
                
