# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:21:50 2019

@author: yl918888
"""

# General GA imports

#Imports from NSGA-III functions

from src.optimization.structure.cost_functions import costFunction
from src.optimization.structure.genetic_operators import GeneticOperator
from src.optimization.structure.individual import Individual
from src.optimization.structure.misc import *
from src.optimization.structure.misc import preserve_diversity
from src.optimization.structure.nsgpiii.generateReferencePoints import generateReferencePoints
from src.optimization.structure.nsgpiii.params import Params
from src.optimization.structure.nsgpiii.sortAndSelectPopulation import sortAndSelectPopulation
from src.reporting.plots import plotParetoFront
from src.reporting.plots import saveGPIteration
from src.tree.neural_tree import NeuralTree
# Python standard libraries
import copy
import numpy as np
import random


class NSGPIII:
    '''
    NSGA - II algorithim based on the implementation:
        https://nbviewer.jupyter.org/github/lmarti/nsgaiii/blob/master/NSGA-III%20in%20Python.ipynb
    '''
    mEvaluateTree = None # Tree evaluation paramters
    mParams = None #  set of paramters
    
    
    # Setting variables 
    mOptimization =  None #  minimization 'MIN' or maximization 'MAX'

    # NSGA    Parameter setting
    mDivision = None # Generating Reference Points 
    
    mMaxIt = None # Maximum Number of Iterations
    mPop = None # Population Size
    
    pCrossover = None # Crossover Percentage
    mCrossover = None # Number of Parnets (Offsprings)
    pMutation = None # Mutation Percentage
    mMutation = None # Number of Mutants
    
    mu = None # Mutation Rate
    sigma = None # Mutation Step Size
    
    cParams = None # Params for the refernce points
    mPopulation = None # population paramter
    mF = None #  Fonts
    
    mBestIndividual = None
    
    performance_record = []
    
    def __init__(self, pEvaluateTree, pParams):
        '''
            Setting param for nsgp to run
        '''
        self.mEvaluateTree = pEvaluateTree
        self.mParams = pParams
        
        # optimization type minimization and maximization
        self.mOptimization = self.mParams.n_optimization 
        # Setup for Number of objectives in the test functions              
        
        # NSGA    Parameter setting
        self.mDivision = self.mParams.n_division #  number of division/sprade of the points between 0 and 1 for an objective function 
        self.mMaxIt = self.mParams.n_max_itrations # Maximum Number of Iterations
        self.mPop = self.mParams.n_max_population # Population Size
        
        self.pCrossover =  self.mParams.n_prob_crossover # Crossover Percentage
        self.mCrossover = 2*round(self.pCrossover*self.mPop/2) # Number of Parnets (Offsprings)
        self.pMutation = 1.0 - self.pCrossover # Mutation Percentage
        self.mMutation = round(self.pMutation*self.mPop) # Number of Mutants
        
        self.mBestIndividual = Individual()
    
        self.performance_record = []
    def start(self, directory, trail):
        
        '''
            Initilization
        '''
        # Generating initial refernce points systmaticaly
        mZr = generateReferencePoints(self.mParams.n_max_objectives, self.mDivision) #  tested OK      
        # collecting and initilizing reference points values 
        self.cParams = Params(self.mPop, mZr, mZr.shape[1], [], [], [])
        
        # Creating initial populations
        self.mPopulation = [Individual() for i in range(0, self.mPop)] # Empty population
        # Generating population fron a unifiorm distribution
        for pop in self.mPopulation:
            n_tree = NeuralTree() #  initating a class of tree
            n_tree.genrateGenericRandomTree(self.mParams)
            pop.mTree = n_tree
            pop.mCost = costFunction(self.mEvaluateTree, 'train', pop.mTree)
        
        # 1c. Sortining of the population according to their approximation error/ classifcation error
        self.mPopulation = [self.mPopulation[i] for i in sort_by_values([i for i in range(0,len(self.mPopulation))], [pop.mCost[0] for pop  in self.mPopulation])]        
        
        self.setBestTree(self.mPopulation[0]) 
        self.performance_record.append(costFunction(self.mEvaluateTree, 'train', self.mBestIndividual.mTree, 2, False))
        
        self.mPopulation, self.mF, self.cParams = sortAndSelectPopulation(self.mPopulation, self.cParams, self.mOptimization)
        return self.nsgpLoop(directory, trail)
        

    def nsgpLoop(self, directory, trail):
        '''
            Loop through genetic process
        '''
        save_interval = int(self.mMaxIt*0.1)
        for itr in range(self.mMaxIt):
            self.mPopulation = preserve_diversity(self.mPopulation, self.mPop)
            #print(' # diverse inds = ', len(self.mPopulation), end=' ')
            #generate new individuals if number of diverse individuals less than total population size
            while(len(self.mPopulation) < self.mPop):
                ind = Individual()
                n_tree = NeuralTree() #  initating a class of tree
                n_tree.genrateGenericRandomTree(self.mParams)
                ind.mTree = n_tree
                ind.mCost = costFunction(self.mEvaluateTree, 'train', ind.mTree, 2) #  1 for one objetive
                self.mPopulation.append(ind)

            #STEP 2: Generating offsprings ---------------------------------------------------------------------
            # 2a. copy population in to itrPopulation
            itrPopulation = [copy.deepcopy(self.mPopulation[i]) for i  in range(0, len(self.mPopulation))]
            countCorssover, countMutation = 0, 0            
            while(len(itrPopulation) < 2*self.mPop):
                # 2b. creating reference for crossopver and mutation oprators
                operator = GeneticOperator(self.mParams)
                if self.mParams.n_prob_crossover/2.0 < random.random() and countCorssover < 2*round(self.mParams.n_prob_crossover*self.mPop/2):
                    # CROSSOVER OPERATION
                    #print('crossover')
                    p1 = random.randint(0,self.mPop-1)
                    p2 = random.randint(0,self.mPop-1)
                
                    nParentTree1 = copy.deepcopy(self.mPopulation[p1].mTree) 
                    nParentTree2 = copy.deepcopy(self.mPopulation[p2].mTree) 
                
                    firstTree, secondTree, is_crossover_done = operator.crossoverTree(nParentTree1,nParentTree2)
                    if is_crossover_done:
                        child1 = Individual()                            
                        child1.mTree = firstTree
                        child1.mCost = costFunction(self.mEvaluateTree, 'train', child1.mTree, 2) 
                        itrPopulation.append(child1)

                        child2 = Individual()                            
                        child2.mTree = secondTree
                        child2.mCost = costFunction(self.mEvaluateTree, 'train', child2.mTree, 2) 
                        itrPopulation.append(child2)
                        countCorssover += 2
                        # just for cleaning up -  not necessary
                        del nParentTree1, nParentTree2, firstTree, secondTree, child1, child2
                else:
                    # MUTATION OPEARATION
                    #print('mutation')
                    p1 = random.randint(0,self.mPop-1)
                    nParentTree = copy.deepcopy(self.mPopulation[p1].mTree) 
                    m_tree = operator.mutation(nParentTree)
                    child = Individual()                            
                    child.mTree = m_tree
                    child.mCost = costFunction(self.mEvaluateTree, 'train', child.mTree, 2) 
                    itrPopulation.append(child)
                    countMutation += 1            
            #Generating offsprings END---------------------------------------------------------------------
            # SAVE - all populaion and for post processing
            self.mPopulation, self.mF, self.cParams = sortAndSelectPopulation(itrPopulation, self.cParams, self.mOptimization)
            # Iteration Information
            #print('Iteration ', itr,' Number of F1 Members = ', len(self.mF[0]))
            #for f in self.mF[0]:
                #print([self.mPopulation[f].mCost[0],self.mPopulation[f].mCost[1]], end=' ')
            #print('\n')
            print_best = self.setBestTree([self.mPopulation[i] for i in sort_by_values([i for i in range(0,len(self.mPopulation))], [pop.mCost[0] for pop  in self.mPopulation])][0], itr+1) 
            self.performance_record.append(costFunction(self.mEvaluateTree, 'train', self.mBestIndividual.mTree, 2, False))
            #if(int(np.mod(itr,10)) == 0) and not print_best:
            #    print('GP Itr', itr+1, 'best tree [',  round(self.mBestIndividual.mCost[0], 5), self.mBestIndividual.mTree.getTreeSize(),']')
            if (int(np.mod(itr,save_interval)) == 0):#if Save Itration = True
                saveGPIteration(self.mPopulation, directory, trail, str(itr), 'nsgp_3_')
                
        #Ploting the last front
        if True:  # plotFinal = True 
            # ploting only the font
            x = [self.mPopulation[popIndex].mCost[0] for popIndex in self.mF[0]]
            y = [self.mPopulation[popIndex].mCost[1] for popIndex in self.mF[0]]
            plotParetoFront(x,y, directory, trail, 'nsgp_3_', 'treeParetoFront_')
            # ploting the whole population
            x = [pop.mCost[0] for pop in self.mPopulation]
            y = [pop.mCost[1] for pop in self.mPopulation]
            plotParetoFront(x,y, directory, trail, 'nsgp_3_')
            saveGPIteration(self.mPopulation, directory, trail, 'final', 'nsgp_3_')
        

                
        return self.mBestIndividual.mTree, self.mPopulation, self.performance_record # returning the best tree

    def setBestTree(self, pIndividual, itr = 0):
        '''
            Compare the current individial tree with the best Tree found so far
            arges:
                param:  pIndividual  an individual tree
        '''
        print_best = False
        if self.mBestIndividual.mTree  == None:
            self.mBestIndividual.mTree = pIndividual.mTree
            self.mBestIndividual.mCost = pIndividual.mCost
            print_best = True

        if self.mBestIndividual.mCost[0] > pIndividual.mCost[0]:
            self.mBestIndividual.mTree = pIndividual.mTree
            self.mBestIndividual.mCost = pIndividual.mCost
            print_best = True
        
        if self.mBestIndividual.mCost[0] == pIndividual.mCost[0] and self.mBestIndividual.mTree.getTreeSize() > pIndividual.mTree.getTreeSize():
            self.mBestIndividual.mTree = pIndividual.mTree
            self.mBestIndividual.mCost = pIndividual.mCost
            print_best = True
        
        #if print_best:
        #    print('GP Itr', itr, 'best tree [',  round(self.mBestIndividual.mCost[0], 5), self.mBestIndividual.mTree.getTreeSize(),'] this is new best')
        
        return print_best
            
            
            