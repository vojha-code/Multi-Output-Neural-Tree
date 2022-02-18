# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:00:43 2019

@author: yl918888

"""
import random
import copy
import numpy as np

from src.tree.neural_tree import NeuralTree
from src.reporting.display_tree import DisplayTree

from src.optimization.structure.individual import Individual
from src.optimization.structure.cost_functions import costFunction
from src.reporting.plots import plotParetoFront
from src.reporting.plots import saveGPIteration
from src.optimization.structure.misc import *
from src.optimization.structure.misc import preserve_diversity
from src.optimization.structure.genetic_operators import GeneticOperator

class GP():
    '''
        Implementaion  of NSGP-II algorithm from K. Deb
    '''
    mEvaluateTree = None # Tree evaluation paramters
    mParams = None #  set of paramters
    mBestIndividual = None # find the best individual tree
    performance_record = None

    def __init__(self, pEvaluateTree, pParams):
        '''
            Setting param for nsgp to run
        '''
        self.mEvaluateTree = pEvaluateTree
        self.mParams = pParams
        self.mBestIndividual = Individual()
        self.performance_record = []


    def start(self, directory, trail):

        pop_size = self.mParams.n_max_population

        print('GP initial population: Tree index [Error, Size]')
        # STEP 1: INITIALIZE POPULATION -------------------------------------------------------------------------
        # 1a. Empty individuals - population is a objeect of individuals
        nPopulation = [Individual() for i in range(0, pop_size)] # Empty population
        # 1b. Create individual tree and evaluate fitness of each individual tree
        for pop in nPopulation:
            n_tree = NeuralTree() #  initating a class of tree
            n_tree.genrateGenericRandomTree(self.mParams)
            pop.mTree = n_tree
            pop.mCost = costFunction(self.mEvaluateTree, 'train', pop.mTree, 1) #  2 for two objetive
            #print('Tree', nPopulation.index(pop), '[',round(pop.mCost[0], 2), pop.mTree.getTreeSize(),']')
        # 1c. Sortining of the population according to their approximation error/ classifcation error
        nPopulation = [nPopulation[i] for i in sort_by_values([i for i in range(0,len(nPopulation))], [pop.mCost[0] for pop  in nPopulation])]
        #print('\nInitial sorted population: Tree index [Error Size]')
        #for pop in nPopulation:
            #print('Tree', nPopulation.index(pop), '[',round(pop.mCost[0], 2), pop.mTree.getTreeSize(),']')

        self.setBestTree(nPopulation[0])
        self.performance_record.append(costFunction(self.mEvaluateTree, 'train', self.mBestIndividual.mTree, 2, False))
        
        itr = 0
        while (itr < self.mParams.n_max_itrations):     
            #STEP 2: Generating offsprings ---------------------------------------------------------------------
            # 2a. copy population in to itrPopulation
            itrPopulation = [copy.deepcopy(nPopulation[i]) for i  in range(0, len(nPopulation))]
            countCorssover, countMutation = 0, 0
            while(len(itrPopulation) < 2*pop_size):
                # 2b. creating reference for crossopver and mutation oprators
                operator = GeneticOperator(self.mParams)
                if self.mParams.n_prob_crossover/2.0 < random.random() and countCorssover < 2*round(self.mParams.n_prob_crossover*pop_size/2):
                    # CROSSOVER OPERATION
                    #print('crossover')
                    p1 = random.randint(0,pop_size-1)
                    p2 = random.randint(0,pop_size-1)

                    nParentTree1 = copy.deepcopy(nPopulation[p1].mTree)
                    nParentTree2 = copy.deepcopy(nPopulation[p2].mTree)

                    firstTree, secondTree, is_crossover_done = operator.crossoverTree(nParentTree1,nParentTree2)
                    if is_crossover_done:
                        child1 = Individual()
                        child1.mTree = firstTree
                        child1.mCost = costFunction(self.mEvaluateTree, 'train', child1.mTree, 1)
                        itrPopulation.append(child1)

                        child2 = Individual()
                        child2.mTree = secondTree
                        child2.mCost = costFunction(self.mEvaluateTree, 'train', child2.mTree, 1)
                        itrPopulation.append(child2)
                        countCorssover += 2
                        # just for cleaning up -  not necessary
                        del nParentTree1, nParentTree2, firstTree, secondTree, child1, child2
                else:
                    # MUTATION OPEARATION
                    #print('mutation')
                    p1 = random.randint(0,pop_size-1)
                    nParentTree = copy.deepcopy(nPopulation[p1].mTree)
                    m_tree = operator.mutation(nParentTree)
                    child = Individual()
                    child.mTree = m_tree
                    child.mCost = costFunction(self.mEvaluateTree, 'train', child.mTree, 1)
                    itrPopulation.append(child)
                    countMutation += 1

            #STEP 3:  Mainatain Elitisum and diversity and sorting a pop_size
            nDiversePop = preserve_diversity(itrPopulation, pop_size)
            #print(' # diverse inds = ', len(nDiversePop), end=' ')
            #for pop in nDiversePop:
                #print('Tree [',round(pop.mCost[0], 2), pop.mTree.getTreeSize(),']')

            #generate new individuals if number of diverse individuals less than total population size
            while(len(nDiversePop) < pop_size):
                ind = Individual()
                n_tree = NeuralTree() #  initating a class of tree
                n_tree.genrateGenericRandomTree(self.mParams)
                ind.mTree = n_tree
                ind.mCost = costFunction(self.mEvaluateTree, 'train', ind.mTree, 1) #  1 for one objetive
                nDiversePop.append(ind)

            nPopulation = [copy.deepcopy(nDiversePop[i]) for i  in range(0, pop_size)]
            # 3b. Sortining of the current population according to their approximation error/ classifcation error
            nPopulation = [nPopulation[i] for i in sort_by_values([i for i in range(0,len(nPopulation))], [pop.mCost[0] for pop  in nPopulation])]
            print_best = self.setBestTree(nPopulation[0], itr+1)
            itrPopulation = [] # reset the iteration population
            #print(costFunction(self.mEvaluateTree, 'train', self.mBestIndividual.mTree, 2, False))
            self.performance_record.append(costFunction(self.mEvaluateTree, 'train', self.mBestIndividual.mTree, 2, False))
            #if(int(np.mod(itr,10)) == 0) and not print_best:
            #    print('GP Itr', itr+1, 'best tree [',  round(self.mBestIndividual.mCost[0], 5), self.mBestIndividual.mTree.getTreeSize(),']')
            
            if False:#if Save Itration = True
                saveGPIteration(nPopulation, directory, trail, str(itr))
            itr = itr + 1
            
        
        #print('GP Itr', itr+1, 'best tree [',  round(self.mBestIndividual.mCost[0], 5), self.mBestIndividual.mTree.getTreeSize(),']')
        if True: #plotFinal = True
            x = [nPopulation[i].mCost[0] for i in range(len(nPopulation))]
            y = [nPopulation[i].mTree.getTreeSize() for i in range(len(nPopulation))]
            plotParetoFront(x,y, directory, trail)
            saveGPIteration(nPopulation, directory, trail, 'final')
            
        # returning the best tree, population, perfromance record for each iteration
        return self.mBestIndividual.mTree, nPopulation, self.performance_record 
    
    #--------------------------------------------------------------------------------------------------------------------------------
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

        if self.mBestIndividual.mCost[0] > pIndividual.mCost[0]: # and abs(self.mBestIndividual.mCost[0] - pIndividual.mCost[0]) < 0.000001:
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