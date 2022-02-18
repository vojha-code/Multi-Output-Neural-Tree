# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:00:43 2019

@author: yl918888

"""
# General GA imports
from src.optimization.structure.cost_functions import costFunction
from src.optimization.structure.genetic_operators import GeneticOperator
from src.optimization.structure.individual import Individual
from src.optimization.structure.misc import *
from src.optimization.structure.misc import preserve_diversity
# NSGA-II specifica functions
from src.optimization.structure.nsgpii.crowding_distance import crowding_distance
from src.optimization.structure.nsgpii.fast_non_dominated_sort import fast_non_dominated_sort
from src.reporting.plots import plotParetoFront
from src.reporting.plots import saveGPIteration
#Tree functions
from src.tree.neural_tree import NeuralTree
# Python standard libraries
import copy
import numpy as np
import random


class NSGPII():
    '''
        Implementaion  of NSGA-II algorithm from K. Deb
    '''
    mEvaluateTree = None # Tree evaluation paramters
    mParams = None #  set of paramters
    mBestIndividual = None
    performance_record = None
    
    def __init__(self, pEvaluateTree, pParams):
        '''
            Setting param for nsga to run
        '''
        self.mEvaluateTree = pEvaluateTree
        self.mParams = pParams
        self.mBestIndividual = Individual()
        self.performance_record = []
        
    #-------------------------------------------------------------------------------------------------------------
    def start(self, directory, trail):
        
        pop_size = self.mParams.n_max_population
        
        print('NSGPII initial population: Tree index [Error, Size]')
        # STEP 1: INITIALIZE POPULATION -------------------------------------------------------------------------
        # 1a. Empty individuals - population is a objeect of individuals
        nPopulation = [Individual() for i in range(0, pop_size)] # Empty population         
        # 1b. Create individual tree and evaluate fitness of each individual tree 
        for pop in nPopulation:
            n_tree = NeuralTree() #  initating a class of tree
            n_tree.genrateGenericRandomTree(self.mParams)
            pop.mTree = n_tree
            pop.mCost = costFunction(self.mEvaluateTree, 'train', pop.mTree, 2) #  1 for one objetive
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
            nPopulation = preserve_diversity(nPopulation, pop_size)
            #print(' # diverse inds = ', len(nPopulation), end=' ')
            #generate new individuals if number of diverse individuals less than total population size
            while(len(nPopulation) < pop_size):
                ind = Individual()
                n_tree = NeuralTree() #  initating a class of tree
                n_tree.genrateGenericRandomTree(self.mParams)
                ind.mTree = n_tree
                ind.mCost = costFunction(self.mEvaluateTree, 'train', ind.mTree, 2) #  1 for one objetive
                nPopulation.append(ind)
            
            # Non dominated sorting
            non_dominated_sorted_solution = fast_non_dominated_sort(nPopulation, self.mParams.n_optimization)
            #print("\nThe best front for Generation number ", itr, " is", len(non_dominated_sorted_solution[0])," : ", non_dominated_sorted_solution[0])
            #for values in non_dominated_sorted_solution[0]:
            #    print([round(x,3) for x in nPopulation[values].mCost], end = " ")
            
            crowding_distance_values = []
            for i in range(0,len(non_dominated_sorted_solution)):
                crowding_distance_values.append(crowding_distance(nPopulation,non_dominated_sorted_solution[i][:]))
            
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
                    p1 = random.randint(0,pop_size-1)
                    nParentTree = copy.deepcopy(nPopulation[p1].mTree) 
                    m_tree = operator.mutation(nParentTree)
                    child = Individual()                            
                    child.mTree = m_tree
                    child.mCost = costFunction(self.mEvaluateTree, 'train', child.mTree, 2) 
                    itrPopulation.append(child)
                    countMutation += 1            
            #Generating offsprings END---------------------------------------------------------------------
             
            #Non dominated sorting
            non_dominated_sorted_solution_gen = fast_non_dominated_sort(itrPopulation, self.mParams.n_optimization)
            #crowding distance computation
            crowding_distance_values_gen=[]            
            for i in range(0,len(non_dominated_sorted_solution_gen)):
                crowding_distance_values_gen.append(crowding_distance(itrPopulation, non_dominated_sorted_solution_gen[i][:]))
            #print(crowding_distance_values_gen)

            new_solution= []
            for i in range(0,len(non_dominated_sorted_solution_gen)):
                #SORTING of the forn
                #Retriving postion (index) of the individuals in the nondominated font
                non_dominated_sorted_solution_gen_indIndexList = [index_of(non_dominated_sorted_solution_gen[i][j],non_dominated_sorted_solution_gen[i]) for j in range(0,len(non_dominated_sorted_solution_gen[i]))]
                #Sortining front index by its crowding distance values
                front_assending_order_indexList = sort_by_values(non_dominated_sorted_solution_gen_indIndexList[:], crowding_distance_values_gen[i][:])
                # retriving index of individuals based on the sorted position(index) of the individual in the nondominated font
                front = [non_dominated_sorted_solution_gen[i][front_assending_order_indexList[j]] for j in range(0,len(non_dominated_sorted_solution_gen[i]))]
                #reverse the list to get the decending order of crowding distance
                front.reverse()
                # Consider the indeividual into new population
                for value in front:
                    new_solution.append(value)
                    if(len(new_solution)==pop_size):
                        break
                if (len(new_solution) == pop_size):
                    break
                
            nPopulation = [copy.deepcopy(itrPopulation[i]) for i  in new_solution]
            del itrPopulation
            print_best = self.setBestTree([nPopulation[i] for i in sort_by_values([i for i in range(0,len(nPopulation))], [pop.mCost[0] for pop  in nPopulation])][0], itr+1)            
            self.performance_record.append(costFunction(self.mEvaluateTree, 'train', self.mBestIndividual.mTree, 2, False))
            #if(int(np.mod(itr,10)) == 0) and not print_best:
            #    print('GP Itr', itr+1, 'best tree [',  round(self.mBestIndividual.mCost[0], 5), self.mBestIndividual.mTree.getTreeSize(),']')
            if False:#if Save Itration = True
                saveGPIteration(nPopulation, directory, trail, str(itr), 'nsgp_2_')
            itr = itr + 1 
            # WHILE ENDS HERE
        print_best = self.setBestTree([nPopulation[i] for i in sort_by_values([i for i in range(0,len(nPopulation))], [pop.mCost[0] for pop  in nPopulation])][0], itr)
        
        #print('GP Itr', itr+1, 'best tree [',  round(self.mBestIndividual.mCost[0], 5), self.mBestIndividual.mTree.getTreeSize(),']')
        plotFinal = True
        if plotFinal:
            x = [nPopulation[i].mCost[0] for i in range(len(nPopulation))]
            y = [nPopulation[i].mCost[1] for i in range(len(nPopulation))]
            plotParetoFront(x,y, directory, trail, 'nsgp_2_')
            saveGPIteration(nPopulation, directory, trail, 'final', 'nsgp_2_')

        return self.mBestIndividual.mTree, nPopulation, self.performance_record # returning the best tree

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        #    print('GP Itr', itr+1, 'best tree [',  round(self.mBestIndividual.mCost[0], 5), self.mBestIndividual.mTree.getTreeSize(),'] this is new best')
