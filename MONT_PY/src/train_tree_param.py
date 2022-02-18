# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:53:18 2019

Training steps of neural tree

@Author: Varun Ojha, Uni of Reading, UK
@Licence: Not decieded

"""

#%reset -f
#%clear
#%% ---------------------------------------------------------------------------------------------------------------------------------------------
# custom made class dependencies
from dataloader.data_loader import data_evaluation
from dataloader.data_loader import data_partition
from dataloader.data_processing import DataProcessing
from datetime import datetime
from optimization.parameter.parameter_optimization import ParameterOptimization
from optimization.structure.genetic_operators import GeneticOperator
from optimization.structure.structure_optimization import StructureOptimization
from reporting.display_tree import DisplayTree
from reporting.plots import *
from setting.setting_params import SettingParamter
from tree.evaluate_tree import EvaluateTree
from tree.neural_tree import NeuralTree
# python dependencies
import numpy as np
import os
import time
import multiprocessing as mp
#%%Optimization methods ---------------------------------------------------------------------------------------------------------
def crossValidationRun(evaluateTree, params, optimumTreeStructure):
    # Parameter optimization
    treeParameter = ParameterOptimization(evaluateTree, params, optimumTreeStructure)
    foldTree, _ =  treeParameter.optimize()
    train_fold = optimalTreeFitness(evaluateTree, foldTree, 'train')
    test_fold = optimalTreeFitness(evaluateTree, foldTree, 'test')
    return [train_fold, test_fold]

def crossValidationRunMP(evaluateTree, params, optimumTreeStructure, send_end):
    # Parameter optimization
    treeParameter = ParameterOptimization(evaluateTree, params, optimumTreeStructure)
    foldTree, _ =  treeParameter.optimize()
    train_fold = optimalTreeFitness(evaluateTree, foldTree, 'train')
    test_fold = optimalTreeFitness(evaluateTree, foldTree, 'test')
    send_end.send([train_fold, test_fold])


def optimizePARM(params, directory, trail, optimize_tree_parameter = True, optimumTreeStructure = None):
    '''
    Optimize paramters
    Argument:
        params: paramters
        trail:  time
        optimize_tree_parameter:  True or False
    '''
    performance_dict = {}
    dataProcessing = DataProcessing()
    dataProcessing.setParams(params) # setting it for data preporcessing

    print('\n Fetch a Tree Structure for optimization....')    
    #dummay  tree for testing
    #if optimumTreeStructure == None:
    #    optimumTreeStructure = NeuralTree()
    #    optimumTreeStructure.genrateGenericRandomTree(params)
    runMP = True
    # Go for parameter optimization
    if params.n_validation_method == 'k_fold' or params.n_validation_method == 'five_x_two_fold':
        k = params.n_validation_folds
        # CROSS VALIDATION
        if params.n_validation_method == 'k_fold':
            data_input_values, data_target_values, random_sequence = data_partition(dataProcessing, params.n_validation_method)
            performance_dict.update({'data random sequence': random_sequence})
            start = datetime.now()
            avg_train = []
            avg_test  = []
            if k == 2:
                nFOLDS = 1
            else:
                nFOLDS = k  
                
            if runMP:
                jobs = []
                pipe_list = []
                for fold in range(nFOLDS):
                    recv_end, send_end = mp.Pipe(False)
                    evaluateTree = data_evaluation(params, data_input_values, data_target_values, params.n_validation_method, fold)
                    p = mp.Process(target=crossValidationRunMP, args=(evaluateTree, params, optimumTreeStructure, send_end))
                    jobs.append(p)
                    pipe_list.append(recv_end)
                    p.start()
                for proc in jobs:
                    proc.join()                
                for fold in range(nFOLDS):
                    foldRes = pipe_list[fold].recv() 
                    train_fold = foldRes[0]
                    test_fold = foldRes[1]
                    print('\nTrain on fold', fold, ': ', train_fold[0])
                    print('Test  on fold', fold, ': ', test_fold[0])
                    avg_train.append(train_fold)
                    avg_test.append(test_fold)
            else:
                for fold in range(nFOLDS):
                    evaluateTree = data_evaluation(params, data_input_values, data_target_values, params.n_validation_method, fold)
                    train_fold, test_fold = crossValidationRun(evaluateTree, params, optimumTreeStructure)
                    print('\nTrain on fold', fold, ': ', train_fold[0])
                    print('Test  on fold', fold, ': ', test_fold[0])
                    avg_train.append(train_fold)
                    avg_test.append(test_fold)
                #end for
            #end else
            print('\nAverange train on all folds: ', sum([row[0] for row in avg_train])/len(avg_train))
            print('Averange test  on all folds: ', sum([row[0] for row in avg_test])/len(avg_test))

        if params.n_validation_method == 'five_x_two_fold':
            data_input_values, data_target_values, _ = data_partition(dataProcessing, params.n_validation_method)
            start = datetime.now()
            avg_train = []
            avg_test  = []
            if runMP:
                jobs = []
                pipe_list = []
                for fold in range(5):
                    for cross in range(2):
                        recv_end, send_end = mp.Pipe(False)
                        evaluateTree = data_evaluation(params, data_input_values, data_target_values, params.n_validation_method, fold, cross)
                        p = mp.Process(target=crossValidationRunMP, args=(evaluateTree, params, optimumTreeStructure, send_end))
                        jobs.append(p)
                        pipe_list.append(recv_end)
                        p.start()
                for proc in jobs:
                    proc.join()         
                    
                j = 0
                for fold in range(5):
                    for cross in range(2):
                        foldRes = pipe_list[j].recv() 
                        train_fold = foldRes[0]
                        test_fold = foldRes[1]
                        print('\nTrain on fold', fold, cross,': ', train_fold[0])
                        print('Test  on fold', fold, cross,': ', test_fold[0])
                        avg_train.append(train_fold)
                        avg_test.append(test_fold)
                        j = j + 1 
            else:
                for fold in range(5):
                    for cross in range(2):
                        evaluateTree = data_evaluation(params, data_input_values, data_target_values, params.n_validation_method, fold, cross)
                        # Parameter optimization
                        train_fold, test_fold = crossValidationRun(evaluateTree, params, optimumTreeStructure)
                        print('\nTrain on fold', fold, cross,': ', train_fold[0])
                        print('Test  on fold', fold, cross,': ', test_fold[0])
                        avg_train.append(train_fold)
                        avg_test.append(test_fold)
            print('\nAverange train on all folds: ', sum([row[0] for row in avg_train])/len(avg_train))
            print('Averange test  on all folds: ', sum([row[0] for row in avg_test])/len(avg_test))
        # End of parameter optimization
        end = datetime.now()
        # COLLECTION CV paramters
        elpsade_time = elpsadeTime(start, end)
        print('TIME: cv optimization:', elpsade_time, 'sec.')
        performance_dict.update({'gd/mh cv error' : [avg_train, avg_test]})
        performance_dict.update({'gd/mh average cv error' : [sum([row[0] for row in avg_train])/len(avg_train), sum([row[0] for row in avg_test])/len(avg_test)]})
        performance_dict.update({'gd/mh cv time (sec)' : elpsade_time})
        #Save paramters
        filePath = os.path.join(directory, 'treePerfroamnce_dict_mh_'+ str(trail))
        np.save(filePath, performance_dict)
        ##read_np = np.load('model/treePerfroamnce_Dict_iris_trail.npy').item()
        #return performance_dict
    
#%%
def readTreeStructure(directory, data_file):
    fileTree = data_file+".json"
    fileTree = os.path.join(os.path.join(directory, 'selected_sig_tree'), fileTree)
    print('File',data_file, fileTree)
    d_tree = DisplayTree() # Creating object fo0r display tree
    return d_tree.retriveTreeFromFile(fileTree)

#%%
if __name__ == '__main__':
    starttime = time.time()
    for i in range(1):
        data_file_set = ['iris']#,             #0
                         #'wisconsin',        #1
                         #'wine',             #2
                         #'heart.csv',        #3
                         #'ionosphere.csv',   #4
                         #'australian.csv',   #5
                         #'pima.csv',         #6
                         #'glass.csv',        #7
                         #'vehicle.csv']      #8
                         #'yeast.csv',        #9
                         #'optdigits.csv']    #10 
                         #'adult.csv',
                         #'kddcup.csv',
                        
        # iterate over all data
        #data_file = data_file_set[4]
        processes = []
        
        for data_file in  data_file_set:
            print('Running dtata file', data_file)
            n_problem_type = 'Classification' # 'Regression' # 'Classification'
            param_opt = 'mh'
            is_norm = True
            isOptimizeParam = False
            normalize = [0.0, 1.0]
            trail = data_file.split('.')[0] + '_' + str(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
            #trail_time = data_file.split('.')[0] + '_trail'
            #trail = '01'
            #Setting working directory
            path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
            #path = os.getcwd()
            directory = os.path.join(os.path.join(path, 'model'), trail)
            #directory = os.path.join(os.path.join(os.getcwd(), 'model'), trail)
            #print('PATH',path, '\nDirectory',directory)
            try:
                os.makedirs(directory)
            except OSError:
                print ("Directory already exisit or fails to create one")
            else:
                print ("Successfully created the directory %s" % directory)
            
            if data_file.find(".csv") > -1:
                file_name = data_file[:-4]
            else:
                file_name = data_file
                
            optimumTreeStructure = readTreeStructure(os.path.join(path, 'model'), file_name)
            params = SettingParamter.setParameter(data_file, n_problem_type, param_opt, is_norm, normalize)
            ## Training
            #optimizePARM(params, directory, trail, isOptimizeParam, optimumTreeStructure)
            p = mp.Process(target=optimizePARM, args=(params, directory, trail, isOptimizeParam, optimumTreeStructure,))
            processes.append(p)
            p.start()
        # end for
        for process in processes:
            process.join()
    #end for
    print('That took {} seconds'.format(time.time() - starttime))

#%%

