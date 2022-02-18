# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:53:18 2019

Training steps of neural tree

@Author: Varun Ojha, Uni of Reading, UK
@Licence: GNU

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


#%Optimization methods ---------------------------------------------------------------------------------------------------------
def optimizeSTR(params, directory, trail,  optimize_tree_parameter = True):
    '''
    Optimize structure and paramters
    Argument:
        params: paramters
        trail:  time
        optimize_tree_parameter:  True or False
    '''
    performance_dict = {}
    dataProcessing = DataProcessing()
    dataProcessing.setParams(params) # setting it for data preporcessing

    print('\nStructure optimization....')
    # recieve data for tree tsructure optimization
    data_input_values, data_target_values, random_sequence = data_partition(dataProcessing)
    performance_dict.update({'data random sequence': random_sequence})
    evaluateTree = data_evaluation(params, data_input_values, data_target_values)
    # Structure
    start = datetime.now()
    treeStucture =  StructureOptimization(evaluateTree, params)
    optimumTreeStructure, tree_str_population, gp_perfromance = treeStucture.optimize(directory, trail)
    end = datetime.now()

    # PLOTS and MODEL SAVING for strcuture optimization
    saveModel(optimumTreeStructure, params.n_algo_structure, directory, trail)
    #plotPerfromance(gp_perfromance, params.n_algo_structure, directory, trail, params.n_data_target_names, params.n_problem_type)
    #plotTreeFitness(evaluateTree, optimumTreeStructure, directory, trail, str(params.n_algo_structure), 'train')
    #plotTreeFitness(evaluateTree, optimumTreeStructure, directory, trail, str(params.n_algo_structure), 'test')
    gp_train_error = optimalTreeFitness(evaluateTree, optimumTreeStructure, 'train')
    gp_test_error  = optimalTreeFitness(evaluateTree, optimumTreeStructure, 'test')
    #if params.n_problem_type == 'Classification':
    #    plotPrecisionVsRecall(gp_train_error, params.n_algo_structure, directory, trail, params.n_data_target_names, 'train')
    #    plotPrecisionVsRecall(gp_test_error, params.n_algo_structure, directory, trail, params.n_data_target_names, 'test')
    elpsade_time = elpsadeTime(start, end)
    # get features
    tree_feature_properties = optimumTreeStructure.getTreeInputFeatuereProperties(params.n_max_target_attr, params.n_data_input_names, params.n_data_target_names)
    # Collection and printing perfroance
    print('TIME: structure optimization: ', elpsade_time, 'sec')
    print('Structure optimization on training set error: ', gp_train_error)
    print('Structure optimization on test     set error: ', gp_test_error)
    # Collection in to a dictionary
    performance_dict.update({'gp iterations' : gp_perfromance})
    performance_dict.update({'gp best error' : [gp_train_error, gp_test_error]})
    performance_dict.update({'gp time (sec)' : elpsade_time})
    performance_dict.update({'tree features properties' : tree_feature_properties})
    
    #Save perfromance of GP
    performance_dict.update(getTraingParamsDict(params))
    filePath = os.path.join(directory, 'treePerfroamnce_dict_gp_'+ str(trail))
    np.save(filePath, performance_dict)
    # delete unnecessary variables of tsructuer optimization
    del tree_str_population, gp_train_error, gp_test_error, elpsade_time, start, end, tree_feature_properties


    # Go for parameter optimization
    if optimize_tree_parameter:
        print('\n Mandatory paramter optimization....')
        # recieve data for tree tsructure optimization Parameter optimization
        # MANDATORY Parameter optimization
        start = datetime.now()
        treeParameter = ParameterOptimization(evaluateTree, params, optimumTreeStructure)
        optimizedTree, param_perfromance =  treeParameter.optimize()
        end = datetime.now()
        # PLOTS and MODEL SAVING for parameter optimization
        saveModel(optimizedTree,  params.n_algo_param, directory, trail)
        plotPerfromance(param_perfromance, params.n_algo_param, directory, trail, params.n_data_target_names, params.n_problem_type)
        plotPerfromanceBoth(gp_perfromance, param_perfromance, params.n_algo_structure, params.n_algo_param, directory, trail, params.n_data_target_names, params.n_problem_type)
        #plotTreeFitness(evaluateTree, optimizedTree, directory, trail, str(params.n_algo_param), 'train')
        #plotTreeFitness(evaluateTree, optimizedTree, directory, trail, str(params.n_algo_param), 'test')
        if params.n_problem_type == 'Classification':
            train_error = optimalTreeFitness(evaluateTree, optimizedTree, 'train')
            test_error = optimalTreeFitness(evaluateTree, optimizedTree, 'test')
            plotPrecisionVsRecall(train_error, params.n_algo_param, directory, trail, params.n_data_target_names, 'train')
            plotPrecisionVsRecall(test_error, params.n_algo_param, directory, trail, params.n_data_target_names, 'test')
        # End of Mandatory parameter optimization
        # COLLECTION Paramters
        elpsade_time = elpsadeTime(start, end)
        performance_dict.update({'gd/mh iterations' : param_perfromance})
        performance_dict.update({'gd/mh time (sec)' : elpsade_time})
        print('TIME: mandatory paramter optimization:', elpsade_time, 'sec.')
        del param_perfromance, train_error, test_error, elpsade_time

        if params.n_validation_method == 'k_fold' or params.n_validation_method == 'five_x_two_fold':
            k = params.n_validation_folds
            # CROSS VALIDATION
            start = datetime.now()
            if params.n_validation_method == 'k_fold':
                data_input_values, data_target_values = data_partition(dataProcessing, params.n_validation_method)
                start = datetime.now()
                avg_train = []
                avg_test  = []
                if k == 2:
                    nFOLDS = 1
                else:
                    nFOLDS = k
                for fold in range(nFOLDS):
                    evaluateTree = data_evaluation(params, data_input_values, data_target_values, params.n_validation_method, fold)
                    # Parameter optimization
                    treeParameter = ParameterOptimization(evaluateTree, params, optimizedTree)
                    foldTree, _ =  treeParameter.optimize()
                    train_fold = optimalTreeFitness(evaluateTree, foldTree, 'train')
                    test_fold = optimalTreeFitness(evaluateTree, foldTree, 'test')
                    print('\nTrain on fold', fold, ': ', train_fold[0])
                    print('Test  on fold', fold, ': ', test_fold[0])
                    avg_train.append(train_fold)
                    avg_test.append(test_fold)
                print('\nAverange train on all folds: ', sum([row[0] for row in avg_train])/len(avg_train))
                print('Averange test  on all folds: ', sum([row[0] for row in avg_test])/len(avg_test))
    
            if params.n_validation_method == 'five_x_two_fold':
                data_input_values, data_target_values = data_partition(dataProcessing, params.n_validation_method)
                start = datetime.now()
                avg_train = []
                avg_test  = []
                for fold in range(5):
                    for cross in range(2):
                        evaluateTree = data_evaluation(params, data_input_values, data_target_values, params.n_validation_method, fold, cross)
                        # Parameter optimization
                        treeParameter = ParameterOptimization(evaluateTree, params, optimizedTree)
                        optimizedTree, param_perfromance =  treeParameter.optimize()
                        train_fold = optimalTreeFitness(evaluateTree, optimizedTree, 'train')
                        test_fold = optimalTreeFitness(evaluateTree, optimizedTree, 'test')
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
#%%
if __name__ == '__main__':
    starttime = time.time()
    for i in range(1):
        data_file_set = [#'pipe_flow_2D_surrey.csv']#,             #0
                         'iris']
                         # 'wisconsin',        #1
                         # 'wine',             #2
                         # 'heart.csv',        #3
                         # 'ionosphere.csv',   #4
                         # 'australian.csv',   #5
                         # 'pima.csv',         #6
                         # 'glass.csv',        #7
                         # 'vehicle.csv']      #8
                         #'yeast.csv',        #9
                         #'optdigits.csv']    #10 
                         #'adult.csv',
                         #'kddcup.csv',
                        
        # iterate over all data
        #data_file = data_file_set[4]
        for data_file in  data_file_set:
            print('Running dtata file', data_file)
            n_problem_type = 'Classification' # 'Regression' # 'Classification'
            param_opt = 'gd'
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
            print('PATH',path, '\nDirectory',directory)
            try:
                os.makedirs(directory)
            except OSError:
                print ("Directory already exisit or fails to create one")
            else:
                print ("Successfully created the directory %s" % directory)
        
            ## Training
            params = SettingParamter.setParameter(data_file, n_problem_type, param_opt, is_norm, normalize)
            optimizeSTR(params, directory, trail, isOptimizeParam)
        # end for
    #end for
    print('That took {} seconds'.format(time.time() - starttime))
#%%
#performance =  performance_dict['gd/mh iterations']
#class_names =  params.n_data_target_names
#is_logging = False
#if is_logging:
#    old_stdout = sys.stdout
#    filePath = 'model/treePerfromanceLog_'+ trail_time + '.log'
#    log_file = open(filePath,"w")
#    sys.stdout = log_file
#if is_logging:
#    sys.stdout = old_stdout
#    log_file.close()
    
#read_np = np.load('treePerfroamnce_dict_gp_iris_02_01_2020_12_48_31.npy').item()
    
