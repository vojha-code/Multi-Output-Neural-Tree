# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:54:40 2019

@author: yl918888
"""
from src.tree.evaluate_tree import EvaluateTree
import numpy as np

#% Setting Evaluation and Data Partition -------------------------------------------------------------
def data_partition(dataProcessing, data_partition = 'holdout'):
    print('DATA PARTITION: ', data_partition)
    if (data_partition == 'holdout'):
        return dataProcessing.holdout_method()
    
    if (data_partition == 'holdout_val'):
        return dataProcessing.holdout_method(True)
    
    if (data_partition == 'k_fold'):
        return dataProcessing.k_fold()

    if (data_partition == 'five_x_two_fold'):
        return dataProcessing.five_x_two_fold()


#% -----------------------------------------------------------------------------------------------------------------------
def data_evaluation(params, data_input_values, data_target_values, data_partition = 'holdout', test_fold = 0, cross = 0):
    k = params.n_validation_folds
    if (data_partition == 'holdout' or k == 2):
        data_train_input_values = data_input_values[0] # Training inputs
        data_test_input_values = data_input_values[1] # Test inputs
        data_val_input_values = None 
        data_train_target_values = data_target_values[0] # Training target
        data_test_target_values = data_target_values[1] # Test target
        data_val_target_values = None
        #del data_input_values, data_target_values 
    
    if (data_partition == 'holdout_val'):
        # preapre trainig data
        data_train_input_values = data_input_values[0] # Training inputs
        data_test_input_values = data_input_values[1] # Test inputs
        data_val_input_values = data_input_values[2] # Validation inputs
        data_train_target_values = data_target_values[0] # Training target
        data_test_target_values = data_target_values[1] # Test target
        data_val_target_values = data_target_values[2] # Validation target
    
    if (data_partition == 'k_fold' and k != 2):
        print('\nTraining on fold:', test_fold)
        # concatenate training sets
        fistNonTestSetAdded = 0
        for fold in range(k):
            if fold != test_fold:
                if fistNonTestSetAdded == 0:
                    fistNonTestSetAdded = 1
                    data_train_input_values = data_input_values[fold] # Training inputs set for the test fold
                    data_train_target_values = data_target_values[fold] # Training inputs set for the test fold
                    #print('First set', data_train_input_values.shape, data_train_target_values.shape)
                else:
                    data_train_input_values = np.concatenate((data_train_input_values, data_input_values[fold]), axis=0)
                    data_train_target_values = np.concatenate((data_train_target_values, data_target_values[fold]), axis=0)
                    #print('Concatenation', data_train_input_values.shape, data_train_target_values.shape)
                    
        data_test_input_values = data_input_values[test_fold] # Test inputs
        data_test_target_values = data_target_values[test_fold] # Test target
        
        data_val_input_values = None 
        data_val_target_values = None
        
    if (data_partition == 'five_x_two_fold'):
        print('\nTraining on fold: ', test_fold, cross)
        if cross == 0:
            #first set
            data_train_input_values = data_input_values[test_fold][0] # Training inputs set fold
            data_train_target_values = data_target_values[test_fold][0] # Training target fold
            #second set
            data_test_input_values = data_input_values[test_fold][1] # Test inputs set fold
            data_test_target_values = data_target_values[test_fold][1] # Test target fold
        else:
            #second set
            data_train_input_values = data_input_values[test_fold][1] # Training inputs set fold
            data_train_target_values = data_target_values[test_fold][1] # Training target fold
            #first set
            data_test_input_values = data_input_values[test_fold][0] # Test inputs set fold
            data_test_target_values = data_target_values[test_fold][0] # Test target fold 
            
        data_val_input_values = None         
        data_val_target_values = None
        
    print('Setting tree evalation paramters')
    n_target_attr_count = params.n_max_target_attr
    n_class_names = params.n_data_target_names
    n_target_denorm_params = [params.n_is_data_normlized]
    if params.n_is_data_normlized and params.n_problem_type != 'Classification':
        n_target_denorm_params.append(params.n_data_target_min) 
        n_target_denorm_params.append(params.n_data_target_max)
        n_target_denorm_params.append(params.n_data_norm_min)
        n_target_denorm_params.append(params.n_data_norm_max)

    #Training data evaluation
    evaluateTree = EvaluateTree(params.n_problem_type,
                                data_train_input_values,
                                data_test_input_values,
                                data_val_input_values,
                                data_train_target_values,
                                data_test_target_values,
                                data_val_target_values,
                                n_target_attr_count,
                                n_class_names,
                                n_target_denorm_params)
    return evaluateTree