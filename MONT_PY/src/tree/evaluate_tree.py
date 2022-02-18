'''
Created on Sat Apr 20 12:42:42 2019

Training steps of neural tree
Author: v ojha
Affiliation: Uni of Reading

'''
import numpy as np
import multiprocessing as mp

from src.dataloader.data_processing import DataProcessing
from src.reporting.perfromance_metrices import Perfromance

class EvaluateTree():
    '''
        Evaluate a given tree for a given datasets
        This clas takes processed:
            training, validation and test datasets in its constractor
            Its method takes tree object to evaluate
    '''    
    m_problem_type = None # check is the problem is classification
    
    # Input examles in training validation and test sets
    m_data_train_inputs = None
    m_data_test_inputs = None
    m_data_val_inputs = None
    
    # coresponding target examles in training validation and test sets
    m_data_train_targets  = None 
    m_data_test_targets = None
    m_data_val_targets  = None
    
    m_target_attr_count = None # information on how many calss clasumn will be
    m_traget_class_names = None # name sof the tragtre class
    
    
    # This is temporary object holders:
    m_data_input_TO_tree = None # for the input set to be used by the tree to get fitness
    m_data_target_TO_tree = None # for the traget set to be used by the tree to get fitness
    m_data_prediction_OF_tree = None # for the prediction set to be used by the tree to get fitness
    
    m_data_denorm_params = []
    
    
    def __init__(self, p_problem_type, 
                 p_data_train_inputs, 
                 p_data_test_inputs, 
                 p_data_val_inputs, 
                 p_data_train_targets, 
                 p_data_test_targets, 
                 p_data_val_targets,
                 p_target_attr_count, 
                 p_target_class_names,
                 p_data_denorm_params = []):
        '''
        Set tree and dataset infromation
        args:
            params: p_tree, p_data_input_TO_tree, p_target_attr_count, p_problem_type
        '''
        self.m_problem_type = p_problem_type
        
        self.m_data_train_inputs = p_data_train_inputs
        self.m_data_test_inputs = p_data_test_inputs    
        self.m_data_val_inputs = p_data_val_inputs
            
        self.m_data_train_targets  = p_data_train_targets 
        self.m_data_test_targets = p_data_test_targets
        self.m_data_val_targets  = p_data_val_targets

        self.m_target_attr_count = p_target_attr_count
        self.m_traget_class_names = p_target_class_names
        self.m_data_denorm_params = p_data_denorm_params 
        #                       [params.n_is_data_normlized, 
        #                        params.n_data_target_min, params.n_data_target_max,
        #                        params.n_data_norm_min, params.n_data_norm_max,]
    
    def getPartitionSeprated(self):
        return self.m_data_train_inputs, self.m_data_test_inputs, self.m_data_train_targets, self.m_data_test_targets
    #-------------------------------------------------------------------------------------------------------------        
    def set_dataset_to_evaluate(self, p_dataset_to_use):
        '''
        This method set type of set to use in the tree used 
        args:
            param:    p_dataset_to_use    string value
        '''
        # set input and traget as training set sets
        if(p_dataset_to_use == 'train'):
            #print('Training data setting...')
            # reset the variables
            self.m_data_input_TO_tree =  None
            self.m_data_target_TO_tree = None            
            self.m_data_input_TO_tree =  self.m_data_train_inputs
            self.m_data_target_TO_tree = self.m_data_train_targets
        # set input and traget as test sets
        if(p_dataset_to_use == 'test'):
            #print('Testing data setting...')
            # reset the variables
            self.m_data_input_TO_tree =  None
            self.m_data_target_TO_tree = None
            self.m_data_input_TO_tree =  self.m_data_test_inputs
            self.m_data_target_TO_tree = self.m_data_test_targets
        # set input and traget as validation set sets
        if(p_dataset_to_use == 'val'):
            #print('Validating data setting...')
            # reset the variables
            self.m_data_input_TO_tree =  None
            self.m_data_target_TO_tree = None            
            self.m_data_input_TO_tree =  self.m_data_val_inputs
            self.m_data_target_TO_tree = self.m_data_val_targets
        
    #-------------------------------------------------------------------------------------------------------------    
    def getTreePredictedOutputs(self, p_treeToEvalute):
        '''
            Evaluate tree fitness for a input datasets return the predicted output
            args:
                param: p_tree takes a tree object to evaluate its fitenss (predicted outputs)
                
            return:    a(an) matrix/array of tree prediction  rows x column
        '''
        multiprocessing_eval = False # -> No multiprocessing since small computaiion is not efficent in MP
        n_list_data_prediction_OF_tree = []
        if not multiprocessing_eval:
            for v_input_vector in self.m_data_input_TO_tree:
                n_list_data_prediction_OF_tree.append(p_treeToEvalute.getOutput(v_input_vector, self.m_target_attr_count))
        else:# Multiprocessing not efficent for small computation
            # test revealse it is not efficient
            jobs = []
            pipe_list = []
            #manager = mp.Manager()
            #return_list = manager.list()
            for v_input_vector in self.m_data_input_TO_tree:
                recv_end, send_end = mp.Pipe(False)
                p = mp.Process(target=getTreeOutOnSingleVector, args=(p_treeToEvalute, v_input_vector, self.m_target_attr_count, send_end))
                #p = mp.Process(target=getTreeOutOnSingleVector, args=(p_treeToEvalute, v_input_vector, self.m_target_attr_count, return_list))
                jobs.append(p)
                pipe_list.append(recv_end)
                p.start()
        
            for proc in jobs:
                proc.join()
            #recieves computed results
            n_list_data_prediction_OF_tree = [x.recv() for x in pipe_list]
            #n_list_data_prediction_OF_tree = [x for x in return_list]
            
            #print(return_list)
            #print(len(self.m_data_input_TO_tree), len(return_list))
            #for i in range(len(self.m_data_input_TO_tree)):            
            #    n_list_data_prediction_OF_tree.append(return_list[i])

        #converting list to an array / matrix
        self.m_data_prediction_OF_tree = np.array(n_list_data_prediction_OF_tree)
        # return the prediction - this may or many not be used by the a function
        return self.m_data_prediction_OF_tree
    
    #-------------------------------------------------------------------------------------------------------------
    def getTreeFitness(self):
        '''
            Evaluate fitnees score of the tree
            args:
                param:    p_prediction_OF_tree    takes an arry/matrix of predicted out of the tree
                
            return:    (float) - score of the tree
        '''
        return self.compareTrueAndPred(self.m_data_target_TO_tree, self.m_data_prediction_OF_tree)
    
    
    #-------------------------------------------------------------------------------------------------------------
    def compareTrueAndPred(self, p_true_target, p_tree_predictions):
        '''
            Compae the the actual and prediction and report back the accuract/mean_sqaure_error
            args:
                param:    p_true_target            (float) mx n true traget 
                param:    p_tree_predictions       (float) mx n predicted traget 
                
            return: (float) error/fitness in terms of error rate or mease sqaured error of the tree -> 0 best
        '''
        performance = Perfromance(self.m_problem_type, p_true_target, p_tree_predictions, self.m_traget_class_names)
        return performance.measures()
    
    def plot(self, directory, trail, norm_or_no_corr_line = False):
        '''
         Plot metric for classification and regression repending upon 
        '''
        if self.m_problem_type == 'Classification':
            performance = Perfromance(self.m_problem_type, self.m_data_target_TO_tree, self.m_data_prediction_OF_tree, self.m_traget_class_names)
            performance.getConfusionMatrixPlot(directory, trail, norm_or_no_corr_line)
        else:
            p_true_target       = self.m_data_target_TO_tree
            p_tree_predictions  = self.m_data_prediction_OF_tree
            if self.m_data_denorm_params[0]:
                #print('PRINT NORM', np.min(p_true_target), np.max(p_true_target), np.min(p_tree_predictions), np.max(p_tree_predictions), )
                p_true_target = DataProcessing.denormalize_data(p_true_target, 
                                                                self.m_data_denorm_params[1], self.m_data_denorm_params[2],
                                                                self.m_data_denorm_params[3], self.m_data_denorm_params[4])
                p_tree_predictions = DataProcessing.denormalize_data(p_tree_predictions, 
                                                                self.m_data_denorm_params[1], self.m_data_denorm_params[2],
                                                                self.m_data_denorm_params[3], self.m_data_denorm_params[4])
                #print('PRINT DENORM', np.min(p_true_target), np.max(p_true_target), np.min(p_tree_predictions), np.max(p_tree_predictions), )
            
            performance = Perfromance(self.m_problem_type, p_true_target, p_tree_predictions, self.m_traget_class_names)
            performance.getScatterPlot(directory, trail, norm_or_no_corr_line)
    
#-------------------------------------------------------------------------------------------------------------                        
def getTreeOutOnSingleVector(p_treeToEvalute, v_input_vector, p_target_attr_count, send_end):
    #return_list.append(p_treeToEvalute.getOutput(v_input_vector, p_target_attr_count)) 
    send_end.send(p_treeToEvalute.getOutput(v_input_vector, p_target_attr_count)) 


