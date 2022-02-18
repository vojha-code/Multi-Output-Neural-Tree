# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:21:44 2019

Setting Parameter

@author: yl918888

"""
import numpy as np
import argparse
from src.dataloader.data_processing import DataProcessing

class SettingParamter():
    '''
        Collecting paramter into aarg parse
    '''
    mParser = None
    
    def __init__(self):
        '''
            intiate a parser
        '''
        print('Gatterning paramters...')
        self.mParser = argparse.ArgumentParser() # parser accumulats all parametrs 
        
    def setData(self, data_file, data, p_problem_type = 'Classification', normalize = [0.0, 1.0], normlize_data = False):
        '''
            Setting data sets parametre
        '''
        self.mParser.add_argument('--n_dataset_name', default = data_file, type=str)
        self.mParser.add_argument('--n_problem_type', default = p_problem_type, type=str)
        print(' --- Preprocessing loded data --- ')
        #Creating/ Setting the imput data Matrix        
        # input values data structuer used is -> float64
        self.mParser.add_argument('--n_is_data_normlized', default = normlize_data, type=bool)
        if normlize_data:
            self.mParser.add_argument('--n_data_norm_min', default = normalize[0], type=float)
            self.mParser.add_argument('--n_data_norm_max', default = normalize[1], type=float)
            self.mParser.add_argument('--n_data_input_min', default = data.feature_min, type=float)
            self.mParser.add_argument('--n_data_input_max', default = data.feature_max, type=float)
            if data_file == "mnist.csv":
                data_input_values = data.data
                data_input_values = data_input_values.astype('float32')             
                data_input_values /= 255 # normalizing by deviding by heighst intensity
            else:
                data_input_values = DataProcessing.normalize_data(data.data, normalize[0], normalize[1])
        else:
            data_input_values = data.data               
        data_input_names = data.feature_names # input names data structuer is -> list 
        data_input_attrs = data_input_values.shape[1] # number of input attributes -> int retirve from columns number of input values
        
        
        #setting number of columns (attributes) in the dataset
        self.mParser.add_argument('--n_data_input_values', default = data_input_values, type=float)
        self.mParser.add_argument('--n_data_input_names',  default = data_input_names, type=str)
        self.mParser.add_argument('--n_max_input_attr', default = int(data_input_attrs), type=int)
        
        # preprocessing of the target columns - this should be further  automated 
        #fetching raw data target
        data_target_raw = data.target
        if(p_problem_type == 'Classification'):
            #set the taget names for classification problem
            data_traget_names = data.target_names
            self.mParser.add_argument('--n_data_target_names', default = data_traget_names, type=str)
            
            # Creating taget column
            #Source code:  https://scikit-learn.org/stable/modules/preprocessing_targets.html
            #print('    --- Class lavel binarization ---')
            from sklearn import preprocessing
            le = preprocessing.LabelBinarizer()
            le.fit(data_target_raw)
            #transform single column target to multicolimn target column 
            data_target_values = le.transform(data_target_raw)
            #print(data_target_values)
            if(data_target_values.shape[1] == 1):
                #print('Changed label')
                data_target_values = np.asarray([np.where(data.target == 0, 1, 0), np.where(data.target == 1, 1, 0)]).T

            data_traget_attrs = data_target_values.shape[1]    
            self.mParser.add_argument('--n_data_target_values', default = data_target_values, type=float)
            # set the number or of output columns equivalent to number of classes
            self.mParser.add_argument('--n_max_target_attr',  default = int(data_traget_attrs), type=int)
        else:
            if normlize_data:
                self.mParser.add_argument('--n_data_target_min', default = data.target_min, type=float)
                self.mParser.add_argument('--n_data_target_max', default = data.target_max, type=float)
                data_target_values = DataProcessing.normalize_data(data_target_raw, normalize[0], normalize[1])
            else:
                data_target_values = data_target_raw
                
            self.mParser.add_argument('--n_data_target_values', default = data_target_values, type=float)
            self.mParser.add_argument('--n_data_target_names', default = ['output'], type=str)
            self.mParser.add_argument('--n_max_target_attr', default = 1, type=int)  
        
        self.mParser.add_argument('--n_validation_method', default = 'holdout', type=str, choices = ['holdout','holdout_val','k_fold','five_x_two_fold'])  
        self.mParser.add_argument('--n_validation_folds', default = 10, type=int, choices = [2,5,10,'...'])  

    def setTreeParameters(self):
        '''
            Setting tree paramters
        '''
        print('Setting tree paramters...')
        self.mParser.add_argument('--n_max_children', default = 5, type=int, choices = [2,3,4,'...'], help='maximum number of child for a node')
        self.mParser.add_argument('--n_max_depth', default = 5, type=int, choices = [2,3,4,'...'], help='depth of the tree:')
        self.mParser.add_argument('--n_weight_range', default = [0, 1], type=float, choices = [[0,1], [-1,1]], help='tree edge weights and bias range')
        self.mParser.add_argument('--n_out_fun_type', default ='sigmoid', type=str, choices = ['Gaussian','tanh', 'sigmoid', 'ReLU', 'softmax'], help='can be same as internam function nodes')
        self.mParser.add_argument('--n_fun_type', default =  'sigmoid', type=str, choices = ['Gaussian','tanh', 'sigmoid', 'ReLU', 'softmax'])
        self.mParser.add_argument('--n_fun_range', default = [0, 1], type=float, help='used for Guassin function only at the moment, center and width range')
        self.mParser.add_argument('--n_int_leaf_rate', default = 0.5, type=float, help='leaf genration rate for interna nodes')
        

    def setStructureOptimization(self):
        '''
            Set structuer optimization algorithms
        '''
        print('Setting structure optimization paramters...')
        self.mParser.add_argument('--n_max_itrations', default = 10, type=int)
        self.mParser.add_argument('--n_max_population', default = 20, type=int)
        self.mParser.add_argument('--n_algo_structure', default = 'gp', type=str, choices = ['gp', 'nsgp_2', 'nsgp_3']) # 
        self.mParser.add_argument('--n_optimization', default = 'MIN', type=str, choices = ['MIN', 'MAX'], help='minimization porblem -> MIN and otherwise')
        self.mParser.add_argument('--n_max_objectives', default = 2, type=int, help='max objective in the problem domain or in the tree itself, eg 2 for tree imples tree erro and tree size')
        self.mParser.add_argument('--n_division', default = 10, type=int, help='number p > 0 of reference points for an objective function in NSGA-III algorithm')
        self.mParser.add_argument('--n_prob_crossover', default = 0.8, type=float)
        self.mParser.add_argument('--n_prob_mutation', default = 0.2, type=float, help='mutation probability is 1 - crossover probability')

    def setParamterOptimization(self, optimizar):
        '''
            Set algorithms and parameter optimization pareamters
        '''
        print('Setting weights optimization paramters...')
        self.mParser.add_argument('--n_param_optimizer', default = optimizar, choices = ['gd','mh'], type=str)        
        self.mParser.add_argument('--n_param_opt_max_itr', default = 50, type=int)
        if optimizar == 'gd':
            self.mParser.add_argument('--n_algo_param', default = 'adam',type=str, choices = ['gd','momentum_gd','nesterov_accelerated_gd','adagrad','rmsprop','adam'])
            self.mParser.add_argument('--n_gd_eval_mode', default = 'stochastic',type=str, choices = ['batch','stochastic'])
            self.mParser.add_argument('--n_gd_precision', default = 0.00001, type=float, help='termonation tolarane')
            self.mParser.add_argument('--n_gd_eta', default = 0.1, type=float, help='gradient descent learning rate')
            self.mParser.add_argument('--n_gd_gamma', default = 0.9, type=float, help='gradient descent momentum rate')
            self.mParser.add_argument('--n_gd_eps', default = 1e-8, type=float, help='gradient epsilon')
            self.mParser.add_argument('--n_gd_beta', default = 0.9, type=float, help='gradient beta')
            self.mParser.add_argument('--n_gd_beta1', default = 0.9, type=float, help='gradient beta')
            self.mParser.add_argument('--n_gd_beta2', default = 0.9, type=float, help='gradient beta')
        else:
            self.mParser.add_argument('--n_algo_param', default = 'mcs',type=str, choices = ['mcs','de','sa','bsde', 'pso', 'abc'])
            self.mParser.add_argument('--n_mh_pop_size', default = 20, type=int)
            #self.mParser.add_argument('--n_mh_algo_de_mut', default =  0.8, type=float, help= 'betweeen 0 and 1')
            #self.mParser.add_argument('--n_mh_algo_de_crossp', default = 0.7, type=float, help= 'betweeen 0 and 1')
            #self.mParser.add_argument('--n_mh_algo_de_strategy', default = 'rand_to_best_1bin', choices = ['rand_to_best_1bin','rand_to_best_2bin'])
                
    def checkParameters(self, params):
        print('Chaking paramters:')
        
        print('Check data: ')
        print('  PARAM: problem type          : ', params.n_problem_type)
        print('  PARAM: exs in instance sapce : ', params.n_data_input_values.shape[0])
        print('  PARAM: max input attributes  : ', params.n_max_input_attr)
        print('  PARAM: max input attributes  : ', params.n_data_input_names)
        print('  PARAM: max output attributes : ', params.n_max_target_attr)
        print('  PARAM: max output attributes : ', params.n_data_target_names)
        print('  PARAM: max output attributes : ', params.n_validation_method)
        print('  PARAM: data normalization    : ', params.n_is_data_normlized)
        if params.n_is_data_normlized:
            print('  PARAM: norm min              : ', params.n_data_norm_min)    
            print('  PARAM: norm max              : ', params.n_data_norm_max)    
            print('  PARAM: data inputs min       : ', params.n_data_input_min)    
            print('  PARAM: data inputs max       : ', params.n_data_input_max)
            if params.n_problem_type != 'Classification':
                print('  PARAM: data target min       : ', params.n_data_target_min)    
                print('  PARAM: data target max       : ', params.n_data_target_max)
        
        
        print('\nCheck Tree paramters:')
        print('  PARAM: max node children     : ', params.n_max_children)
        print('  PARAM: max tree depth        : ', params.n_max_depth)
        print('  PARAM: edge weight range     : ', params.n_weight_range)
        print('  PARAM: node function type    : ', params.n_fun_type)
        print('  PARAM: func value range      : ', params.n_fun_range)
        
        print('\nCheck Structur oprimization paramters:')
        print('  PARAM: algo for structure opt: ', params.n_algo_structure)
        print('  PARAM: optimization type     : ', params.n_optimization)
        print('  PARAM: max population        : ', params.n_max_population)
        print('  PARAM: max iterations        : ', params.n_max_itrations)
        print('  PARAM: max objectives        : ', params.n_max_objectives)
        print('  PARAM: division of obj       : ', params.n_division)
        print('  PARAM: crossover probability : ', params.n_prob_crossover)
        print('  PARAM: mutation probability  : ', params.n_prob_mutation)
        
        
        print('\nCheck Parameter oprimization paramters:')
        print('  PARAM: param evaluation itr  : ', params.n_param_opt_max_itr)
        print('  PARAM: gd algorithm          : ', params.n_algo_param)
        if params.n_param_optimizer == 'gd':
            print('  PARAM: gd evaluation mode    : ', params.n_gd_eval_mode)
            print('  PARAM: gd precision          : ', params.n_gd_precision)
            print('  PARAM: gd learning rate      : ', params.n_gd_eta)
            print('  PARAM: gd momentum rate      : ', params.n_gd_gamma)
        else:
            print('  PARAM: mh population         : ', params.n_mh_pop_size)

    def setParameter(data_file = 'input_a_file.csv', n_problem_type = 'Classification', param_opt = 'gd', is_norm = False, normalize = [0.0, 1.0]):   
        print('Setiing problem...') 
        dataProcessing = DataProcessing()
        data = dataProcessing.load_data(n_problem_type, data_file)
        #%
        setParams = SettingParamter()
        setParams.setData(data_file, data, n_problem_type, normalize, is_norm)
        setParams.setTreeParameters()
        setParams.setStructureOptimization()
        setParams.setParamterOptimization(param_opt)
        
        #Retriving paramters
        params = setParams.mParser.parse_args()
        #setParams.checkParameters(params)#  checking paramters
        return params