import numpy as np
import pandas as pd
import os
from os import path


class Data:
    data = None
    feature_names = None
    feature_min = None
    feature_max = None
    
    target = None
    target_min = None
    target_max = None    
    target_names = None
        
    
    def __init__(self,pdata, pfeature_names, ptarget, ptarget_names = None):
        '''
            Setting data objects
        '''
        self.data = pdata
        self.feature_min = pdata.min(axis=0)
        self.feature_max = pdata.max(axis=0)
        self.feature_names = pfeature_names
        
        self.target = ptarget
        self.target_min = ptarget.min(axis=0)
        self.target_max = ptarget.max(axis=0)
        self.target_names = ptarget_names

class DataProcessing:
    '''
        Process data for traning
    '''
    m_params =  None # set paramters
    
    #------------------------------------------------------------------------------------------
    #def __init__(self):

       
    #------------------------------------------------------------------------------------------
    def load_data(self, p_problem_type, dataset = 'input_a_file.csv'):
        '''
            Loading data
        '''
        fromFile = os.path.join(os.getcwd(), 'data'+os.sep+dataset)
        print('DATA: file path', fromFile)
        #fromFile = os.path.join("data","mnist.csv")
        if path.exists(fromFile):
            print('DATA:loading from file ', fromFile)
            
            dataPD = pd.read_csv(fromFile)
            #collecting input and input names
            data = dataPD.loc[:, dataPD.columns != 'target']
            feature_names = data.columns.tolist()
            data = data.values
            # Collecting target                                
            target = dataPD['target']
            target_names = None
            if p_problem_type == 'Classification':
                # converting string to int
                target_names = np.unique(np.asarray(target)).tolist()
                for x in target_names:
                    target = target.replace(x, target_names.index(x))
            target = np.asanyarray(target)
            return Data(data, feature_names, target, target_names)
        else:
            if p_problem_type == 'Classification':
                print('DATA: Classification - loading dataset ', dataset ,' from sklearn library...')
                if dataset.lower() == 'Iris'.lower():
                    from sklearn.datasets import  load_iris
                    data = load_iris()
                    
                if dataset.lower() == 'Wine'.lower():
                    from sklearn.datasets import load_wine
                    data = load_wine()
                if dataset.lower() == 'Wisconsin'.lower():
                    from sklearn.datasets import load_breast_cancer
                    data = load_breast_cancer()
                #if dataset.lower() == 'MINIS'.lower():
                #    from sklearn import datasets
                #    digits = datasets.load_digits()
                return Data(data.data, data.feature_names, data.target, data.target_names)
            else:
                print('DATA: Classification - loading dataset ', dataset ,' from sklearn library...')
                if dataset.lower() == 'Diabetes'.lower():
                    from sklearn.datasets import  load_diabetes
                    data = load_diabetes()                
                if dataset.lower() == 'Boston'.lower():
                    from sklearn.datasets import  load_boston
                    data = load_boston()
                return Data(data.data, data.feature_names, data.target)
    #------------------------------------------------------------------------------------------
    def setParams(self, p_params):
        '''
            Setting paramters
        '''
        self.m_params = p_params

    #Normalizing data value
    #-----------------------------------------------------------------------------------------
    def normalize_data(X, nmin=0.0, nmax=1.0):
        '''
        Normalizing (Scaling) each featuer between nmin and nmax
        args:
            param: X is a numpy matrix matrix
            param: nmin lowerbound for normalization
            param: nmax upper bound for normalization
        '''
        if type(X) == list:
            X = np.ndarray(X)
        Xmin = X.min(axis=0)
        Xmax = X.max(axis=0)
        
        X_std = (X - Xmin) / (Xmax - Xmin)
        X_scaled = X_std * (nmax - nmin) + nmin
        
        return X_scaled
    
    #De-normalizing data value
    #-----------------------------------------------------------------------------------------
    def denormalize_data(X_norm, Xmin, Xmax, nmin=0.0, nmax=1.0):
        '''
        De-normalizing (De-scaling) each featuer
        args:
            param: X is a numpy matrix matrix
            param: Xmin original data Min vector used for normalization
            param: Xmax original data MAx vector used for normalization
            param: nmin original lowerbound used for normalization
            param: nmax original upper bound used for normalization
        '''
        return ((Xmin - Xmax) * X_norm - nmax * Xmin + Xmax * nmin) / (nmin - nmax)

    #------------------------------------------------------------------------------------------
    def holdout_method(self, val_set = False, training_set = 0.8):
        '''
            Spliting data
            paramL param - set of parameters
        '''
        print('Shuffling and Spliting data into training, validation, and test sets:')
        #Fetching data from the param
        data_input_values = self.m_params.n_data_input_values
        data_target_values = self.m_params.n_data_target_values
        #prearing training, validation, and test sets
        # Randomizing the data samples
        # making a deep copy so the keep object of original file safe as np.random.shuffle(arr) modifies original object
        # and equating a = b means object remais the same but assigned to two variables
        random_sequence = np.arange(len(data_input_values))
        if self.m_params.n_dataset_name == "mnist.csv":
            # Do not randomize the data
            training_set = 0.8571428571428571
            data_input_values_rand = data_input_values
            data_target_values_rand = data_target_values
        else:
            np.random.shuffle(random_sequence)
            #fetching data from array 
            data_input_values_rand = np.zeros(data_input_values.shape)
            data_target_values_rand = np.zeros(data_target_values.shape)
            for index in range(len(data_input_values)):
                data_input_values_rand[index] = data_input_values[random_sequence[index]]
                data_target_values_rand[index] = data_target_values[random_sequence[index]]
            
        # Spliting into training test and validation sets
        if val_set:
            data_inputs = np.split(data_input_values_rand, [int(.6 * len(data_input_values_rand)), int(.8 * len(data_input_values_rand))]) 
            data_targets = np.split(data_target_values_rand, [int(.6 * len(data_target_values_rand)), int(.8 * len(data_target_values_rand))]) 
        else:
            data_inputs = np.split(data_input_values_rand, [int(training_set * len(data_input_values_rand))]) 
            data_targets = np.split(data_target_values_rand, [int(training_set * len(data_target_values_rand))])
            
        
        #Cheking the percent of the split
        print('    training set   :', len(data_inputs[0]),' -> ', len(data_inputs[0])*100.00/ len(data_input_values), '%')
        print('    test set       :', len(data_inputs[1]),' -> ',len(data_inputs[1])*100.00/ len(data_input_values), '%')
        if val_set:
            print('    validation set :', len(data_input_values[2])*100.00/ len(data_input_values), '%')
        
        #returning data split        
        return data_inputs, data_targets, random_sequence


    #------------------------------------------------------------------------------------------
    def k_fold(self):
        '''
            Spliting data in 10 fold
            paramL param - set of parameters
        '''
        print('Shuffling and Spliting data into training, validation, and test sets:')
        #Fetching data from the param
        data_input_values = self.m_params.n_data_input_values
        data_target_values = self.m_params.n_data_target_values
        #prearing training, validation, and test sets
        # Randomizing the data samples
        # making a deep copy so the keep object of original file safe as np.random.shuffle(arr) modifies original object
        # and equating a = b means object remais the same but assigned to two variables
        random_sequence = np.arange(len(data_input_values))
        np.random.shuffle(random_sequence)
        #fetching data from array 
        data_input_values_rand = np.zeros(data_input_values.shape)
        data_target_values_rand = np.zeros(data_target_values.shape)
        for index in range(len(data_input_values)):
            data_input_values_rand[index] = data_input_values[random_sequence[index]]
            data_target_values_rand[index] = data_target_values[random_sequence[index]]
            
        # Spliting into k fold traiing and test
        k = self.m_params.n_validation_folds
        if (k == 2):
            return self.holdout_method(False, training_set = 0.5)
        else:
            length = int(len(data_input_values_rand)/k) #length of each fold
            print(' Each fold length           :',length, 'for data length',len(data_input_values_rand))
            data_inputs_folds = []
            data_target_folds = []
            for i in range(k-1):
                data_inputs_folds += [data_input_values_rand[i*length:(i+1)*length]]
                data_target_folds += [data_target_values_rand[i*length:(i+1)*length]]
            data_inputs_folds += [data_input_values_rand[(k-1)*length:len(data_input_values_rand)]]
            data_target_folds += [data_target_values_rand[(k-1)*length:len(data_target_values_rand)]]
            
            # ONLY works for even data set
            #data_inputs_folds = np.split(data_input_values_rand, 10) 
            #data_target_folds = np.split(data_target_values_rand, 10)
            
            #Cheking the percent of the split
            print('    each fold set size (input, target) : (', len(data_inputs_folds[0])*100.00/ len(data_input_values), '%, ', 
                                                                len(data_target_folds[0])*100.00/ len(data_target_values), '% )')
            #print('    each test     fold set size :', )
            
            #returning data split
            return data_inputs_folds, data_target_folds, random_sequence

    #------------------------------------------------------------------------------------------
    def five_x_two_fold(self):
        '''
            Spliting data in 10 fold
            paramL param - set of parameters
        '''
        print('Shuffling and Spliting data into training, validation, and test sets:')
        #Fetching data from the param
        data_input_values = self.m_params.n_data_input_values
        data_target_values = self.m_params.n_data_target_values
        #prearing training, validation, and test sets
        # Randomizing the data samples
        # making a deep copy so the keep object of original file safe as np.random.shuffle(arr) modifies original object
        # and equating a = b means object remais the same but assigned to two variables
        # Spliting into 5 fold traiing and test each with 50%
        length = int(len(data_input_values)/2) #length of each fold
        print(' Each fold length           :',length, 'for data length',len(data_input_values))
        data_inputs_folds, data_target_folds = [], []
        for i in range(5):
            random_sequence = np.arange(len(data_input_values))
            np.random.shuffle(random_sequence)
            #fetching data from array 
            data_input_values_rand = np.zeros(data_input_values.shape)
            data_target_values_rand = np.zeros(data_target_values.shape)
            for index in range(len(data_input_values)):
                data_input_values_rand[index] = data_input_values[random_sequence[index]]
                data_target_values_rand[index] = data_target_values[random_sequence[index]]
                
            data_inputs = []
            data_target = []
            # Training
            data_inputs += [data_input_values_rand[0:length]]
            data_target += [data_target_values_rand[0:length]]
            # Test
            data_inputs += [data_input_values_rand[length:len(data_input_values_rand)]]
            data_target += [data_target_values_rand[length:len(data_target_values_rand)]]
            
            data_inputs_folds.append(data_inputs)
            data_target_folds.append(data_target)
            
            #data_inputs_folds.append(np.split(data_input_values_rand, 2))
            #data_target_folds.append(np.split(data_target_values_rand, 2))
            
        #Cheking the percent of the split
        print('    each fold set size (input, target) :', len(data_inputs_folds[0][0])*100.00/ len(data_input_values), '%', 
                                                          len(data_target_folds[0][1])*100.00/ len(data_target_values), '%')
        #returning data split
        return data_inputs_folds, data_target_folds, np.arange(1)
    
##    
#a = np.arange(442)
###np.random.shuffle(a)
#length = int(len(a)/2) #length of each fold
#folds = []
###for i in range(9):
###    folds += [a[i*length:(i+1)*length]]
###folds += [a[9*length:len(a)]]
#
#
#folds += [a[0:length]]
#folds += [a[length:len(a)]]
#b = np.split(a, 10) 
###
###c = []
###a, b = [], []
###
###c.append(b)
###c[0]


