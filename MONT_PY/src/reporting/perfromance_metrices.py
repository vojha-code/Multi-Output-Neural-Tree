'''
Created on Sat Apr 20 12:42:42 2019

Training steps of neural tree
Author: v ojha
Affiliation: Uni of Reading

'''

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score
from matplotlib.ticker import FormatStrFormatter
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss

class Perfromance:
    m_y_type = None # type of problem
    m_y_true = None # actual traget
    m_y_pred = None # predicted target
    m_y_names = None # type of problem
    
    
    def __init__(self, p_prob_type, p_y_true, p_y_pred, p_class_names):
        '''
            Set the tru and predicted values
            Args:
                param: p_prob_type claisiffication or  regression
                param: p_y_true an array
                param: p_y_pred an array
                param: p_y_names an array of class col names
        '''
        self.m_y_type = p_prob_type #  Classifcation/ Regression
        
        self.m_y_true = p_y_true #  actual target values
        self.m_y_pred = p_y_pred #  tree predicted target values
        
        self.m_y_names = p_class_names #classs names or in other words target columns names
        

    #-------------------------------------------------------------------------------------------------------------    
    def measures(self):
        '''
            Compute error of the tree for y_ture and y_pred
            
            Return: (float) accuracy/meas squared error of the tree
        '''
        performance = []
        if(self.m_y_type == 'Classification'):
            #If problem is classifcation
            # return error rate
            
            
            y_true, y_pred = self.classOuputProcessing(True)
            error = 1.0 - accuracy_score(y_true, y_pred)
            performance.append(error)
            performance = performance + self.get_FP_FN_TP_TN(confusion_matrix(y_true, y_pred))
            #performance.append(log_loss(self.m_y_true, self.m_y_pred))
            #performance = performance + list(precision_recall_fscore_support(y_true, y_pred, average='macro'))
            #return 1.0 - accuracy_score(y_true, y_pred)
        else:
            #retrun mse
            performance.append(mean_squared_error(self.m_y_true, self.m_y_pred))
            performance.append(r2_score(self.m_y_true, self.m_y_pred))
            #return mean_squared_error(self.m_y_true, self.m_y_pred)
        return performance

    def get_FP_FN_TP_TN(self, cm):
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        ## Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        
        ## Specificity or true negative rate
        TNR = TN/(TN+FP) 
        
        ## Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        #NPV = TN/(TN+FN)
        # Fall out or false positive rate
        #FPR = FP/(FP+TN)
        ## False negative rate
        #FNR = FN/(TP+FN)
        ## False discovery rate
        #FDR = FP/(TP+FP)
        
        # Overall accuracy
        #ACC = (TP+TN)/(TP+FP+FN+TN)
        #print('Precision',PPV)
        #print('Recall/Sensitivity',TPR)
        #print('Specificity',TNR)
        
        #precision, recall, Specificity
        return [PPV, TPR, ]
    #-------------------------------------------------------------------------------------------------------------    
    # Methods for processing classification outputs and its perfromance measuers metics
    #-------------------------------------------------------------------------------------------------------------    
    def accuracy(self):
        '''
            Return accuracy of the tree
        '''
        return accuracy_score(self.m_y_true, self.m_y_pred)

    #-------------------------------------------------------------------------------------------------------------
    def classOuputProcessing(self, AssignLebels = True):
        '''
            Method for printing calssification confision metric and accuracy
            variables used:
                param:    p_class_names         (string) class name 
                param:    p_data_true_traget    (numeric) binarized class multi column calassss lebels
                param:    p_data_pred_traget    (float) class multi column callss lebels
                
                Optional:
                param:    AssignLebels          (boolean) "True" for NOMINAL return and "False" for NUMERIC
            
            return: NOMINAL/NUMERIC class array
        '''
        #Computeing single level for the binary multilevel data   
        if(AssignLebels):
            #print('Test Assign Labels')
            y_true = self.assignClassLebel(self.m_y_names, self.multilevelTOsinglelevel(self.m_y_true))
            y_pred = self.assignClassLebel(self.m_y_names, self.multilevelTOsinglelevel(self.classPredictionBinarization(self.m_y_pred)))
        else:
            y_true = self.multilevelTOsinglelevel(self.m_y_true)
            y_pred = self.multilevelTOsinglelevel(self.classPredictionBinarization(self.m_y_pred))
        return y_true, y_pred

    #-------------------------------------------------------------------------------------------------------------    
    def classPredictionBinarization(self, p_mxn_array_float):
        '''
            Convert real-value (float) matrix of row x col (class) matrix into binary calss data
            args:
                param:    p_mxn_array_float    prediction data from tree output
            
            return:    n_mxn_array_binary (m x n)    p_mxn_array_binary.shape[0] x p_mxn_array_binary.shape[1] binary data
        '''
        if len(list(p_mxn_array_float.shape)) == 1:
            p_mxn_array_float = np.asmatrix(p_mxn_array_float).T
            
        if p_mxn_array_float.shape[1] == 1:
            n_mxn_array_binary = np.where(p_mxn_array_float > 0.499, 1, 0)
        else:
            n_mxn_array_binary = np.zeros_like(p_mxn_array_float)
            n_mxn_array_binary[np.arange(len(p_mxn_array_float)), p_mxn_array_float.argmax(1)] = 1
        return n_mxn_array_binary

    #-------------------------------------------------------------------------------------------------------------    
    def multilevelTOsinglelevel(self, p_mxn_array_binary):
        '''
            Transform multilevel to sinbgle level
            args:
                param:      p_mxn_array_binary        m x n binarized class multi column callss lebels
            
            return: numeric array - number single class labels
        '''
        if p_mxn_array_binary.shape[1] == 1:
            return p_mxn_array_binary.flatten()
        else:
            # fetch number of columns into m x n p_mxn_array_binary data i.e., value of n 
            # and create an array of column index
            col_index = np.arange(p_mxn_array_binary.shape[1]) 
            # reapate col index for the length (m times) of p_mxn_array_binary data
            col_index_matrix = np.array([col_index]*len(p_mxn_array_binary))
            # Multiply to insert right column (numeric class value/lebel) at right place in thge matrix 
            col_index_matrix = np.multiply(p_mxn_array_binary, col_index_matrix)
            return np.sum(col_index_matrix, axis = 1) # sum will generate numeric single class labels (index start from zero)
    
    #-------------------------------------------------------------------------------------------------------------
    def assignClassLebel(self, p_class_names, p_array):
        '''
            Assign class lebels to the class values
            args:
                param:      p_class_names        original paramter settings
                param:      p_array       single collumn numeric level of classes
            
            return: labled data with nominal (string attributes) 
        '''
        return [str(p_class_names[int(x)]) for x in p_array.tolist()]

    
    #-------------------------------------------------------------------------------------------------------------
    def plot_confusion_matrix(self, y_true, y_pred, directory, trail, normalize = False,
                              title = None,
                              cmap=plt.cm.Blues):
        
        """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
            
            return plot object
        """
        classes = self.m_y_names
        
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
    
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               #title=title,
               ylabel='True label',
               xlabel='Predicted label')
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        #plt.show()
        # Save figure
        graph_filepath = os.path.join(directory, 'treeCM_Plot_' + trail +'.pdf')
        plt.savefig(graph_filepath, dpi=300, format='pdf', bbox_inches='tight')        
        return ax
    
   #-------------------------------------------------------------------------------------------------------------
    def getConfusionMatrixPlot(self, directory, trail, p_normalize = False):
        '''
        Print confusion matrix 
        args:
            param:    y_true    an array of nominal true class values
            param:    y_pred    an array of nominal predicted class values
            param:    p_normalize boolean
        '''
        y_true, y_pred = self.classOuputProcessing(True)
        self.plot_confusion_matrix(y_true, y_pred, directory, trail, p_normalize)

    def getScatterPlot(self, directory, trail, no_line = False):
        '''
        Print confusion matrix 
        args:
            param:    y_true    an array of nominal true class values
            param:    y_pred    an array of nominal predicted class values
            param:    p_normalize boolean
        '''
        self.plot_scatter(self.m_y_true, self.m_y_pred, directory, trail, no_line)

    
    #-------------------------------------------------------------------------------------------------------------
    def plot_scatter(self, x, y, directory, trail, without_correlation_line = False):
        '''
        http://stackoverflow.com/a/34571821/395857
        x does not have to be ordered.
        '''
        # Scatter plot
        fig, ax = plt.subplots()
        plt.scatter(x, y, s=40, c='r', alpha = 0.6, marker='o')
        plt.margins(0, 0)
        if not without_correlation_line:
            # Add correlation line
            axes = plt.gca()
            m, b = np.polyfit(x, y, 1)
            X_plot = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
            plt.plot(X_plot, m*X_plot + b, '-', c = 'b')
        plt.xlabel("Target")
        plt.ylabel("Prediction")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.tight_layout()
        #plt.show()
        # Save figure
        graph_filepath = os.path.join(directory, 'treeReg_Plot_' + trail +'.pdf')
        plt.savefig(graph_filepath, dpi=300, format='pdf', bbox_inches='tight')  
        
        return ax
    
#def main():
#    # Data
#    x = np.random.rand(100)
#    y = x + np.random.rand(100)*0.1
#
#    # Plot
#    getScatterPlot(x, y, 'scatter_plot')
#
#if __name__ == "__main__":
#    main()