# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:39:52 2019

@author: yl918888
"""
from src.reporting.display_tree import DisplayTree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#% ------------------------------------------------------------
def elpsadeTime(start, end):
    delta = end - start
    return delta.total_seconds()

#% -----------------------------------------------------------
def optimalTreeFitness(evaluateTree, pTree, pSet = 'test'):
    evaluateTree.set_dataset_to_evaluate(pSet)
    _ = evaluateTree.getTreePredictedOutputs(pTree)
    return evaluateTree.getTreeFitness() + [pTree.getTreeSize()]

#% ------------------------------------------------------------
def saveModel(n_tree, algo, directory, trail, index = '', saveBothFormat = True):
    d_tree = DisplayTree() # Creating object fo0r display tree
    trail = str(algo) + '_' + str(trail) + index
    if saveBothFormat:
        d_tree.saveTreeOutline(n_tree, directory, trail)
        d_tree.saveTreeModel(n_tree, directory, trail)
    else:
        d_tree.saveTreeModel(n_tree, directory, trail)

#% -------------------------------------------------------------
def plotTreeFitness(evaluateTree, pTree, directory, trail, algo = 'gp_gd', pSet = 'test'):
    evaluateTree.set_dataset_to_evaluate(pSet)
    _ = evaluateTree.getTreePredictedOutputs(pTree)
    trail = str(algo) + '_' + str(pSet) + '_' + str(trail)
    evaluateTree.plot(directory, trail)
    
def plotPerfromance(performance, algo, directory, trail, class_names, isClass = 'Classification'):
    x = np.arange(len(performance))
    y = [row[0] for row in performance]
    
    fig, ax = plt.subplots()
    plt.plot(x, y, color='b')
    #plt.margins(0)
    plt.xlabel('generations '+ str(algo))
    plt.ylabel('error rate')
    plt.ylim([0.0, 1.05])
    #plt.show()    
    plt.tight_layout()    
    # Save figure
    graph_filepath = os.path.join(directory, 'treePerformance_' + str(algo) + '_' + str(trail) +'.pdf')
    plt.savefig(graph_filepath, dpi=300, format='pdf', bbox_inches='tight')  
    #plt.close()
    if isClass == 'Classification':
        CLASSES = len(performance[0][1])
        fig = plt.figure()
        sns.reset_orig()  # get default matplotlib styles back
        clrs = sns.color_palette('husl', n_colors=CLASSES)  # a list of RGB tuples
        fig, ax = plt.subplots(1)
        for i in range(CLASSES):
            prec = [row[1][i] for row in performance] # precision
            recall = [row[2][i] for row in performance] # recall, sensictivty
            #specificity = [row[0][3][i] for row in performance] # specificity
            lines = ax.plot(recall, prec, marker='o',  label=str(class_names[i]))
            lines[0].set_color(clrs[i])
            #plt.plot(recall, prec, label=str(class_names[i]))
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('recall')
        plt.ylabel('precision')    
        plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
        plt.tight_layout()    
        #plt.show()
        graph_filepath = os.path.join(directory, 'treePrecRecall_Curve_' + str(algo) + '_' + str(trail) +'.pdf')
        fig.savefig(graph_filepath)
        
        
    
def plotPerfromanceBoth(performanceGP, performanceMH, algo_gp, algo_mh, directory, trail, class_names, isClass = 'Classification'):
    performance = performanceGP + performanceMH
    x = np.arange(len(performance))
    y = [row[0] for row in performance]
    x0 = len(performanceGP)
    clrs = sns.color_palette('husl', n_colors=2)  # a list of RGB tuples
    fig, ax = plt.subplots()
    plt.plot(x[:x0+1], y[:x0+1], color=clrs[0], label = str(algo_gp))
    plt.plot(x[x0:], y[x0:], color=clrs[1], label = str(algo_mh))

    #plt.margins(0)
    plt.xlabel('generations '+ str(algo_gp) +' and '+str(algo_mh))
    plt.ylabel('error')
    plt.ylim([0.0, 1.05])
    plt.tight_layout()    
    plt.legend(loc='upper right')
    #plt.show()
    # Save figure
    graph_filepath = os.path.join(directory, 'treePerformance_' +  str(algo_gp) +'_'+str(algo_mh)  + '_' + str(trail) +'.pdf')
    plt.savefig(graph_filepath, dpi=300, format='pdf', bbox_inches='tight')  
    
    if isClass == 'Classification':
        CLASSES = len(performance[0][1])
        fig = plt.figure()
        sns.reset_orig()  # get default matplotlib styles back
        clrs = sns.color_palette('husl', n_colors=CLASSES)  # a list of RGB tuples
        fig, ax = plt.subplots(1)
        for i in range(CLASSES):
            prec = [row[1][i] for row in performance] # precision
            recall = [row[2][i] for row in performance] # recall, sensictivty
            #specificity = [row[0][3][i] for row in performance] # specificity
            lines = ax.plot(recall, prec, marker='o',  label=str(class_names[i]))
            lines[0].set_color(clrs[i])
            #plt.plot(recall, prec, label=str(class_names[i]))
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('recall')
        plt.ylabel('precision')    
        plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
        plt.tight_layout()    
        #plt.show()
        # Save figure
        graph_filepath = os.path.join(directory, 'treePrecRecall_Curve_' +  str(algo_gp) +'_'+str(algo_mh) + '_' + str(trail) +'.pdf')
        fig.savefig(graph_filepath)
           
    
def plotPrecisionVsRecall(performance, algo, directory, trail, class_names, pSet):
    CLASSES = len(performance[1])
    fig = plt.figure()
    sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette('husl', n_colors=CLASSES)  # a list of RGB tuples
    fig, ax = plt.subplots(1)
    for i in range(CLASSES):
        prec = performance[1][i] # precision
        recall = performance[2][i] # recall, sensictivty
        #specificity = performance[3][i] # specificity
        ax.scatter(recall, prec, label=str(class_names[i]), color=clrs[i])
    
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')    
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    plt.tight_layout()    
    #plt.show()
    graph_filepath = os.path.join(directory, 'treePrecRecall_Plot_' + str(algo) + '_' + str(pSet)+ '_' +str(trail) +'.pdf')
    fig.savefig(graph_filepath)
    
def getTraingParamsDict(params):
    training_params = {}
    for key in vars(params):
        if str(key) == 'n_data_input_values' :
            training_params.update({'n_data_input_examples' : params.__dict__[key].shape})    
            continue
        if str(key) == 'n_data_target_values':
            training_params.update({'n_data_target_examples' : params.__dict__[key].shape})    
            continue
        training_params.update({str(key) : params.__dict__[key]})
    return training_params


def plotParetoFront(x,y, directory, trial, algo = 'gp_', plotName= 'treePopulation_'):
    '''
        GP population data and saving error rate and tree size
    '''
    # Plot
    fig, ax = plt.subplots()
    plt.scatter(x, y, alpha=0.6, s=80, color='b', marker ='o')
    #plt.margins(0,0)
    plt.xlabel('error rate')
    plt.ylabel('tree size')
    plt.tight_layout()    
    # Save figure
    graph_filepath = os.path.join(directory, str(plotName) + str(algo) + str(trial) +'.pdf')
    plt.savefig(graph_filepath, dpi=300, format='pdf', bbox_inches='tight')  
    graph_filepath = os.path.join(directory, str(plotName) + str(algo) + str(trial))
    np.save(graph_filepath, zip(x,y))
    #plt.show()
    #plt.close()
    
def saveGPIteration(pPopulation, directory, trial, dirItr = '0', algo = 'gp_', plotName= 'treeItrPopulation_'):
    '''
        Saveing GP iteration data
    '''
    directoryItr = os.path.join(directory, str(dirItr))
    try:
        os.makedirs(directoryItr)
    except OSError:
        print ("Directory already exisit or fails to create one")
    # Save iteration_data
    x = [pPopulation[i].mCost[0] for i in range(len(pPopulation))]
    y = [pPopulation[i].mTree.getTreeSize() for i in range(len(pPopulation))]
    
    for indx in range(len(pPopulation)):
        saveModel(pPopulation[indx].mTree, algo, directoryItr, trial, dirItr+'_'+str(indx), False)
    data_filepath = os.path.join(directoryItr, str(plotName) + str(algo) + str(trial))
    np.save(data_filepath, zip(x,y))
