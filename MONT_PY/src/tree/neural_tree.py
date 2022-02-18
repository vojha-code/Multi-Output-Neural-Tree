'''
Generating random generic tree
Author: v ojha
Affiliation Uni of Reading

Naming convention:
    m_var - meber of class
    p_var - parameters of the functions
    n_var - local variables
'''
#custom class dependencies
from src.tree.node import Node
from src.tree.function_node import FunctionNode

# importing python dependecies
import numpy as np
import random
import ast


#class tree taks 
class NeuralTree:
    m_root = None # inialize root as a function node  
    m_depth = 0 #depth/heigt of the tree (including root level) initialize to zeero
    m_treeFitness = 9999999 # arbitary value to set fitness
    m_max_children = 0
    
    FunNodesList = [] # list of function nodes
    LeafNodesList = [] # list of leef nodes
    
    def __init__(self):
        self.m_root = None # inialize root node         
        self.FunNodesList = [] # list of function nodes
        self.LeafNodesList = [] # list of leef nodes 
        self.m_max_children = 0
        self.m_depth = 0
        self.m_treeFitness = 999 # arbitary value to set fitness
    
    #----------------------------------------------------------------------------------------------------------        
    def genrateGenericRandomTree(self, params):
        '''
            Generates a random tree         
                :param    m_max_input_attr     maximum number of input variable (attributes) in the data set
                :param    m_max_children       maximum number of childs one node can takes
                :param    m_max_depth          maximum depth/height of the tree
        '''  
        p_max_target_attr = params.n_max_target_attr
        p_max_children = params.n_max_children
        p_fun_range = params.n_fun_range
        p_fun_type = params.n_fun_type
        p_out_fun_type = params.n_out_fun_type
        
        # Strict ROOT function type setting for regression problem whoose output is normalized between 0 and 1, use Guassian function
        # if it is not already set to Guassian or sigmoid
        #if p_max_target_attr == 1 and (p_fun_type != 'Gaussian' or p_fun_type != 'sigmoid'): # root node for a regression  probl
        #    p_fun_type = 'Gaussian'
        
        n_root_weight = 1.0 # Fix weight for root node (not to  be used anyway - it's only for Function Node to work)
        n_root_bias = 1.0 # Fix bias for root node (not to  be used anyway - it's only for Function Node to work)
        n_min_child = 2 # setting minimum nodes for a node in the tree is 2
        n_arity_range = p_max_children - n_min_child # compute range for randomize number of child for a node
        
        # Check number of outputs of the proble
        if(p_max_target_attr > 1):
            n_children = p_max_target_attr # fix the number of child of a tree for a classification problem
        else:
            n_children = n_min_child + random.randrange(n_arity_range) # random children numbner (ensure atleast 2 child)
        
        # a,b,c -> a and b paramters of the activation fun at the fun node, c is function type
        # Function params are mostly used for Guassian function
        if p_out_fun_type == 'softmax':
            n_fun_params = [random.uniform(p_fun_range[0],p_fun_range[1]), # unused for softmax
                            random.uniform(0.1,1.0),   # unused for softmax /tanh/sigmoid
                            p_out_fun_type] #  this can be anny thing Gaussing and tanh, sigm etc.
        else:
            n_fun_params = [random.uniform(p_fun_range[0],p_fun_range[1]), 
            random.uniform(0.1,1.0), 
            p_fun_type]

        
        # Intialize root node of the tree  and set parten as none   
        n_current_depth = -1 # Set depth as -1 for root node since root do not have a parent node - None
        # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
        self.m_root = FunctionNode(n_root_weight, n_root_bias, n_fun_params, n_children, None, n_current_depth)
        
        n_current_depth = 0 # re -set current depth (root node depth)  to zero                
        # Generate children of the tree - gerenate the whole tree in recursive mannen
        self.m_root.genrateChildren(params, n_current_depth)
        
        self.inspectChildNodes() # calling a function withing the class to inspect
    #End defination generic Random tree    
    
    #----------------------------------------------------------------------------------------------------------
    def print_tree(self):
        '''
        print the tree in JSON formate - It will create a dictnary
        '''
        return  "{ " + self.m_root.print_node(0) # returns a JSON string
    # End defniation print_tree
    
    #----------------------------------------------------------------------------------------------------------
    def saveTree(self):
        '''
        Save the tree in JSON formate - It will create a dictnary
        '''
        return  "{ " + self.m_root.saveNode(0) # returns a JSON string
    # End defniation print_tree
    
    
    #----------------------------------------------------------------------------------------------------------
    def retrive_JSON_TreeOriginal(self, p_json_tree):
        '''
        retrive the tree in JSON formate - It will create a dictnary
        '''
        _NUL = object()  # unique value guaranteed to never be in JSON data
        if isinstance(p_json_tree, dict):
            n_root_children = p_json_tree.get('children', _NUL)  # _NUL if key not present
            if n_root_children is not _NUL:
                n_list_names = p_json_tree['name']
                n_list_names = n_list_names.split(';')
                if(len(n_list_names) < 4 and len(n_list_names) > 4 ):
                    print('Please input a correct tree mode file treeModel.json')
                    return None
        
                # names - > zero is number of children
                #n_children = int(n_list_names[0].split(':')[1])
                # Alternatively can number of children taken from children size
                n_children = len(p_json_tree['children'])
                # names - > one is the weight
                n_root_weight = float(n_list_names[1].split(':')[1])
                # names - > two is the bias
                n_root_bias = float(n_list_names[2].split(':')[1])
                # names - > three is the function paramter
                n_fun_params = ast.literal_eval(n_list_names[3].split(':')[1])
                
                
                
                n_current_depth = -1 # Set depth as -1 for root node
                # set parent None: # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
                self.m_root =  FunctionNode(n_root_weight, n_root_bias, n_fun_params, n_children, None, n_current_depth)
                
                n_current_depth = 0 #
                self.m_root.retriveChildren(n_root_children, n_current_depth)
                #print('retrviving root child')    
                self.inspectChildNodes() # calling a function withing the class to inspect
            else:
                print('Root has not children \'key\' somting wrong with the input file')    
        else:
            print('Require json file loaded as a dictinary format')
            
        print('Tree succefully retirived')
    # End defniation print_tree

    #----------------------------------------------------------------------------------------------------------
    def retrive_JSON_Tree(self, p_json_tree):
        '''
        retrive the tree in JSON formate - It will create a dictnary
        '''
        _NUL = object()  # unique value guaranteed to never be in JSON data
        if isinstance(p_json_tree, dict):
            n_root_children = p_json_tree.get('children', _NUL)  # _NUL if key not present
            if n_root_children is not _NUL:
                n_list_names = p_json_tree['name']
                n_list_names = n_list_names.split(';')
                if(len(n_list_names) < 4 and len(n_list_names) > 4 ):
                    print('Please input a correct tree mode file treeModel.json')
                    return None
        
                # names - > zero is number of children
                #n_children = int(n_list_names[0].split(':')[1])
                # Alternatively can number of children taken from children size
                n_children = len(p_json_tree['children'])
                # names - > one is the weight
                n_root_weight = float(n_list_names[1].split(':')[1])
                # names - > two is the bias
                n_root_bias = float(n_list_names[2].split(':')[1])
                # names - > three is the function paramter
                #print(n_list_names)
                #print(n_list_names[3])
                strVal = str(n_list_names[3].split(':')[1])+""
                #print(strVal)
                n_list_string = strVal
                n_list_string = n_list_string.replace("[","")
                n_list_string = n_list_string.replace("]","")
                n_list_string = n_list_string.replace(" ","").split(',')
                n_fun_params = [float(n_list_string[0]), float(n_list_string[1]), n_list_string[2]]
                #print(n_fun_params)
                
                
                n_current_depth = -1 # Set depth as -1 for root node
                # set parent None: # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
                self.m_root =  FunctionNode(n_root_weight, n_root_bias, n_fun_params, n_children, None, n_current_depth)
                
                n_current_depth = 0 #
                self.m_root.retriveChildren(n_root_children, n_current_depth)
                #print('retrviving root child')    
                self.inspectChildNodes() # calling a function withing the class to inspect
            else:
                print('Root has not children \'key\' somting wrong with the input file')    
        else:
            print('Require json file loaded as a dictinary format')
            
        #print('Tree succefully retirived')
    # End defniation print_tree    
    
    
    #----------------------------------------------------------------------------------------------------------
    def inspectChildNodes(self):
        '''
            Thi function inspect function nodes and leaf nodes and create a list of them
        '''
        self.FunNodesList.clear()
        self.LeafNodesList.clear()
        self.m_depth = 0
        self.m_root.inspect_node(self,0)
    
    #----------------------------------------------------------------------------------------------------------    
    def addLeafNodesList(self, p_LeafNode):
        '''
            add leaf child 
                :param    p_LeafNode    an object of leaf nodes
        '''
        self.LeafNodesList.append(p_LeafNode)
    
    #----------------------------------------------------------------------------------------------------------
    def getLeafNodesList(self):
        '''
            return leaf child nodes list
        '''
        return self.LeafNodesList
    
    #----------------------------------------------------------------------------------------------------------
    def addFunNodesList(self, p_FuntNode):
        '''
            add function node child 
                :param    p_FuntNode    an object of function nodes
            
        '''
        self.FunNodesList.append(p_FuntNode)   
    
    #----------------------------------------------------------------------------------------------------------
    def getFunNodesList(self):
        '''
            return function node child list
            
        '''
        return self.FunNodesList
    
    def getTreeInputFeatuereProperties(self, p_target_attr_count, input_featuer_names, outputs_names):
        if type(input_featuer_names) == list:
            input_featuer_names = np.array(input_featuer_names)
            
        if not(type(outputs_names) == list):
            outputs_names = outputs_names.tolist()
            
        root_node_input_attr_all = []
        for node in self.getLeafNodesList():
            root_node_input_attr_all.append(node.getInputAttribute())
            
        root_node_unique_featuers = list(np.unique(root_node_input_attr_all))
        root_node_unique_featuers_names = input_featuer_names[np.unique(root_node_input_attr_all)].tolist()
        root_node_each_featuers_count = []
        for value in root_node_unique_featuers:
            root_node_each_featuers_count.append(root_node_input_attr_all.count(value))
        
        if p_target_attr_count == 1:
            each_out_nodes_feature_all = root_node_input_attr_all
            each_out_nodes_feature_unique = root_node_unique_featuers
            each_out_nodes_feature_unique_names = root_node_unique_featuers_names
            each_out_nodes_each_feature_count = root_node_each_featuers_count
            each_out_nodes_feature_count = self.getTreeSize()

        else:
            each_out_nodes_feature_all = []
            each_out_nodes_feature_unique = []
            each_out_nodes_feature_unique_names = []
            each_out_nodes_feature_count = []
            each_out_nodes_each_feature_count = []
            for node in self.m_root.m_ChildrenList:
                out_noeds_inputs_attr = []
                node_count = []
                out_noeds_inputs_attr, node_count = node.getLeafNodesValue(out_noeds_inputs_attr, node_count)
                # collecting nodes in a list 
                each_out_nodes_feature_all.append(out_noeds_inputs_attr)
                this_node_unique_featuers = np.unique(out_noeds_inputs_attr)
                each_out_nodes_feature_unique.append(list(this_node_unique_featuers))
                each_out_nodes_feature_unique_names.append(input_featuer_names[this_node_unique_featuers].tolist())
                this_node_each_feature_count = []
                for value in this_node_unique_featuers:
                    this_node_each_feature_count.append(out_noeds_inputs_attr.count(value))
                each_out_nodes_each_feature_count.append(this_node_each_feature_count)
                each_out_nodes_feature_count.append(len(node_count)+1)
        
        tree_feature_properties = {}
        tree_feature_properties.update({'tree_out_nodes': outputs_names})        
        tree_feature_properties.update({'tree_featueres_all': root_node_input_attr_all})
        tree_feature_properties.update({'tree_featueres_unique': root_node_unique_featuers})
        tree_feature_properties.update({'tree_featueres_unique_names': root_node_unique_featuers_names})
        tree_feature_properties.update({'tree_each_featuere_count': root_node_each_featuers_count})
        tree_feature_properties.update({'tree_size_overall': self.getTreeSize()})
        tree_feature_properties.update({'each_out_node_featueres_all': each_out_nodes_feature_all})
        tree_feature_properties.update({'each_out_node_featueres_unique': each_out_nodes_feature_unique})
        tree_feature_properties.update({'each_out_node_featueres_unique_names': each_out_nodes_feature_unique_names})
        tree_feature_properties.update({'each_out_node_each_featueres_count': each_out_nodes_each_feature_count})
        tree_feature_properties.update({'each_out_node_size': each_out_nodes_feature_count})

        return tree_feature_properties
    
    #----------------------------------------------------------------------------------------------------------
    def setDepth(self, p_depth):
        '''
            setting variavle depth o the tree
                :param    p_depth current depth set after inspection of the tree
        '''
        self.m_depth = p_depth
    #----------------------------------------------------------------------------------------------------------   
    def getDepth(self):
        '''
            returning tree current depth
        '''
        return self.m_depth
    #----------------------------------------------------------------------------------------------------------
    def setTreeFitness(self, p_fitness):
        '''
            set the fitness of the tree
            (depend on the objective function user RMSE of Error Rate)
                :param    p_fitness    RMSE or Erorr rate of tree for a training data
        '''
        self.m_treeFitness = p_fitness
    #----------------------------------------------------------------------------------------------------------   
    def getTreeFitness(self):
        '''
            return fitness of the tree
            (depend on the objective function user RMSE of Error Rate)
        '''
        return self.m_treeFitness
    #----------------------------------------------------------------------------------------------------------
    def getTreeSize(self):
        '''
            Retrun total number of nodes in the tree (i.e a root node have)        
        '''
        return len(self.FunNodesList) + len(self.LeafNodesList) + 1 # + 1 for root node
    
    #----------------------------------------------------------------------------------------------------------
    def getTreeFuncNodeSize(self):
        '''
            Retrun total number of nodes in the tree (i.e a root node have)        
        '''
        return len(self.FunNodesList)

    #----------------------------------------------------------------------------------------------------------
    def getTreeLeafNodeSize(self):
        '''
            Retrun total number of nodes in the tree (i.e a root node have)        
        '''
        return len(self.LeafNodesList)
    
    #----------------------------------------------------------------------------------------------------------
    def copy_Tree(self):
        '''
            returen a copy of itself with new instance / refernce to objects
        '''
        # returning a new tree object - other old object will be modified
        n_tree = NeuralTree() 
        
        n_tree.m_max_children = self.m_max_children
        n_tree.m_depth = self.m_depth
        n_tree.m_treeFitness = self.m_treeFitness
        n_tree.m_root = self.m_root.copyNode(None)
        n_tree.inspectChildNodes()
        return n_tree
    #----------------------------------------------------------------------------------------------------------
    def getOutput(self, p_input_attr_val, p_target_attr_count):
        '''
            Evaluate tree output for a given inputs data
            param:    p_input_attr_val          input example vector 
            param:    p_output_attr_count       target columns number - equivalent to number of classes
        '''
        if(p_target_attr_count > 1):
            '''
            MULITI OUTPUT - REGRSSION / CLASSIFICATION Problem for each child node
            Evaluate the function nodes output for a given inputs data     
            '''
            if (self.m_root.m_FunctionParams.__contains__('softmax')): 
                return self.softmax(self.m_root.getMultiNodeOutput(p_input_attr_val))
            else:
                return self.m_root.getMultiNodeOutput(p_input_attr_val)
        else:
            '''
            SINGLE OUTPUT For regression / clasification and normal node output use this function
            Evaluate the function nodes output for a given inputs data 
            '''
            return self.m_root.getSingleNodeOutput(p_input_attr_val)
                    
    #----------------------------------------------------------------------------------------------------------
    def getTreeParameters(self, p_target_attr_count, pRerival = 'all'):
        '''
            Retriving tree parameters
            args:
                param: p_target_attr_count number of output class. for classification it should be 2 or more. 
                For regression MUST be 1
            Returns a vector of edge and function parameters values
        '''
        nParameters = []
        if pRerival == 'all': #  retrive Weight, Bias, and Function params
            # For single output problem - ROOT NODE's parameters (Bias and Function_params) are usefull (not weight)
            # Else all root node's parameters are useless
            if p_target_attr_count == 1:
                # NO WEIGHT beacuse root is output node and its weight has no use
                # TAKE ROOT BIAS
                nParameters.append(self.m_root.getBias()) 
                funParams = self.m_root.m_FunctionParams # TAKE ROOT FUN PARAMS
                for i in range(0,len(funParams)-1):
                    nParameters.append(funParams[i]) # adding a and b of fun params
                    
            # FUNCTION NODES - Retrive weight, bias, and func prameters [a and b] from the function list
            for node in self.FunNodesList:
                if p_target_attr_count == 1:
                    # For single output problem, root node is the output node 
                    # Hence all its children function nodes are hidden nodes. This their weights are usefull 
                    nParameters.append(node.getEdgeWeight())# # take all hidden function node's (roots any child node) weights
                else:
                    # For multi output problem root node is useless and all roots imediate child nodes are function nodes
                    # Hence take ONLY NON-ROOT child weights
                    if(node not in self.m_root.m_ChildrenList):# NOT a ROOT's Imediate child
                        nParameters.append(node.getEdgeWeight()) # TAKE WEIGHTS ONLY for NON OUTPUT NODE
                
                
                # check if node is NOT Gausian
                funParams = node.m_FunctionParams # TAKE NODE's FUN PARAMS
                if (not funParams.__contains__('Gaussian')):
                    # Retriving Function bias 
                    nParameters.append(node.getBias()) # TAKE NODE's BIAS
                    
                # Retriving function params
                for i in range(0,len(funParams)-1):
                    nParameters.append(funParams[i])
                    
            # LEAF NODES - retrive weight and prameters a, b from the function list
            for node in self.LeafNodesList:
                nParameters.append(node.getEdgeWeight())
          
        elif pRerival == 'weights_and_bias':
            if p_target_attr_count == 1:
                # NO WEIGHT only # TAKE ROOT BIAS
                nParameters.append(self.m_root.getBias()) 
                
            # retrive fnction node weights and prameters a, b from the function list
            for node in self.FunNodesList:
                # Retriving weights
                if p_target_attr_count == 1:
                    # for single output problem take all its child weights
                    nParameters.append(node.getEdgeWeight())
                else:
                    # for multi output  problem only take non roots child weights
                    if(node not in self.m_root.m_ChildrenList):# NOT IN ROOT CHILD -> ok
                        nParameters.append(node.getEdgeWeight()) # TAKE WEIGHTS ONLY for NON OUTPUT NODE
                # Retriving bias for all function nodes
                nParameters.append(node.getBias()) # TAKE NODE's BIAS
                
            # retrive leaf node weights - it has no bias
            for node in self.LeafNodesList:
                nParameters.append(node.getEdgeWeight())
                
        elif pRerival == 'weights':
            # retrive fnction node weights and prameters a, b from the function list
            for node in self.FunNodesList:
                # Retriving weights
                if p_target_attr_count == 1:# for regression proble take all its child weights
                    nParameters.append(node.getEdgeWeight())
                else:# if multi output  problem only take non roots child weights
                    if(node not in self.m_root.m_ChildrenList):# NOT IN ROOT CHILD -> ok
                        nParameters.append(node.getEdgeWeight()) # TAKE WEIGHTS ONLY for NON OUTPUT NODE
                
            # retrive leaf node weights
            for node in self.LeafNodesList:
                nParameters.append(node.getEdgeWeight())
                
        elif pRerival == 'bias':
            if p_target_attr_count == 1:
                # NO WEIGHT only # TAKE ROOT BIAS
                nParameters.append(self.m_root.getBias()) 
            # retrive fnction node weights and prameters a, b from the function list
            for node in self.FunNodesList:
                # Retriving bias for all function nodes
                nParameters.append(node.getBias()) # TAKE NODE's BIAS
        else:
            print('please input one of these: all, weights_and_bias,  weights,  bias')
        # return accumulated parameters
        return nParameters         
    
    #----------------------------------------------------------------------------------------------------------
    def setTreeParameters(self, pParameters, p_target_attr_count, pSet = 'all'):
        '''
            Set tree edges with the vector pEdgeWeightsVec
        '''
        if pSet == 'all': # set Weight, Bias, and Function params
            indx = 0
            # setting roots function parameters only for regeression problem
            if p_target_attr_count == 1:
                # ROOT  NODE's bias and parameter only usefull in regression problems
                self.m_root.setBias(pParameters[indx]) # SET ROOT's BIAS
                indx += 1
                for i in range(0,len(self.m_root.m_FunctionParams)-1):
                    self.m_root.m_FunctionParams[i] = pParameters[indx]
                    indx += 1
            # set function node weights and prameters a, b from the function list
            for node in self.FunNodesList:
                # SET nodes weights
                if p_target_attr_count == 1:
                    # For single output problem child weights were taken so will be set
                    node.setEdgeWeight(pParameters[indx])
                    indx += 1
                else:
                    # For multi output problem problem only non roots child weights were taken 
                    # so only non roots child weight will be set
                    if(node not in self.m_root.m_ChildrenList):# SET Weight only for node NOT IN ROOT CHILD -> ok
                        node.setEdgeWeight(pParameters[indx])
                        indx += 1
                
                # check if node is NOT Gausian
                funParams = node.m_FunctionParams 
                if (not funParams.__contains__('Gaussian')):
                    # SET nodes bias
                    node.setBias(pParameters[indx]) # SET ROOT BIAS
                    indx += 1
                
                #SET nodes function params
                for i in range(0,len(self.m_root.m_FunctionParams)-1):
                    node.m_FunctionParams[i] = pParameters[indx]
                    indx += 1
                    
            # set leaf node weights
            for node in self.LeafNodesList:
                node.setEdgeWeight(pParameters[indx])
                indx += 1
                
        if pSet == 'weights_and_bias': 
            indx = 0
            if p_target_attr_count == 1:
                # ROOT  NODE's bias and parameter only usefull in regression problems
                self.m_root.setBias(pParameters[indx]) # SET ROOT's BIAS
                indx += 1           
            # set node weights 
            for node in self.FunNodesList:
                # SET nodes weights
                if p_target_attr_count == 1:# for regression problem child weights were taken so will be set
                    node.setEdgeWeight(pParameters[indx])
                    indx += 1
                else:# if classification problem only take non roots child weights
                    if(node not in self.m_root.m_ChildrenList):# SET Weight only for node NOT IN ROOT CHILD -> ok
                        node.setEdgeWeight(pParameters[indx])
                        indx += 1
                # SET nodes bias
                node.setBias(pParameters[indx]) # SET ROOT BIAS
                indx += 1
            # set leaf node weights
            for node in self.LeafNodesList:
                node.setEdgeWeight(pParameters[indx])
                indx += 1 
                
        if pSet == 'weights': 
            indx = 0
            # set node weights 
            for node in self.FunNodesList:
                # SET nodes weights
                if p_target_attr_count == 1:# for regression problem child weights were taken so will be set
                    node.setEdgeWeight(pParameters[indx])
                    indx += 1
                else:# if classification problem only take non roots child weights
                    if(node not in self.m_root.m_ChildrenList):# SET Weight only for node NOT IN ROOT CHILD -> ok
                        node.setEdgeWeight(pParameters[indx])
                        indx += 1

            # set leaf node weights
            for node in self.LeafNodesList:
                node.setEdgeWeight(pParameters[indx])
                indx += 1
                
        if pSet == 'bias': 
            indx = 0
            if p_target_attr_count == 1:
                # ROOT  NODE's bias and parameter only usefull in regression problems
                self.m_root.setBias(pParameters[indx]) # SET ROOT's BIAS
                indx += 1   
            # set node weights 
            for node in self.FunNodesList:
                # SET nodes bias
                node.setBias(pParameters[indx]) # SET ROOT BIAS
                indx += 1
    #END of function setWeight
    
    #----------------------------------------------------------------------------------------------------------
    def getGradient(self, x, y, p_target_attr_count):
        ypred = self.getOutput(x, p_target_attr_count)
        #print('y_pred: ', ypred, y, type(ypred), type(y))
        if (type(ypred) is not list):
            ypred = [ypred]
            y = [y]
        
        #if p_target_attr_count > 1:
        #    ypred = self.softmax(ypred)
            #print(ypred[0])
        
        self.setDeltaJ(y, ypred, p_target_attr_count)
        self.setGradient(x, p_target_attr_count)
        return self.getDeltaWeightandBias(p_target_attr_count)

    #----------------------------------------------------------------------------------------------------------
    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)
    #----------------------------------------------------------------------------------------------------------
    def setDeltaJ(self, y, ypred, p_target_attr_count):
        '''
            Setting delta_j of each function node of the tree
            param:  x                       input example vector 
            param:  y                       output vector (target)
            param:  y_pred                  tree output vector 
            param:  p_target_attr_count        target columns number - equivalent to number of classes
        '''
        if p_target_attr_count == 1:
            # Root node delta j is only useful for single output problems
            #print('root delta: ')
            self.m_root.setDeltaJ(y, ypred, 0, True)
        else:
            for node in self.m_root.m_ChildrenList:
                #print('root ',  self.m_root.m_ChildrenList.index(node),' delta: ')
                node.setDeltaJ(y, ypred, self.m_root.m_ChildrenList.index(node), True)
                
  
    #----------------------------------------------------------------------------------------------------------
    def setGradient(self, x, p_target_attr_count):
        '''
            Setting delta_j of each function node of the tree
            param:  p_target_attr_count        target columns number - equivalent to number of classes
        '''
        if p_target_attr_count == 1:
            # No need to use gradient of weight for the output node but gradient of bias is necessary
            self.m_root.setGradient([], False) # for output nodes we set only bias -so set both w and b = False
            #print('delta_j',self.m_root.m_delta_j)
        
        # For all function node of the root node 
        for node in self.FunNodesList:
            #print('delta_j',node.m_parent_node.m_delta_j)
            if p_target_attr_count == 1:
                # for single output problem, all child (of root) takes weight and biad. 
                # Hence we set delta_weight and delta_bias 
                node.setGradient()
            else:
                # if multi output problem only set non roots child take weights and bias. 
                # Hence we set delta weights and delta_bias
                if(node not in self.m_root.m_ChildrenList):# NOT IN ROOT CHILD -> ok
                    # for NON OUTPUT nodes delta_weight and delta_bias are nececcary
                    node.setGradient() 
                else:
                    # for OUTPUT nodes only delta_bias are nececcary
                    node.setGradient([], False) # for output nodes we set only bias -soe set both w and b = False
                   
        # For all leaf node only weights are necessary
        for node in self.LeafNodesList:
            node.setGradient(x)    

    #----------------------------------------------------------------------------------------------------------
    def getDeltaWeightandBias(self, p_target_attr_count):
        nGrad = []
        if p_target_attr_count == 1:
            # NO WEIGHT only # TAKE ROOT BIAS
            nGrad.append(self.m_root.getDeltaBias()) 
            
        # retrive fnction node weights and prameters a, b from the function list
        for node in self.FunNodesList:
            # Retriving weights
            if p_target_attr_count == 1:
                # for single output problem take all its child weights
                nGrad.append(node.getDeltaEdgeWeight())
            else:
                # for multi output  problem only take non roots child weights
                if(node not in self.m_root.m_ChildrenList):# NOT IN ROOT CHILD -> ok
                    nGrad.append(node.getDeltaEdgeWeight()) # TAKE WEIGHTS ONLY for NON OUTPUT NODE
            # Retriving bias for all function nodes
            nGrad.append(node.getDeltaBias()) # TAKE NODE's BIAS
            
        # retrive leaf node weights - it has no bias
        for node in self.LeafNodesList:
            nGrad.append(node.getDeltaEdgeWeight())

        return nGrad