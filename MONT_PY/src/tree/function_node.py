'''
Generating random generic tree
Author: v ojha
Affiliation Uni of Reading

'''

from src.tree.node import Node
from src.tree.leaf_node import LeafNode
from src.tree.activation_fun import ActivationFunction
import ast
import random

class FunctionNode(Node):
    '''
        Function node in addition to its own members, inherits 
        m_edge_weight,m_parent_node, and some functions inherited from Node
    '''
    m_depth = 0 # initalize to be set by constructor -  current (self) depth
    m_children = 0 # initalize to set by constructor - current (self) children
    m_ChildrenList = [] # contain list of children nod of this (self list of children) node
    m_bias = None #bias node initialize to randaom value
    m_delta_bias = None # gradient of the node
    m_delta_j = None # function nde delta_j for dlata weight computation
    m_FunctionParams = []

    #m_Random
    
    #constrcutor for Function node
    # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
    def __init__(self, p_weight, p_bias, p_fun_params, p_children, p_parent_node, p_depth):
        '''
        Genreating the Function node through its constrcutor
            :parm    p_weight           weigth of the eadge connecting leaf node
            :parm    p_children         number of children of this function node
            :parm    p_depth            current depth of the function node
            :parm    p_parent_node      parent node of this function node
            :parm    p_fun_params       list pof paramter a function takes
        
        '''
        self.m_depth = p_depth
        self.m_edge_weight = p_weight
        self.m_bias = p_bias
        self.m_children = p_children
        self.m_ChildrenList = [] # clear list before appending to it.
        # At least three parameters a,b,c -> a and b paramters of the activation fun at the fun node, c is function type
        self.m_FunctionParams = [p for p in p_fun_params] # will create ne objects
        self.m_parent_node = p_parent_node        
        
    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def genrateChildren(self, params, p_current_depth):
        '''
            Genrates a random tree. It is recursive procedue for tree generation.
                :parm    p_max_input_attr       maximum number of input attributes
                :pram    p_max_children         maximumn number of children one node can take
                :pram    p_max_depth            maximum depth of the whole tree
                :pram    p_weight_range         edge weight's range
                :pram    m_current_depth        current depth of the tree
        '''
        
        p_max_target_attr = params.n_max_target_attr
        p_max_input_attr = params.n_max_input_attr
        p_max_children = params.n_max_children
        p_max_depth = params.n_max_depth
        p_weight_range = params.n_weight_range
        p_fun_range = params.n_fun_range
        p_fun_type = params.n_fun_type
        p_out_fun_type = params.n_out_fun_type
        
        if(p_current_depth < p_max_depth):
            #iterate through number of childern of this (current, i.e., self) function node
            for i  in range(self.m_children):
                n_weight = random.uniform(p_weight_range[0],p_weight_range[1]) # tree edge weight between uniformaly taken between 0 and 1   
                n_bias = random.uniform(p_weight_range[0],p_weight_range[1]) # tree bias weight between uniformaly taken between 0 and 1   
                n_min_child = 2 # in any case min number of childrren will be 2 for a node
                # a,b,c -> a and b paramters of the activation fun at the fun node, c is function type
                # only used for tanh and sigmoid
                n_fun_params = [random.uniform(p_fun_range[0],p_fun_range[1]),   # mean of Gaussian function
                                random.uniform(0.1,1), # Sigma of Gaussian function
                                p_fun_type] 
                
                # Child node generation function as well as leaf nodes  
                if(p_max_target_attr > 1 and p_current_depth == 0):
                    # For multiple class problem - i.e., target has more than has one column (outputs)
                    # We make sure all the child of roots are a function node and not a leaf node
                    
                    # generating random number of chid for this function node  
                    n_children_range = p_max_children - n_min_child # compute range for randomize number of child for a node
                    n_children = n_min_child + random.randrange(n_children_range) # random children numbner (ensure atleast 2 child)
                    
                    if p_out_fun_type == 'softmax':
                        n_fun_params = [random.uniform(p_fun_range[0],p_fun_range[1]), # unused for softmax
                                    random.uniform(0.1,1.0),   # unused for softmax /tanh/sigmoid
                                    p_out_fun_type]
                        
                    # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
                    n_funNode = FunctionNode(n_weight, n_bias, n_fun_params, n_children, self, p_current_depth) #parent of this node is self                    
                    # recurrrsion
                    n_funNode.genrateChildren(params, p_current_depth + 1)
                    self.m_ChildrenList.append(n_funNode)
                else:
                    # For single class (typicaly regression problem root can function as well as leaf nodes                  
                    # a random choice between next node is leaf or a function
                    # probalistice descision to generate a leaf node or a function node
                    if(random.random() < params.n_int_leaf_rate):
                        # generate a leaf child of self as parent
                        n_num = random.randrange(p_max_input_attr) # random number to determine a leef node or a function  node child
                        n_leafnode =  LeafNode(n_weight, n_num, self)
                        self.m_ChildrenList.append(n_leafnode)
                    else:
                        # generate a function node (child) of the current node (self as a parent)
                        # compute number of children for the new function node
                        n_children_range = p_max_children - n_min_child # compute range for randomize number of child for a node
                        n_children = n_min_child + random.randrange(n_children_range) # random children numbner (ensure atleast 
                        # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
                        n_funNode = FunctionNode(n_weight, n_bias, n_fun_params, n_children, self, p_current_depth) #parent of this node is self
                        
                        # recurrrsion
                        n_funNode.genrateChildren(params, p_current_depth + 1)
                        
                        self.m_ChildrenList.append(n_funNode)
                    # End of if-else for problistic decision
                # End for multi class problem
            #end of for loop for all p_max_children
        else:
            # generate only leaf node when max depth is reached
            # iterate through number of children of this function node
            for i  in range(self.m_children):
                n_weight = random.uniform(p_weight_range[0],p_weight_range[1]) # tree edge weight between uniformaly taken between 0 and 1   
                # this generate random input attr number (p_max_input_attr is exlusive)
                n_input_num = random.randrange(p_max_input_attr) 
                n_leafnode =  LeafNode(n_weight, n_input_num, self) #  leaf node does not contain bias
                self.m_ChildrenList.append(n_leafnode)
            #end for loop
        #ende if-else for depth of the three
    
    #end of defination generateChildren
    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def print_node(self, p_depth):
        '''
            Printing function node childs - or collecting them in a JSON format.
                :param p_depth    current depth of the tree
        '''
        #print("(" + str(p_depth) + "",end=" ")
        for i in range(p_depth):
            if(False): # false becuase it was used just for testing
                print("-",end=" ") # I do print only for testing
        
        #print(" +" + str(self.m_children))
        jason_string = "\"name\":"+ "\"f "+str(self.m_children) +"\","
        
        jason_string = jason_string + "\"children\":["
        
        listLength = len(self.m_ChildrenList)
        conut_loop = 0
        for node in self.m_ChildrenList:            
            #node.print_node(p_depth + 1)
            jason_string = jason_string + "{" + node.print_node(p_depth + 1) 
            
            conut_loop = conut_loop + 1
            if(conut_loop < listLength ):
                jason_string = jason_string   + "," # no comma for the last child in the list.
            
        jason_string = jason_string + " ]" 
            
        return jason_string + "}"
    # end printing function nodes
    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def saveNode(self, p_depth):
        '''
            Saving function node childs - or collecting them in a JSON format.
                :param p_depth    current depth of the tree
        '''
        #print("(" + str(p_depth) + "",end=" ")
        for i in range(p_depth):
            if(False): # false becuase it was used just for testing
                print("-",end=" ") # I do print only for testing
        
        #print(" +" + str(self.m_children))
        jason_string = "\"name\":" + "\"f:"+str(self.m_children) + "; e:" + str(self.m_edge_weight) + "; b:" + str(self.m_bias) +  "; p:" + str(self.m_FunctionParams) + "\","
        
        jason_string = jason_string + "\"children\":["
        
        listLength = len(self.m_ChildrenList)
        conut_loop = 0
        for node in self.m_ChildrenList:            
            #node.print_node(p_depth + 1)
            jason_string = jason_string + "{" + node.saveNode(p_depth + 1) 
            
            conut_loop = conut_loop + 1
            if(conut_loop < listLength ):
                jason_string = jason_string   + "," # no comma for the last child in the list.
            
        jason_string = jason_string + " ]" 
            
        return jason_string + "}"
    # end printing function nodes

    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def retriveChildrenOriginal(self, myjson, p_depth):
        '''
            Retrive children from the json file
        '''
        #print('retriving tree data')
        _NUL = object()  # unique value guaranteed to never be in JSON data
        if isinstance(myjson, dict):
            print('dictionary operation not in use in this case')
            for jsonkey in myjson:
                jsonvalue = myjson[jsonkey]
                print('perhaps not require to use it - is you seee this as print - somthing wrong', jsonvalue)
                
        elif isinstance(myjson, list):
            #print('list operation')
            for item in myjson:
                #iterate over all item (each iteam is a dictionary) in the list of children
                #check if the item is leaf 
                if (len(item) == 1 and (item.get('children', _NUL) is _NUL)):
                    #if dictitionary has 1 iteams and children is _NUL then its a leaf node
                    # generate a leaf child of self as parent
                    index = 0 # 0 is index of attribute
                    edge = 1 # 1 is index of attribute
                    n_num = int(item['name'].split(';')[index].split(':')[1]) 
                    n_weight = float(item['name'].split(';')[edge].split(':')[1]) 
                    n_leafnode =  LeafNode(n_weight, n_num, self)
                    self.m_ChildrenList.append(n_leafnode)
                
                if (len(item) == 2 and (item.get('children', _NUL) is not _NUL)):
                    #generate function node                          
                    n_fun_children = item.get('children') # fetch the list of children
                    n_children = len(item['children']) # this is the length of children list
                    # n_children length can  be also found alternatively  using -> n_list_names position 0 
                    
                    n_list_names = item['name']
                    n_list_names = n_list_names.split(';')
                    
                    # n_list_names - > one is the weight
                    n_edge_weight = float(n_list_names[1].split(':')[1])
                    # n_list_names - > two is the function paramter
                    n_bias_weight = float(n_list_names[2].split(':')[1])
                    # n_list_names - > three is the function paramter
                    n_fun_params = ast.literal_eval(n_list_names[3].split(':')[1])
                    # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
                    n_funNode =  FunctionNode(n_edge_weight, n_bias_weight, n_fun_params, n_children, self, p_depth)
                    n_funNode.retriveChildren(n_fun_children, p_depth + 1)
                    # after all reacursion the child list will be saved here
                    self.m_ChildrenList.append(n_funNode) 

    # ---------------------------------------------------------------------------------------------------------------------------    
    def retriveChildren(self, myjson, p_depth):
        '''
            Retrive children from the json file
        '''
        #print('retriving tree data')
        _NUL = object()  # unique value guaranteed to never be in JSON data
        if isinstance(myjson, dict):
            print('dictionary operation not in use in this case')
            for jsonkey in myjson:
                jsonvalue = myjson[jsonkey]
                print('perhaps not require to use it - is you seee this as print - somthing wrong', jsonvalue)
                
        elif isinstance(myjson, list):
            #print('list operation')
            for item in myjson:
                #iterate over all item (each iteam is a dictionary) in the list of children
                #check if the item is leaf 
                if (len(item) == 1 and (item.get('children', _NUL) is _NUL)):
                    #if dictitionary has 1 iteams and children is _NUL then its a leaf node
                    # generate a leaf child of self as parent
                    index = 0 # 0 is index of attribute
                    edge = 1 # 1 is index of attribute
                    n_num = int(item['name'].split(';')[index].split(':')[1]) 
                    n_weight = float(item['name'].split(';')[edge].split(':')[1]) 
                    n_leafnode =  LeafNode(n_weight, n_num, self)
                    self.m_ChildrenList.append(n_leafnode)
                
                if (len(item) == 2 and (item.get('children', _NUL) is not _NUL)):
                    #generate function node                          
                    n_fun_children = item.get('children') # fetch the list of children
                    n_children = len(item['children']) # this is the length of children list
                    # n_children length can  be also found alternatively  using -> n_list_names position 0 
                    
                    n_list_names = item['name']
                    n_list_names = n_list_names.split(';')
                    
                    # n_list_names - > one is the weight
                    n_edge_weight = float(n_list_names[1].split(':')[1])
                    # n_list_names - > two is the function paramter
                    n_bias_weight = float(n_list_names[2].split(':')[1])
                    # n_list_names - > three is the function paramter
                    
                    strVal = str(n_list_names[3].split(':')[1])+""
                    #print(strVal)
                    n_list_string = strVal
                    n_list_string = n_list_string.replace("[","")
                    n_list_string = n_list_string.replace("]","")
                    n_list_string = n_list_string.replace(" ","").split(',')
                    #n_fun_params = ast.literal_eval(n_list_names[3].split(':')[1])
                    n_fun_params = [float(n_list_string[0]), float(n_list_string[1]), n_list_string[2]]
                    #print(n_fun_params)
                    
                    
                    # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
                    n_funNode =  FunctionNode(n_edge_weight, n_bias_weight, n_fun_params, n_children, self, p_depth)
                    n_funNode.retriveChildren(n_fun_children, p_depth + 1)
                    # after all reacursion the child list will be saved here
                    self.m_ChildrenList.append(n_funNode) 
    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def inspect_node(self, p_tree, p_depth):
        '''
            Inpects current function node -  This function will inspect all node
                :parame   p_tree        tree passed as a paramter
                :param    p_depth       current depth of the tree
            
        '''
        for node in self.m_ChildrenList:
            # if print_message = True it print node type (except for root node) so toal print will one less that toatal nodes
            print_message = False
            if(node.isLeaf(print_message)): 
                # add leaf child
                node.inspect_node(p_tree, p_depth + 1)
                p_tree.addLeafNodesList(node)
            else:
                #add function child
                node.inspect_node(p_tree, p_depth + 1)
                p_tree.addFunNodesList(node)
    # ---------------------------------------------------------------------------------------------------------------------------                
    def getLeafNodesValue(self, out_noeds_inputs_attr, node_count):
        '''
            Retiriving the corrent nodes all input attributes value
            params:
                out_noeds_inputs_attr: is a list
        '''
        
        for node in self.m_ChildrenList:
            node_count.append(1)
            if(node.isLeaf()):
                out_noeds_inputs_attr.append(node.getInputAttribute())
            else:
                node.getLeafNodesValue(out_noeds_inputs_attr, node_count)
        
        return out_noeds_inputs_attr, node_count
           
    # ---------------------------------------------------------------------------------------------------------------------------                
    def copyNode(self, p_nodes):
        '''
            return a copy nodes 
        '''
        # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
        node = FunctionNode(self.m_edge_weight, self.m_bias, self.m_FunctionParams, self.m_children, p_nodes, self.m_depth)
        for nodes in self.m_ChildrenList:
            node.m_ChildrenList.append(nodes.copyNode(node))
            
        return node
    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def setParentNode(self, p_parent_node):
        '''
            setting parent node information
        '''
        self.m_parent_node = p_parent_node
    
    # ---------------------------------------------------------------------------------------------------------------------------        
    def getParentNode(self):
        '''
            Refturn parent node information
        '''
        return self.m_parent_node
    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def getDepth(self):
        '''
            Return current depth of the tree
        '''
        return self.m_depth
    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def replaceSubTree(self, p_old_nodes, p_new_nodes):
        '''
            replace subtree of the parent node
        '''
        if(self.m_ChildrenList.__contains__(p_old_nodes)):
            self.m_ChildrenList[self.m_ChildrenList.index(p_old_nodes)] = p_new_nodes
            #print('subtree is replaced')
        else:
            print('no replacments')
    
    # ---------------------------------------------------------------------------------------------------------------------------            
    def removeAndReplace(self, p_fun_node_to_replace, p_weight, p_input_rand_attr):
        '''
            Remove function node-> p_fun_node 
            And
            Replace by p_fun_node by a leaf node-> p_leaf_node
            param:    p_fun_node_to_replace     a finction node
            param:    p_weight_range            weight of the leaf node
            param:    p_input_rand_attr         random input attr index at the leaf node
        '''
        if(self.m_ChildrenList.__contains__(p_fun_node_to_replace)):
            by_leaf_node = LeafNode(p_weight, p_input_rand_attr, self)
            self.m_ChildrenList[self.m_ChildrenList.index(p_fun_node_to_replace)] = by_leaf_node
            #print('subtree is replaced')
        else:
            print('no replacments')
    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def removeAndGrow(self, p_leaf_node_to_replace, params):
        '''
            Remove leaf node node p_leaf_node_to_replace
            And
            Replace by a subtree
            param:    p_leaf_node_to_replace    node to be replaced 
            param:    prams                     set of input paramters
        '''
        if(self.m_ChildrenList.__contains__(p_leaf_node_to_replace)):
            #selected node is removed from the self child list
            #index_in_m_ChildrenList = self.m_ChildrenList.index(p_leaf_node_to_replace)
            #self.m_ChildrenList.remove(p_leaf_node_to_replace)
            
            p_max_children = params.n_max_children
            p_weight_range = params.n_weight_range
            p_fun_range = params.n_fun_range
            p_fun_type = params.n_fun_type
            
            n_weight = random.uniform(p_weight_range[0],p_weight_range[1]) # tree edge weight between uniformaly taken between 0 and 1                
            n_bias = random.uniform(p_weight_range[0],p_weight_range[1]) # tree bias weight between uniformaly taken between 0 and 1   
            n_min_child = 2 # in any case min number of childrren will be 2 for a node
            # a,b,c -> a and b paramters of the activation fun at the fun node, c is function type
            n_fun_params = [random.uniform(p_fun_range[0],p_fun_range[1]),   # mean of Gaussian 
                            random.uniform(0.1,1.0), # sigma of Gaussian  
                            p_fun_type] 
            
            # generating random number of chid for this function node  
            n_children_range = p_max_children - n_min_child # compute range for randomize number of child for a node
            n_children = n_min_child + random.randrange(n_children_range) # random children numbner (ensure atleast             
            # generate a children of the
            # Function Node = WEIGHT, BIAS, FUN_PARMS, #CHILDREN, PARENT, DEPTH
            n_fun_node = FunctionNode(n_weight, n_bias, n_fun_params, n_children, self, self.m_depth) #parent of this node is self                    
            # recurrrsion
            n_fun_node.genrateChildren(params, self.m_depth + 1)
            # add to the childeren list of the parent node
            self.m_ChildrenList[self.m_ChildrenList.index(p_leaf_node_to_replace)] = n_fun_node
            #self.m_ChildrenList.append(n_fun_node)
        else:
            print('no replacments')

    # ---------------------------------------------------------------------------------------------------------------------------    
    def getEdgeWeight(self):
        '''
            Return edge weight of the node
        '''
        return self.m_edge_weight

    # ---------------------------------------------------------------------------------------------------------------------------    
    def getDeltaEdgeWeight(self):
        '''
            Return edge weight of the node
        '''
        return self.m_delta_weight
    
    # ---------------------------------------------------------------------------------------------------------------------------    
    def setEdgeWeight(self, pEdgeWeight):
        '''
            Setting wedge weight
        '''
        self.m_edge_weight = pEdgeWeight

    # ---------------------------------------------------------------------------------------------------------------------------    
    def getBias(self):
        '''
            Return edge weight of the node
        '''
        return self.m_bias

    # ---------------------------------------------------------------------------------------------------------------------------    
    def getDeltaBias(self):
        '''
            Return edge weight of the node
        '''
        return self.m_delta_bias
        
    # ---------------------------------------------------------------------------------------------------------------------------    
    def setBias(self, pBias):
        '''
            Setting wedge weight
        '''
        self.m_bias = pBias
            
    # ---------------------------------------------------------------------------------------------------------------------------    
    def getMultiNodeOutput(self, p_input_attr_val):
        '''
            Function return tree output 
                either the child nodes of the tree -  multi input single output (MISO) problem
                or the root node of the tree -  mulit input multi output (MIMO) problem
            param:    p_input_attr_val          input example vector 
            param:    p_output_attr_count       target columns number - equivalent to number of classes
        '''
        #MULITI OUTPUT - REGRSSION / CLASSIFICATION Problem for each child node
        #Evaluate the function nodes output for a given inputs data 
        # iterate for all child nodes of this corrent node
        # for the first call it starts with all child nodes of the root.
        # so for calssifciation problem all child node of the root will be taken           
        net_root_child_outputs = [] # list for the 
        for node in self.m_ChildrenList:
            net_root_child_outputs.append(node.getSingleNodeOutput(p_input_attr_val))
        
        #return output for each class (each child of the root)
        return net_root_child_outputs
    
    # ---------------------------------------------------------------------------------------------------------------------------        
    def getSingleNodeOutput(self, p_input_attr_val):
        '''
            Evaluate the function nodes output for a given inputs data 
                param:    p_input_attr_val          input example vector 
            '''
        net_node_weighted_sum_of_inputs = 0.0 # list for the 
        # iterate for all child nodes of this current node
        for node in self.m_ChildrenList:
            #weighted sum of all incomining values of the children
            # Its a recurssion here - 
            # IF the node is function it will call itself "getSingleNodeOutput()" in function_node.py 
            # IF a leaf it will call leaf node "getSingleNodeOutput()" in leaf_node.py
            net_node_weighted_sum_of_inputs = net_node_weighted_sum_of_inputs + node.getEdgeWeight() * node.getSingleNodeOutput(p_input_attr_val)
        
        # compute activation of the current node node based on its activation function
        node_activation = ActivationFunction(self.m_FunctionParams, net_node_weighted_sum_of_inputs, self.m_bias)
        self.m_activation = node_activation.value()
        #print('Activation:', self.m_activation)
        return node_activation.value()
    #END of single node output
    
    # ---------------------------------------------------------------------------------------------------------------------------        
    def setDeltaJ(self, y = [], ypred = [], j = 0, isOutputNode = False):
        '''
            Compuute and sets the nodes delta j velaue that will we used to computer weight change
            args:
                param:  d = y  -        desired output
                param:  o = ypred       predicted output
                prams:  j               index in the case of ouput node
                prams:  isOutputNode    set to false by defualt for hidden node the hidden function nodes takes no arguemnt 
        '''
        if isOutputNode:
            #print('Out ',j,':', y[j], ypred[j], self.m_FunctionParams.__contains__('sigmoid'), end = ' ')
            if (self.m_FunctionParams.__contains__('softmax')): # [ypred[j] -y[j]]*ypred[j]*ypred[j]
                self.m_delta_j = (ypred[j] - y[j])

            if (self.m_FunctionParams.__contains__('sigmoid')): # [ypred[j] -y[j]]*ypred[j]*ypred[j]
                self.m_delta_j = (ypred[j] - y[j]) * ypred[j] * (1.0 - ypred[j])

            if (self.m_FunctionParams.__contains__('tanh')): # -[ypred[j] -y[j]]*ypred[j]*ypred[j]
                self.m_delta_j = (-1.0*(ypred[j] -y[j])) * ypred[j] * ypred[j]

        else:# for all hidden nodes
            hj = self.m_activation #  the nodes own activation 
            delta_k = self.m_parent_node.m_delta_j #  delata of the higher/ paren nodes
            wjk = self.m_edge_weight #  weight leads to the parent node (uper/higher layer)            

            #print('H ', hj, delta_k, wjk, self.m_FunctionParams.__contains__('sigmoid'), end = ' ')
            if (self.m_FunctionParams.__contains__('sigmoid')): #  yj[1.0-yj]*deltak*wk
                self.m_delta_j = hj*(1.0 - hj)*delta_k*wjk

            if (self.m_FunctionParams.__contains__('tanh')): #  -yj*yj*deltak*wk 
                self.m_delta_j = (-1.0*hj*hj)*delta_k*wjk
        
        #print('del_j', self.m_delta_j)
        
        for node in self.m_ChildrenList:
            if(not node.isLeaf()): 
                node.setDeltaJ()
                
                
    def setGradient(self, p_input_attr_val = [], both_w_n_b = True):
        '''
            for function node gradient of weight has its own activation as the input
        '''
        if both_w_n_b:
            # FOR NON OUTPUT Function nodes (for leaf node check in leaf function)
            yi = self.m_activation # activation of the corrent node act as an input for the next nodes (parent node)
            del_j = self.m_parent_node.m_delta_j #  delta_j of parent node backproagate to previous layer
            #print('gradient: ',yi, del_j)
            self.m_delta_weight = del_j * yi
            self.m_delta_bias = self.m_delta_j
        else:
            # FOR OUTPUT function nodes
            self.m_delta_bias = self.m_delta_j
                

    
    
    


           
        


            
            
            
            
            
        
            
            
            
        
    
