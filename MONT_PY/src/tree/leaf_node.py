'''
Generating random generic tree
Author: v ojha
Affiliation: Uni of Reading

'''

from src.tree.node import Node

class LeafNode (Node):
    m_input_attr = -99 # initialize arbitarly to be set by function/constrcutor
    
    #constructor for Leaf node
    def __init__(self, p_weight, p_input_attr, p_parent_node):
        '''
            Genrating a leaf node through its constructor
                :parm    m_weight           weigth of the eadge connecting leaf node
                :parm    m_input            the input attribute to be considered at the leaf node
                :parm    m_parent_node      parent node of the leaf node
        
        '''
        self.m_edge_weight = p_weight
        self.m_input_attr = p_input_attr;
        self.m_parent_node = p_parent_node
    
    
    def isLeaf(self, p_toPrintNodeType = False):
        if(p_toPrintNodeType):
            print('Leaf node')
        return True
    
        
    def print_node(self, p_depth):
        '''
            Printing only leaf nodes (returnining leaf node string)
                :param p_depth    current depth of the tree
        '''
        for i in range(p_depth):
            if(False):
                print("-",end=" ")
        
        #print(" :" + str(self.m_input_attr))
        return "\"name\":"+ "\" i "+ str(self.m_input_attr) +"\" }"
    
    def saveNode(self, p_depth):
        '''
            Printing only leaf nodes (returnining leaf node string)
                :param p_depth    current depth of the tree
        '''
        for i in range(p_depth):
            if(False):
                print("-",end=" ")
        
        #print(" :" + str(self.m_input_attr))
        return "\"name\":"+ "\" i:"+ str(self.m_input_attr) +  "; e:" +  str(self.m_edge_weight) + "\" }"
    
    def inspect_node(self, p_tree, p_depth):
        '''
            Inpects current function node
                :parame   p_tree        tree passed as a paramter
                :param    p_depth       current depth of the tree
            
        '''
        if(p_depth > p_tree.getDepth()):
            p_tree.setDepth(p_depth)
    # end inspect node (leaf)    
    
    def copyNode(self, p_node):
        '''
            return leaf node copy with its parent node being p_node
        '''
        #new object of leaf node
        node = LeafNode(self.m_edge_weight, self.m_input_attr, p_node)
        return node
    
    def setInputAttribute(self, p_input_attr):
        '''
            Replacing previous input attrivute number wth p_input_attr
        '''
        self.m_input_attr = p_input_attr
    
    def getInputAttribute(self):
        '''
            Returning the leaf node input attr index
        '''
        return self.m_input_attr
    
    
    def getParentNode(self):
        '''
            Refturn parent node information
        '''
        return self.m_parent_node

    def getEdgeWeight(self):
        '''
            Return edge weight of the node
        '''
        return self.m_edge_weight    

    def getDeltaEdgeWeight(self):
        '''
            Return edge weight of the node
        '''
        return self.m_delta_weight

    def setEdgeWeight(self, pEdgeWeight):
        '''
            Setting wedge weight
        '''
        self.m_edge_weight = pEdgeWeight    
        
    def getSingleNodeOutput(self, p_input_attr_val):
        '''
            Return value of the value at the index m_input_attr from the leaf node 
            param:    p_input_attr_val    input vector 
        '''
        return p_input_attr_val[self.m_input_attr] 

    def setGradient(self, p_input_attr_val = [], both_w_n_b = True):
        '''
            for Leaf node node gradient of weight has its own input value as the input
        '''
        xi = p_input_attr_val[self.m_input_attr] #  inut to the next layer
        del_j = self.m_parent_node.m_delta_j #  delta_j of the next layer (gradient of the parent node)
        self.m_delta_weight =  del_j *  xi
