'''
Generating random generic tree
Author: v ojha
Affiliation Uni of Reading

'''

class Node:
    m_edge_weight = None #edge weight of the node initialize to randaom value
    m_delta_weight = None # gradient of the node
    m_activation = None # output of this node
    m_parent_node = None # the parent node of the node
    
    
    def __init__(self, p_weight, p_parent_node):
        '''
        '''        
        self.m_edge_weight = p_weight
        self.m_parent_node = p_parent_node
        
    def print_node(self, p_depth):
        print('Overide this methods')
    
    def saveNode(self, p_depth):
        print('Overide this methods')
        
    def inspect_node(self, p_tree, p_depth):
        print('Overide inspect_node methods')
        
    def isLeaf(self, p_toPrintNodeType = False):
        if(p_toPrintNodeType):
            print('Function node')
        return False
    
    def copyNode(self, p_node):
        '''
            Copy node taks one paramter
        '''
        print('Overide copyNode methods')
    
    def setParentNode(self, p_parent_node):
        '''
            setting parent node information
        '''
        self.m_parent_node = p_parent_node
        
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
            set wedge weight
        '''
        self.m_edge_weight = pEdgeWeight

        
    def getSingleNodeOutput(self, p_input_attr_val):
        '''
            Output of a sngle node 
        '''
        print('Override defult - refgression output method')
        return -999.000 # intentinally given a rubish value becuase it will override
    
    def getMultiNodeOutput(self, p_input_attr_val):
        '''
            Output of a sngle node 
        '''
        print('Override defult - refgression output method')
        return [] # intentinally given a rubish value becuase it will override
    
    def setGradient(self, p_input_attr_val = [], both_w_n_b = True):
        '''
            Seting weight change
        '''
        self.m_delta_weight = -999.000
    

        
        
    