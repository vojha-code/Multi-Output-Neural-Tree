'''
Created on Sat Apr 20 12:42:42 2019

Training steps of neural tree
Author: v ojha
Affiliation: Uni of Reading

'''
import json
from anytree.importer import DictImporter
from anytree import RenderTree
import random
import copy

class GeneticOperator():
    
    mParams = None
    
    #----------------------------------------------------------------------------------------------------------------
    def __init__(self, pParams):
        '''
            Contractor of the genetic operator
        '''
        self.mParams = pParams
    
    #----------------------------------------------------------------------------------------------------------------
    # one point Crossover 
    def crossoverTree(self, p_tree_p1, p_tree_p2, Check_crossover = False):
        '''
           Crossover operation of two parent tree 
           param:    p_tree_p1          first parent tree
           param:    p_tree_p2          second parent tree
           param:    check_crossove     print masages of corssove
        '''
        first_Tree = copy.deepcopy(p_tree_p1)
        second_Tree = copy.deepcopy(p_tree_p2)
        xoverTrue = False
        max_target_attr = self.mParams.n_max_target_attr
        first_funNodes_count = len(first_Tree.getFunNodesList())
        second_funNodes_count = len(second_Tree.getFunNodesList())
        if(max_target_attr == 1 and (first_funNodes_count == 0  or second_funNodes_count == 0)) or (max_target_attr > 1 and (first_funNodes_count <= max_target_attr  or second_funNodes_count <= max_target_attr)):
            if(Check_crossover):
                print('Tree inequality', first_funNodes_count, second_funNodes_count, self.mParams.n_max_target_attr)
            return first_Tree, second_Tree, False
        else:
            if(Check_crossover):
                print('\nSuffcient nodes for crossover:', first_funNodes_count, 'and ', second_funNodes_count)
            # a ranong seslion of subtree, i.e., a function node
            first_Subtree_Index = random.randrange(first_funNodes_count)
            second_Subtree_Index = random.randrange(second_funNodes_count)
            if(Check_crossover):
                print('Indices',first_Subtree_Index, 'and', second_Subtree_Index)
            
            # retriving a function node means all the nodes under the object of this function node will be copyied
            first_SubTree = first_Tree.getFunNodesList()[first_Subtree_Index]
            second_SubTree = second_Tree.getFunNodesList()[second_Subtree_Index]
            if(Check_crossover):
                print('First subtrre in the list: ',first_SubTree in first_Tree.getFunNodesList())
                print('Second subtrre in the list: ',second_SubTree in second_Tree.getFunNodesList())
            
            # selecting parents of two subtree to rplace thir child
            first_SubTree_Parent = first_SubTree.getParentNode()   
            second_SubTree_Parent = second_SubTree.getParentNode()
            if(Check_crossover):
                if(first_SubTree_Parent in first_Tree.getFunNodesList()):
                    print('First Subtre: Parent is among the children of the tree')
                else:
                    if(first_SubTree_Parent == first_Tree.m_root):
                        print('First Subtre: Parent is the root')
            
                if(second_SubTree_Parent in second_Tree.getFunNodesList()):
                    print('Second Subtre: Parent is among the children of the tree')
                else:
                    if(second_SubTree_Parent == second_Tree.m_root):
                        print('Second Subtre: Parent is the root')
                        
            # replcae the subtrees old - new 
            first_SubTree_Parent.replaceSubTree(first_SubTree,second_SubTree)
            second_SubTree_Parent.replaceSubTree(second_SubTree,first_SubTree)
            
            # reset the parents of the subtrees
            first_SubTree.setParentNode(second_SubTree_Parent)
            second_SubTree.setParentNode(first_SubTree_Parent)
            
            # inspect the tree to fill the function nodes
            first_Tree.inspectChildNodes()
            second_Tree.inspectChildNodes()
            
            if (p_tree_p1.getTreeSize() + p_tree_p2.getTreeSize() ==  first_Tree.getTreeSize() + second_Tree.getTreeSize()):
                #print('All good with crossover')
                xoverTrue = True
            else:
                print('Crossover operation needs varification')
            # End els-if - check crossover tree operation print
        
            return  first_Tree, second_Tree, xoverTrue
        # End if-else corrover operation
    #End corover defination

    #----------------------------------------------------------------------------------------------------------------
    def verifyCrossover(self, n_tree_1, n_tree_2, first_Tree, second_Tree, is_crossover_done):
        if(is_crossover_done):
            print('\nTrees before crossover:')
            json_tree = json.loads(n_tree_1.print_tree())
            print("The tree 1 (copy of priginal) structure for ", n_tree_1.getTreeSize(),' nodes is:\n')
            importer = DictImporter()
            root = importer.import_(json_tree)
            print(RenderTree(root))
            
            # We have two trees -  n_tree_1 and n_tree_2
            json_tree = json.loads(n_tree_2.print_tree())
            print("\nThe tree 2 structure for ", n_tree_2.getTreeSize(),' nodes is:\n')
            importer = DictImporter()
            root = importer.import_(json_tree)
            print(RenderTree(root))
            
            print('Trees after crossover:')
            
            json_tree = json.loads(first_Tree.print_tree())
            print("The copied tree structure for ", first_Tree.getTreeSize(),' nodes is:\n')
            importer = DictImporter()
            root = importer.import_(json_tree)
            print(RenderTree(root))
            
            json_tree = json.loads(second_Tree.print_tree())
            print("The copied tree structure for ", second_Tree.getTreeSize(),' nodes is:\n')
            importer = DictImporter()
            root = importer.import_(json_tree)
            print(RenderTree(root))
        else:
            print('Crossover: Operation was NOT perfromed')
        
        if (n_tree_1.getTreeSize() + n_tree_2.getTreeSize() ==  first_Tree.getTreeSize() + second_Tree.getTreeSize()):
            print('\nCrossover: Operation was perfromed')
        else:
            print('\nCrossover operation needs varification')
            
            

    #----------------------------------------------------------------------------------------------------------------            
    def mutation(self, p_tree):
        '''
            Randomly selecting a mutation type over 
            param:    p_tree    a tree for mutation
        '''
        tree = copy.deepcopy(p_tree)
        
        # generating a random number to randomly chooase a mutation type
        rand_num = random.choice([0,1,2,3])
        if(rand_num == 0):
            return self.mutationOneLeaf(tree)            
        if(rand_num == 1):
            return self.mutationAllLeaf(tree)
        if(rand_num == 2):
            return self.mutatationSubTree(tree)
        if(rand_num == 3):
            return self.mutationGrowSubTree(tree)
    # End mutation
    
    #----------------------------------------------------------------------------------------------------------------    
    def mutationOneLeaf(self, tree):
        '''
            Mutate ONLY one randon leaf in the tree
        '''
        #print('mutatate: one leaf')
        #Select a leaf node in the tree to  be replaced ny a random attr
        index_rand_leaf = random.randrange(len(tree.getLeafNodesList()))
        # Fetching that (randentify) node object from the tree leaf node list
        node = tree.getLeafNodesList()[index_rand_leaf]
        #Generating a randon attribute index
        input_rand_attr = random.randrange(0,self.mParams.n_max_input_attr,1)
        # Setting /replaceing the old index by the new index
        node.setInputAttribute(input_rand_attr)
        #print('At node ', index_rand_leaf, ' new attr is: ',input_rand_attr)
        return tree # returning mutated tree
    
    #----------------------------------------------------------------------------------------------------------------
    def mutationAllLeaf(self, tree):
        '''
            Muate all leaf node of the tree
        '''
        #print('mutatate: all leaves')
        #Iterating over all leaf node in the tree
        for node in tree.getLeafNodesList():
            #Generating a random attribute for the leaf node
            input_rand_attr = random.randrange(0,self.mParams.n_max_input_attr,1)
            #replacing the old leaf node by new leaf node
            node.setInputAttribute(input_rand_attr)
        
        return tree # returning mutated tree

    #----------------------------------------------------------------------------------------------------------------        
    def mutatationSubTree(self, tree):
        '''
            Mutate a subtree
        '''
        #print('mutatate: delete a subtree')
        # get the length of the function node - total subtrees in the tree
        n_funNodes_count = len(tree.getFunNodesList())
        if n_funNodes_count < self.mParams.n_max_target_attr:
            return tree
        #Else
        #select a subtree from the tree to prune
        subtree_Index = random.randrange(n_funNodes_count)
        # retriving a function node means all the nodes under the object of this function node will be copyied
        subTree = tree.getFunNodesList()[subtree_Index]
        # Do not prune root's child
        if (self.mParams.n_max_target_attr > 1 and tree.m_root.m_ChildrenList.__contains__(subTree)):
            return tree # returning unmutated tree
        #Else
        #Fetch the parent of the selected subtree
        subTree_parent_node = subTree.getParentNode()
        # tree edge weight between uniformaly taken between 0 and 1                
        n_weight = random.uniform(self.mParams.n_weight_range[0],self.mParams.n_weight_range[1])
        # generate a randon input attr
        n_input_rand_attr = random.randrange(self.mParams.n_max_input_attr)
        #Replace finctionnide by a leaf node
        subTree_parent_node.removeAndReplace(subTree, n_weight, n_input_rand_attr)
        tree.inspectChildNodes()
        #print('The function node', subtree_Index , 'was replaced by a leaf node', n_funNodes_count)
        return tree # returning mutated tree
        
    #----------------------------------------------------------------------------------------------------------------
    def mutationGrowSubTree(self, tree):
        '''
            Remove a leaf node and grow a subtree
        '''

        #print('mutatate: grow a leaf node')
        #Select a leaf node in the tree to  be replaced ny a random attr
        index_rand_leaf = random.randrange(len(tree.getLeafNodesList()))
        # Fetching that (randentify) node object from the tree leaf node list
        leaf_node = tree.getLeafNodesList()[index_rand_leaf]
        #Fetch the parenm node of the selected leaf node
        parent_node = leaf_node.getParentNode()
        parent_node.removeAndGrow(leaf_node, self.mParams)
        tree.inspectChildNodes()
        return tree

        
        