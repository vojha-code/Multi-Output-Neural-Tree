'''
Created on Sat Apr 20 12:42:42 2019

Training steps of neural tree
Author: v ojha
Affiliation: Uni of Reading
pip install anytree
'''
from src.tree.neural_tree import NeuralTree
import os
import json
from anytree.importer import DictImporter
#from anytree.importer import JsonImporter
#from anytree.exporter import DotExporter
from anytree import RenderTree

class DisplayTree:
    '''
    Class for desplaying tree structuer
    '''

    def __inti__(self):
        print('Printing Tree')

    #----------------------------------------------------------------------------------------
    def displayTree(self,p_tree):
        '''
            This displays a tree
        '''
        # Its dictionary of the tree -  JSON string
        json_tree = json.loads(p_tree.print_tree())
        importer = DictImporter()
        root = importer.import_(json_tree)
        print(RenderTree(root))
        #DotExporter(root).to_dotfile('tree.dot')
        # not effective for same node names (for distin g node name it can be use for tree)
        #DotExporter(root).to_picture('tree.png')

    #----------------------------------------------------------------------------------------
    def displayTreeFile(self, json_file = 'model/treeOutline_trial.json'):
        '''
            This displays a tree
        '''
        # Its dictionary of the tree -  JSON string
        json_tree = self.readTree(json_file)
        importer = DictImporter()
        root = importer.import_(json_tree)
        print(RenderTree(root))

    #----------------------------------------------------------------------------------------
    def retriveTreeFromFile(self,  json_file = 'model/treeModel_trial.json'):
        '''
            This display a tree
        '''
        # Its dictionary of the tree -  JSON string
        json_tree = self.readTree(json_file)
        n_tree =  NeuralTree()
        n_tree.retrive_JSON_Tree(json_tree) # a funciton in NeuralTree Class
        return n_tree

    #----------------------------------------------------------------------------------------
    def readTree(self, filename):
        '''
            This function saves tree to current directory
            param:    p_tree        a tree
            param:    filename      filename string

        '''
        if filename:
            with open(filename, 'r') as f:
                return json.load(f)
        else:
            print('Wrong filename string')
            
    #----------------------------------------------------------------------------------------
    def saveTreeOutline(self, p_tree,  directory = 'model', uniquefileName = 'trial', outlineJson = True):
        '''
            This function saves tree to current directory
            param:    p_tree    a tree to save

        '''
        filePath = os.path.join(directory, 'treeOutline_'+ uniquefileName + '.json')

        json_tree = json.loads(p_tree.print_tree())
        if outlineJson:
            with open(filePath, 'w') as f:
                json.dump(json_tree,f) # this can be used for d3js
        
        # creating a tree data for d3js visualization
        self.replaceTreeView(json_tree, directory, uniquefileName)

    #----------------------------------------------------------------------------------------
    def replaceTreeView(self, json_tree, directory = 'model', uniquefileName = 'trial'):
        '''
            creats a tree data for d3js visualization
            param:
                json_tree format of a tree to save as html and d3js
                directory adressss
        '''
        oldTreeData = 'var treeData'
        newTreeData = 'var treeData = [' + json.dumps(json_tree) + '];'

        filename_org = os.path.join(os.path.normpath(directory + os.sep + os.pardir), 'view')
        #filename_org = os.path.join(directory, 'view')
        filename_org = os.path.join(filename_org, 'tree_view_org.html')
        filename_target = os.path.join(directory, 'tree_view_' + uniquefileName +'.html')

        fin = open(filename_org, "r")
        fout = open(filename_target, "w")

        for num, line in enumerate(fin, 1):
            if oldTreeData in line:
                print('Model Oultiin Saved - Check replaced HTML line: 41? = line replaced is', num)
                fout.write(newTreeData)
            else:
                fout.write(line)
        fin.close()
        fout.close()

    #----------------------------------------------------------------------------------------
    def saveTreeModel(self, p_tree, directory = 'model', uniquefileName = 'trial'):
        '''
            This function saves tree to current directory
            param:
                p_tree    a tree to save
                directory adressss
        '''
        filePath = os.path.join(directory, 'treeModel_'+ uniquefileName + '.json')

        json_tree = json.loads(p_tree.saveTree())
        with open(filePath, 'w') as f:
            json.dump(json_tree,f) # this can be used for d3js

#%%
#importer = JsonImporter()
#root = importer.import_(json.dumps(json_tree))
#print(RenderTree(root))
#DotExporter(root).to_dotfile('tree.dot')
##DotExporter(root).to_picture('tree.png') # not effective for same node names (for distin g node name it can be use for tree)

#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#from graphviz import Source
#path = 'tree.dot'
#s = Source.from_file(path)
#s.view()
