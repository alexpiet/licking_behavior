import numpy as np
import pandas as pd
from copy import copy


class HData():
    def __init__(self, df, hierarchy):
        '''
        df: Dataframe containing the data and all grouping columns contained in hierarchy
        hierarchy: List of the hiererarchy levels. For a dataset where you have flashes organized in cells
                   inside sessions this would be something like ['root', session_id', 'cell_id']
        '''
        self.df = df
        self.df['iloc'] = np.arange(len(self.df)) #Append a column with the index of each row
        self.hierarchy = hierarchy
        self.root = Node(None, 
                         np.arange(len(self.df)),
                         self,
                         level=0,
                         ontology=[])
class Node():
    def __init__(self, unique_val, inds, h_obj, level, ontology):
        '''
        inds: The inds that define the group in the dataframe
        h_obj: The top-level object that contains the tree and the dataframe
        level: What level this node is in the hierarchy (at h_obj.hierarchy)
        ontology: A list of the parents for this node
        '''
        self.inds = inds
        self.h_obj = h_obj
        self.level = level
        self.label = self.h_obj.hierarchy[level]
        self.ontology = ontology
        self.children = []
        if self.level < len(self.h_obj.hierarchy)-1:
            self.make_children()
    def make_children(self):
        df_self = self.h_obj.df.iloc[self.inds]
        next_label = self.h_obj.hierarchy[self.level+1]
        unique_vals = df_self[next_label].unique()
        child_ontology = self.ontology + [self]
        for unique_val in unique_vals:
            query_string = '{} == {}'.format(next_label, unique_val)
            inds_this_child = df_self.query(query_string)['iloc'].values
            self.children.append(Node(
                unique_val,
                inds_this_child,
                self.h_obj,
                self.level+1,
                child_ontology
            ))


def resample_recursive(start_node, inds_list):
    if start_node.level < len(start_node.h_obj.hierarchy)-1:
        n_children = len(start_node.children)
        child_inds_to_sample = np.random.choice(np.arange(n_children), replace=True, size=n_children)
        for ind_child in child_inds_to_sample:
            resample_recursive(start_node.children[ind_child], inds_list)
    else:
        inds_list.extend(np.random.choice(start_node.inds, replace=True, size=len(start_node.inds)))









