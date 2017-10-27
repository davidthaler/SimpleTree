'''
Bare-bones wrapper to get simple tree to be sklearn compatible.
All calls are fed through to functions in simple_tree_builder.

author: David Thaler
date: August 2017
'''
from . import simple_tree_builder

class SimpleTree():
    '''
    SimpleTree is a simple sklearn-compatible class built over the functions
    build_tree, apply, predict and predict_proba.
    '''

    def __init__(self, max_depth=None, min_samples_leaf=1):
        '''
        A simple decision tree.

        Args:
            max_depth: maximum depth of tree, of None (default) for no limit
            min_samples_leaf: minimum number of samples in a leaf;
                default 1, must be >= 1
        '''
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def __repr__(self):
        '''
        Repr method of SimpleTree gives name and constructor params.

        Returns:
            String representation of self
        '''
        return ('SimpleTree(max_depth=%s, min_samples_leaf=%s)' % 
                    (self.max_depth, self.min_samples_leaf))

    def fit(self, x, y):
        '''
        Fits this tree to with the provided training data

        Args:
            x: m x n numpy array of numeric features
            y: length m 1-D vector of 0/1 labels

        Returns:
            self; also fits the estimator
        '''
        max_depth = -1 if self.max_depth is None else self.max_depth
        self.tree_ = simple_tree_builder.build_tree(x, y, 
                            self.min_samples_leaf, max_depth)
        return self


    def apply(self, x):
        '''
        Finds the node number of the leaf each instance in x lands in

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype int) of leaf node numbers for each point in x
        '''
        return simple_tree_builder.apply(self.tree_, x)


    def predict(self, x):
        '''
        Makes 0/1 predictions for the data x.

        NB: predicts p=0.5 as False

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype float) of 0.0 and 1.0 for the two classes
        '''
        return simple_tree_builder.predict(self.tree_, x)


    def predict_proba(self, x):
        '''
        Predicts the probability of class 1 membership for each row in x

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype float) of probabilities of class 1 membership
        '''
        return simple_tree_builder.predict_proba(self.tree_, x)
