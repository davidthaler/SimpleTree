'''
SimpleTree implements a base class for decision tree models.
ClassificationTree is a single-output binary classifier for 0-1 labels
that minimizes Gini impurity.
RegressionTree if a single-output regression model that minimizes mean
squared error or, equivalently, residual sum of squares.

author: David Thaler
date: August 2017
'''
from . import simple_tree_builder
from . import simple_splitter


class SimpleTree():
    '''
    SimpleTree is a simple sklearn-compatible class built over the functions
    build_tree, apply, predict and predict_proba.
    '''

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
        self.tree_ = simple_tree_builder.build_tree(x, y, self.split_fn,
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

    @property
    def values(self):
        '''
        Node values of this tree.

        Returns:
            the values for the nodes in this tree
        '''
        return simple_tree_builder.values(self.tree_)

    def decision_function(self, x):
        '''
        Returns a decision value for each point in x.
        For classification tasks, this is a probablility of class 1 membership.
        For regression tasks, it is a prediction.

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype float) of decision function values.
        '''
        leaf_idx = self.apply(x)
        return self.values[leaf_idx]


class ClassificationTree(SimpleTree):
    '''
    ClassificationTree implements a decision tree with a gini impurity
    split criterion.
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
        self.split_fn = simple_splitter.gini_split
    
    def predict(self, x):
        '''
        Makes 0/1 predictions for the data x.

        NB: predicts p=0.5 as False

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype int) of 0 and 1 for the two classes
        '''
        return (self.predict_proba(x) > 0.5).astype(int)

    def predict_proba(self, x):
        '''
        Predicts the probability of class 1 membership for each row in x

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype float) of probabilities of class 1 membership
        '''
        return self.decision_function(x)

    def __repr__(self):
        '''
        Repr method of ClassificationTree gives name and constructor params.

        Returns:
            String representation of self
        '''
        return ('ClassificationTree(max_depth=%s, min_samples_leaf=%s)' % 
                    (self.max_depth, self.min_samples_leaf))


class RegressionTree(SimpleTree):
    '''
    RegressionTree implements a decision tree regressor that minimizes
    residual sum of squares or, equivalently, mse.
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
        self.split_fn = simple_splitter.mse_split

    def predict(self, x):
        '''
        Predicts the regression target values, given input data.

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype float) of estimates
        '''
        return self.decision_function(x)

    def __repr__(self):
        '''
        Repr method of RegressionTree gives name and constructor params.

        Returns:
            String representation of self
        '''
        return ('RegressionTree(max_depth=%s, min_samples_leaf=%s)' % 
                    (self.max_depth, self.min_samples_leaf))

    