'''
SimpleTree implements a base class for decision tree models.
ClassificationTree is a single-output binary classifier for 0-1 labels
that minimizes Gini impurity.
RegressionTree if a single-output regression model that minimizes mean
squared error or, equivalently, residual sum of squares.

author: David Thaler
date: August 2017
'''
import numpy as np
from . import simple_splitter

'''
The tree is represented internally as an array with several data fields.
Column definitions:
    0) Split feature, -1 if leaf
    1) Split threshold
    2) Node number of this node (nodes are numbered in pre-order).
    3) Node number of left child, -1 if leaf
    4) Node number of right child, -1 if leaf
    5) Number of data points in this node
    6) Value of this node (mean label)
'''
FEATURE_COL = 0
THR_COL = 1
NODE_NUM_COL = 2
CHILD_LEFT_COL = 3
CHILD_RIGHT_COL = 4
CT_COL = 5
VAL_COL = 6


def build_tree(x, y, split_fn, min_samples_leaf, max_features,
                    depth_limit=-1, node_num=0):
    '''
    Recursively build a decision tree. 
    Returns a 2-D array of shape (num_nodes x 7) that describes the tree.
    Each row represents a node in pre-order (root, left, right).
    See the module comment for the column definitions.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.
        split_fn: one of simple_splitter.gini_split/mse_split or similar
        min_samples_leaf: minimum number of samples in a leaf, must be >= 1
        max_features: (int) max number of features to try per split
        depth_limit: maximum depth of tree/subtree below this node
            returns when =0; default of -1 yields no depth limit
        node_num: the node number of this node
            default 0 is for the root node

    Returns:
        2-D numpy array of dtype 'float' with data specifying the tree
    '''
    ct = len(y)
    val = y.sum() / ct
    if depth_limit == 0:
        return np.array([[-1, 0.0, node_num, -1, -1, ct, val]])
    col_idx = np.random.choice(x.shape[1], size=max_features, replace=False)
    subfeature, thr, _, _ = split_fn(x[:, col_idx], y, min_samples_leaf)
    if subfeature == -1:
        return np.array([[-1, 0.0, node_num, -1, -1, ct, val]])
    # NB: subfeature is relative to x[:, col_idx], must change back
    feature = col_idx[subfeature]
    mask = x[:, feature] <= thr
    left_root = node_num + 1
    left_tree = build_tree(x[mask], y[mask], split_fn, min_samples_leaf, 
                                max_features, depth_limit - 1, left_root)
    right_root = left_root + len(left_tree)
    right_tree = build_tree(x[~mask], y[~mask], split_fn, min_samples_leaf, 
                                max_features, depth_limit - 1, right_root)
    root = np.array([[feature, thr, node_num, left_root, right_root, ct, val]])
    return np.concatenate([root, left_tree, right_tree])


class SimpleTree():
    '''
    SimpleTree is a base class for sklearn-compatible decision tree estimators.
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
        max_features = (x.shape[1] if self.max_features is None
                            else self.max_features)
        self.tree_ = build_tree(x, y, self.split_fn, self.min_samples_leaf,
                                max_features, max_depth)
        return self

    def apply(self, x):
        '''
        Finds the node number of the leaf each instance in x lands in

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype int) of leaf node numbers for each point in x
        '''
        n = len(x)
        node = np.zeros(n, dtype=int)
        active = np.ones(n).astype(bool)
        while active.any():
            active = (self.tree_[node, CHILD_LEFT_COL] != -1)
            xa = x[active]
            na = node[active]
            cfeat = (self.tree_[na, FEATURE_COL]).astype(int)
            cx = xa[np.arange(len(xa)), cfeat]
            cthr = self.tree_[na, THR_COL]
            cleft = (self.tree_[na, CHILD_LEFT_COL]).astype(int)
            cright = (self.tree_[na, CHILD_RIGHT_COL]).astype(int)
            cnode = np.where(cx <= cthr, cleft, cright)
            node[active] = cnode
        return node

    @property
    def values(self):
        '''
        Node values of this tree.

        Returns:
            the values for the nodes in this tree
        '''
        return self.tree_[:, VAL_COL]

    @values.setter
    def values(self, vals):
        '''
        Update the leaf node values for this tree

        Args:
            vals: the new leaf node values
        '''
        self.tree_[:, VAL_COL] = vals

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

    def __repr__(self):
        '''
        Repr method of SimpleTree gives class name and constructor params.

        Returns:
            String representation of self
        '''
        name = self.__class__.__name__
        return ('%s(min_samples_leaf=%s, max_features=%s, max_depth=%s)' %
                    (name, self.min_samples_leaf, self.max_features,
                    self.max_depth))


class ClassificationTree(SimpleTree):
    '''
    ClassificationTree implements a decision tree with a gini impurity
    split criterion. It does single-output, binary classification and
    requires 0-1 labels.
    '''

    def __init__(self, min_samples_leaf=1, max_features=None, max_depth=None):
        '''
        A simple decision tree classifier using gini impurity as the
        split criterion.

        Args:
            min_samples_leaf: minimum number of samples in a leaf;
                default 1, must be >= 1
            max_features: max number of features to try per split;
                default of None for all features
            max_depth: maximum depth of tree, of None (default) for no limit
        '''
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
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


class RegressionTree(SimpleTree):
    '''
    RegressionTree implements a decision tree regressor that minimizes
    residual sum of squares or, equivalently, mse.
    '''

    def __init__(self, min_samples_leaf=1, max_features=None, max_depth=None):
        '''
        A decision tree regressor that minimizes residual sum of squares.

        Args:
            min_samples_leaf: minimum number of samples in a leaf;
                default 1, must be >= 1
            max_features: max number of features to try per split;
                default of None for all features
            max_depth: maximum depth of tree, of None (default) for no limit
        '''
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
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



    