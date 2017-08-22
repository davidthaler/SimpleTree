import numpy as np
from gini_splitter import split
import numba

# Position constants for the fields in the tree
FEATURE_COL = 0
THR_COL = 1
CT_COL = 2
POS_COL = 3
NODE_NUM_COL = 4
CHILD_LEFT_COL = 5
CHILD_RIGHT_COL = 6


def build_tree(x, y, node_num=0):
    '''
    Recursively build a decision tree. 
    Returns a 2-D array that describes the tree of dimension num_nodes x 7.
    Each row represents a node in pre-order (root, left, right).
    Columns:
    0) Split feature, -1 if leaf
    1) Split threshold
    2) Number of data points in this node
    3) Number of positives in this node
    4) Node number of this node (nodes are numbered in pre-order).
    5) Node number of left child, -1 if leaf
    6) Node number of right child, -1 if leaf

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.
        node_num: the node number of this node
            default 0 is for the root node

    Returns:
        2-D numpy array of dtype 'float' with 
    '''
    ct = len(y)
    pos = y.sum()
    feature, thr, _ = split(x, y)
    if feature == -1.0:
        return np.array([[feature, thr, ct, pos, node_num, -1, -1]])
    mask = x[:, feature] <= thr
    left_root = node_num + 1
    left_tree = build_tree(x[mask], y[mask], left_root)
    right_root = left_root + len(left_tree)
    right_tree = build_tree(x[~mask], y[~mask], right_root)
    root = np.array([[feature, thr, ct, pos, node_num, left_root, right_root]])
    return np.concatenate([root, left_tree, right_tree])


@numba.jit
def apply(tree, x):
    '''
    Finds the node number in the provided tree (from build_tree) that each
    instance in x lands in.

    NB: using numba.jit w/o nopython=True option allows astype(int) at end

    Args:
        tree: the array returned by build_tree
        x: m x n numpy array of numeric features

    Returns:
        1-D numpy array (dtype int) of leaf node numbers for each point in x.
    '''
    out = np.zeros(len(x))
    for k in range(len(x)):
        node = 0                                    # the root
        while tree[node, FEATURE_COL] >= 0:         # not a leaf
            feature_num = int(tree[node, FEATURE_COL])
            thr = tree[node, THR_COL]
            if x[k, feature_num] <= thr:
                node = int(tree[node, CHILD_LEFT_COL])
            else:
                node = int(tree[node, CHILD_RIGHT_COL])
        out[k] = node
    return out.astype(int)


def predict_proba(tree, x):
    '''
    Predicts the probability of class 1 membership for each row in x
    using the provided tree from build_tree.

    Args:
        tree: the array returned by build_tree
        x: m x n numpy array of numeric features

    Returns:
        1-D numpy array (dtype float) of probabilities of class 1 membership.
    '''
    leaf_idx = apply(tree, x)
    tot = tree[leaf_idx, CT_COL]
    pos = tree[leaf_idx, POS_COL]
    return pos / tot


def predict(tree, x):
    '''
    Makes 0/1 predictions for the data x using the provided tree
    from build_tree.

    NB: predicts p=0.5 as False

    Args:
        tree: the array returned by build_tree
        x: m x n numpy array of numeric features

    Returns:
        1-D numpy array (dtype float) of 0.0 and 1.0 for the two classes.
    '''
    return (predict_proba(tree, x) > 0.5).astype(int)


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
        self.n_features_ = x.shape[1]
        self.tree_ = build_tree(x, y)
        return self


    def apply(self, x):
        '''
        Finds the node number of the leaf each instance in x lands in

        NB: using numba.jit w/o nopython=True option allows astype(int) at end

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype int) of leaf node numbers for each point in x
        '''
        return apply(self.tree_, x)


    def predict(self, x):
        '''
        Makes 0/1 predictions for the data x.

        NB: predicts p=0.5 as False

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype float) of 0.0 and 1.0 for the two classes
        '''
        return predict(self.tree_, x)


    def predict_proba(self, x):
        '''
        Predicts the probability of class 1 membership for each row in x

        Args:
            x: m x n numpy array of numeric features

        Returns:
            1-D numpy array (dtype float) of probabilities of class 1 membership
        '''
        return predict_proba(self.tree_, x)
