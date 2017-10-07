'''
These functions build a decision tree, which is output as a table 
contained in a numpy array.

Column definitions:
    0) Split feature, -1 if leaf
    1) Split threshold
    2) Number of data points in this node
    3) Number of positives in this node
    4) Node number of this node (nodes are numbered in pre-order).
    5) Node number of left child, -1 if leaf
    6) Node number of right child, -1 if leaf

The tree is the simplest decision tree that I know how to make.
It does single-output, binary classification using the Gini impurity 
criterion. The tree is always grown out full, so there are no capacity 
control parameters.

author: David Thaler
date: August 2017
'''
import numpy as np
from .simple_splitter import split

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
    Returns a 2-D array of shape (num_nodes x 7) that describes the tree.
    Each row represents a node in pre-order (root, left, right).
    See the module comment for the column definitions.

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
    feature, thr = split(x, y)
    if feature == -1:
        return np.array([[feature, thr, ct, pos, node_num, -1, -1]])
    mask = x[:, feature] <= thr
    left_root = node_num + 1
    left_tree = build_tree(x[mask], y[mask], left_root)
    right_root = left_root + len(left_tree)
    right_tree = build_tree(x[~mask], y[~mask], right_root)
    root = np.array([[feature, thr, ct, pos, node_num, left_root, right_root]])
    return np.concatenate([root, left_tree, right_tree])


def apply(tree, x):
    '''
    Finds the node number in the provided tree (from build_tree) that each
    instance in x lands in.

    Args:
        tree: the array returned by build_tree
        x: m x n numpy array of numeric features

    Returns:
        1-D numpy array (dtype int) of leaf node numbers for each point in x.
    '''
    n = len(x)
    node = np.zeros(n, dtype=int)
    active = np.ones(n).astype(bool)
    while active.any():
        active = (tree[node, CHILD_LEFT_COL] != -1)
        xa = x[active]
        na = node[active]
        cfeat = (tree[na, FEATURE_COL]).astype(int)
        cx = xa[np.arange(len(xa)), cfeat]
        cthr = tree[na, THR_COL]
        cleft = (tree[na, CHILD_LEFT_COL]).astype(int)
        cright = (tree[na, CHILD_RIGHT_COL]).astype(int)
        cnode = np.where(cx <= cthr, cleft, cright)
        node[active] = cnode
    return node


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
