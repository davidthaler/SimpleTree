'''
These functions build a decision tree, which is output as a table 
contained in a numpy array.

Column definitions:
    0) Split feature, -1 if leaf
    1) Split threshold
    2) Node number of this node (nodes are numbered in pre-order).
    3) Node number of left child, -1 if leaf
    4) Node number of right child, -1 if leaf
    5) Number of data points in this node
    6) Value of this node (mean label)

author: David Thaler
date: August 2017
'''
import numpy as np
from .simple_splitter import split


# Position constants for the fields in the tree
FEATURE_COL = 0
THR_COL = 1
NODE_NUM_COL = 2
CHILD_LEFT_COL = 3
CHILD_RIGHT_COL = 4
CT_COL = 5
VAL_COL = 6


def build_tree(x, y, min_samples_leaf, depth_limit=-1, node_num=0):
    '''
    Recursively build a decision tree. 
    Returns a 2-D array of shape (num_nodes x 7) that describes the tree.
    Each row represents a node in pre-order (root, left, right).
    See the module comment for the column definitions.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.
        min_samples_leaf: minimum number of samples in a leaf, must be >= 1
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
    feature, thr, _, _ = split(x, y, min_samples_leaf)
    if feature == -1:
        return np.array([[feature, thr, node_num, -1, -1, ct, val]])
    mask = x[:, feature] <= thr
    left_root = node_num + 1
    left_tree = build_tree(x[mask], y[mask], min_samples_leaf, 
                                depth_limit - 1, left_root)
    right_root = left_root + len(left_tree)
    right_tree = build_tree(x[~mask], y[~mask], min_samples_leaf, 
                                depth_limit - 1, right_root)
    root = np.array([[feature, thr, node_num, left_root, right_root, ct, val]])
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


def values(tree):
    '''
    Extractor for the value column of this tree.

    Returns:
        the values for the nodes in this tree
    '''
    return tree[:, VAL_COL]
