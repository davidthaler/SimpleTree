import numpy as np
from gini_splitter import split
import numba
import pdb

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
    feature = -1
    thr = 0.0
    if pos == 0 or pos == ct:
        return np.array([[feature, thr, ct, pos, node_num, -1, -1]])
    feature, thr, _ = split(x, y)
    mask = x[:, feature] <= thr
    left_root = node_num + 1
    left_tree = build_tree(x[mask], y[mask], left_root)
    right_root = left_root + len(left_tree)
    right_tree = build_tree(x[~mask], y[~mask], right_root)
    root = np.array([[feature, thr, ct, pos, node_num, left_root, right_root]])
    return np.concatenate([root, left_tree, right_tree])


@numba.jit
def apply(tree, x):
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
    return out
