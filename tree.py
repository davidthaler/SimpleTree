import numpy as np
from gini_splitter import split


def build_tree(x, y, node_num=0):
    '''
    Recursively build a decision tree.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.
        node_num: the node number of this node
            default 0 is for the root node

    Returns:
        parallel numpy arrays with the feature index(in x), 
        split threshold, sample count and positive count
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
