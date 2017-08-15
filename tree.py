import numpy as np
from gini_splitter import split


def build_tree(x, y):
    '''
    Recursively build a decision tree.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.

    Returns:
        parallel numpy arrays with the feature index(in x), 
        split threshold, sample count and positive count
    '''
    ct = len(y)
    pos = y.sum()
    feature = -1
    thr = 0.0
    if pos == 0 or pos == ct:
        return np.array([[feature, thr, ct, pos]])
    feature, thr, _ = split(x, y)
    left_mask = x[:, feature] <= thr
    x_left = x[left_mask]
    y_left = y[left_mask]
    left_tree = build_tree(x_left, y_left)
    right_mask = x[:, feature] > thr
    x_right = x[right_mask]
    y_right = y[right_mask]
    right_tree = build_tree(x_right, y_right)
    root = np.array([[feature, thr, ct, pos]])
    return np.concatenate([root, left_tree, right_tree])
