'''
The split function here finds splits that minimize gini impurity.
It runs under numba for speed, since these are the innermost loops in 
decision tree fitting.

author: David Thaler
date: August 2017
'''
import numpy as np


def split(x, y):
    '''
    Given features x and labels y, find the feature index and threshold for a
    split that produces the largest reduction in Gini impurity.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.

    Returns:
        2-tuple of feature index and split threshold of best split.
    '''
    m, n = x.shape
    NO_SPLIT = (-1, 0.0)
    # the Gini impurity of this node before splitting
    ys = y.sum()
    node_score = 1 - (ys / m)**2 - ((m - ys) / m)**2
    # check for pure nodes
    if node_score == 0:
        return NO_SPLIT
    # Stores score, threshold for each feature (1 is a max value for gini)
    results = np.ones((n, 2))
    # Iterate over features of x
    for feature_idx in range(n):
        f = x[:, feature_idx]

        # Produce 3 arrays:
        # 1) unique values in f
        # 2) count of each unique value (often 1)
        # 3) # of positives for each unique value
        uniq, ubins = np.unique(f, return_inverse=True)
        # ensures at least 2 uniques (1 split) in x
        if len(uniq) == 1:
            continue
        cts = np.bincount(ubins)
        pos = np.bincount(ubins, weights=y)
        
        # Get cumulative counts/positives/negatives for each possible split
        nleft = cts.cumsum()[:-1]
        npos_left = pos.cumsum()[:-1]
        nneg_left = nleft - npos_left
        nright = m - nleft
        npos_right = ys - npos_left
        nneg_right = nright - npos_right

        # Compute Gini impurity for each split
        gini_left = 1 - (npos_left/nleft)**2 - (nneg_left/nleft)**2
        gini_right = 1 - (npos_right/nright)**2 - (nneg_right/nright)**2
        gini_split = (nleft/m) * gini_left + (nright/m) * gini_right

        # Store the best split on this feature
        split_idx = gini_split.argmin()
        score = gini_split[split_idx]
        thr = 0.5 * (uniq[split_idx] + uniq[split_idx + 1])
        results[feature_idx] = (score, thr)
    best_split_idx = results[:, 0].argmin()
    best_score = results[best_split_idx, 0]
    if best_score < node_score:
        best_thr = results[best_split_idx, 1]
        return (best_split_idx, best_thr)
    else:
        return NO_SPLIT
