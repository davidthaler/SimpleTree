import numpy as np
from collections import Counter
import pdb

def split(x, y):
    '''
    Given features x and labels y, find the feature index and threshold for a
    split that produces the largest reduction in Gini impurity.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.

    Returns:
        2-tuple of feature index and split threshold.
    '''
    m, n = x.shape
    best_feature = -1
    best_thr = 0.0
    best_score = 0.5
    # Iterate over features of x
    for j in range(n):
        f = x[:, j]
        # TODO: get the Counters out of here to help numba out
        cts = Counter()
        pos = Counter()
        neg = Counter()
        # get totals, positives and negatives for each *unique* value in f
        for k in range(m):
            cts[f[k]] += 1
            pos[f[k]] += y[k]
            neg[f[k]] += (1 - y[k])
        # We start with all data on the right
        n_left = 0                           # number of data points on left branch
        npos_left = 0                        # number of positives on left
        nneg_left = 0                        # number of positives on left
        n_right = m                          # number of data points on right branch
        npos_right = y.sum()                 # number of positives on right
        nneg_right = n_right - npos_right    # number of negatives on right
        g = np.sort(np.unique(f))
        # range(len(g) - 1) omits the split with an empty right branch
        for k in range(len(g) - 1):
            val = g[k]
            #pdb.set_trace()
            n_left += cts[val]
            n_right -= cts[val]
            npos_left += pos[val]
            npos_right -= pos[val]
            nneg_left += neg[val]
            nneg_right -= neg[val]
            gini_left  = 1 - (npos_left/n_left)**2 - (nneg_left/n_left)**2
            gini_right = 1 - (npos_right/n_right)**2 - (nneg_right/n_right)**2
            gini_split = (n_left/m) * gini_left + (n_right/m) * gini_right
            if gini_split < best_score:
                best_feature = j
                best_thr = 0.5 * (g[k] + g[k+1])
                best_score = gini_split
    return (best_feature, best_thr, best_score)
