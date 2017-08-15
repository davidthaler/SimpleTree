'''
This version of the gini_splitter avoids using a dict(), which forces 
numba into object mode, and np.unique, which is not supported in numba.
It runs about 50x faster (in numba) on the iris data that the earlier 
python version of gini_splitter.

author: David Thaler
date: August 2017
'''
import numpy as np
import numba

# TODO: handle case of one unique value in a field

@numba.jit(nopython=True)
def split(x, y):
    '''
    Given features x and labels y, find the feature index and threshold for a
    split that produces the largest reduction in Gini impurity.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.

    Returns:
        3-tuple of feature index, split threshold and impurity of best split.
    '''
    m, n = x.shape
    best_feature = -1
    best_thr = 0.0
    best_score = 0.5
    # Iterate over features of x
    for feature_idx in range(n):
        f = x[:, feature_idx]
        sort_idx = f.argsort()
        fsort = f[sort_idx]
        ysort = y[sort_idx]
        ntot = np.zeros(m)
        uniq = np.zeros(m)
        npos = np.zeros(m)
        uniq[0] = fsort[0]                  # fsort[0] is unique
        ntot[0] = 1
        npos[0] += ysort[0]
        num_uniq = 1
        for k in range(1, m):
            if fsort[k] != fsort[k - 1]:    # fsort[k] is new.
                uniq[num_uniq] = fsort[k]
                num_uniq += 1
            ntot[num_uniq - 1] += 1
            npos[num_uniq - 1] += ysort[k]
        uniq = uniq[:num_uniq]
        npos = npos[:num_uniq]
        ntot = ntot[:num_uniq]
        
        nleft = ntot.cumsum()[:-1]
        npos_left = npos.cumsum()[:-1]
        nneg_left = nleft - npos_left
        nright = m - nleft
        npos_right = y.sum() - npos_left
        nneg_right = nright - npos_right

        gini_left = 1 - (npos_left/nleft)**2 - (nneg_left/nleft)**2
        gini_right = 1 - (npos_right/nright)**2 - (nneg_right/nright)**2
        gini_split = (nleft/m) * gini_left + (nright/m) * gini_right
        score = gini_split.min()
        if score < best_score:
            best_score = score
            best_feature = feature_idx
            split_idx = gini_split.argmin()
            best_thr = 0.5 * (uniq[split_idx] + uniq[split_idx + 1])
    return (best_feature, best_thr, best_score)
