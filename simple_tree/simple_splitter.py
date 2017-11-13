'''
The gini_split function finds the split (feature index, feature value)
that minimizes gini impurity. Function mse_split does the same for mse.

author: David Thaler
date: August 2017
'''
import numpy as np


def gini_split(x, y, min_samples_leaf):
    '''
    Given features x and labels y, find the feature index and threshold for a
    split that produces the largest reduction in Gini impurity.

    Args:
        x: m x n numpy array of numeric features
        y: m-element 1-D numpy array of labels; must be 0-1.
        min_samples_leaf: minimum number of samples in a leaf, must be >= 1

    Returns:
        4-tuple of feature index and split threshold of best split,
        node gini score and gini improvement from the split
    '''
    m, n = x.shape
    ys = y.sum()
    # the Gini impurity of this node before splitting
    node_score = 1 - (ys / m)**2 - ((m - ys) / m)**2
    NO_SPLIT = (-1, 0.0, node_score, 0.0)
    # check for pure nodes
    if node_score == 0:
        return NO_SPLIT
    # Stores score, threshold for each feature (1 > max value for gini)
    results = np.ones((n, 2))
    # Iterate over features of x
    for feature_idx in np.random.permutation(n):
        # Produce 3 arrays:
        # 1) unique values in feature
        # 2) count of each unique value (often 1)
        # 3) # of positives for each unique value
        uniq, ubins = np.unique(x[:, feature_idx], return_inverse=True)
        # ensures at least 2 uniques (1 split) in x
        if len(uniq) == 1:
            continue
        cts = np.bincount(ubins)
        pos = np.bincount(ubins, weights=y)
        
        # Get cumulative counts/positives/negatives for each possible split
        nleft = cts.cumsum()
        nright = m - nleft
        if min_samples_leaf == 1:           # common special case
            a, b = 0, -1
        else:
            mask = (nleft >= min_samples_leaf) & (nright >= min_samples_leaf)
            a = mask.argmax()
            b = -(mask[::-1].argmax())
        nleft = nleft[a:b]
        if len(nleft) == 0:                 # no valid splits
            continue
        nright = nright[a:b]
        npos_left = pos.cumsum()[a:b]
        nneg_left = nleft - npos_left
        npos_right = ys - npos_left
        nneg_right = nright - npos_right

        # Compute Gini impurity for each split
        gini_left = 1 - (npos_left/nleft)**2 - (nneg_left/nleft)**2
        gini_right = 1 - (npos_right/nright)**2 - (nneg_right/nright)**2
        gini_split = (nleft/m) * gini_left + (nright/m) * gini_right

        # Store the best split on this feature
        split_loc = gini_split.argmin()
        score = gini_split[split_loc]
        split_idx = a + split_loc
        thr = 0.5 * (uniq[split_idx] + uniq[split_idx + 1])
        results[feature_idx] = (score, thr)
    best_split_idx = results[:, 0].argmin()
    best_score = results[best_split_idx, 0]
    if best_score < node_score:
        best_thr = results[best_split_idx, 1]
        return (best_split_idx, best_thr, node_score, node_score - best_score)
    else:
        return NO_SPLIT

def mse_split(x, y, min_samples_leaf):
    '''
    Python-only function to find the split (feature index, feature value)
    that minimized mean sqaured error on regression targets.
    '''
    m, n = x.shape
    ys = y.sum()
    yy = y * y
    yys = yy.sum()
    ym = y.mean()
    # RSS before splitting
    node_score = ((y - ym)**2).sum()
    # feature, threshold, score, improvement
    NO_SPLIT = (-1, 0.0, node_score, 0.0)
    # check for pure nodes
    if node_score == 0:
        return NO_SPLIT

    # node_score is an upper limit value for the split
    results = node_score * np.ones((n, 2))

    for feature_idx in np.random.permutation(n):
        # Produce 4 arrays:
        # 1) sorted unique values in f
        # 2) count of each unique value (usually 1)
        # 3) sum of targets for each unique
        # 4) sum of squared targets for each unique
        uniq, ubins = np.unique(x[:, feature_idx], return_inverse=True)

        # ensure at least 2 uniques (1 split)
        if len(uniq) == 1:
            continue
        cts = np.bincount(ubins)

        # sum of y and y^2, grouped by unique values in x[:, feature_idx]
        y_gp = np.bincount(ubins, weights=y)
        yy_gp = np.bincount(ubins, weights=yy)
        
        # Get cumulative counts for each possible split
        nleft = cts.cumsum()
        nright = m - nleft

        # find splits that respect min_samples_leaf
        mask = (nleft >= min_samples_leaf) & (nright >= min_samples_leaf)
        a = mask.argmax()
        b = -(mask[::-1].argmax())

        # trim nleft, nright to those valid splits
        nleft = nleft[a:b]
        nright = nright[a:b]

        # Are there any valid splits?
        if len(nleft) == 0:
            continue

        # Get cumulative sums of y, y^2 for each possible split
        yleft = y_gp.cumsum()[a:b]
        yright = ys - yleft
        yyleft = yy_gp.cumsum()[a:b]
        yyright = yys - yyleft

        # Compute combined mse for each split
        sk_left = yyleft - (yleft**2) / nleft
        sk_right = yyright - (yright**2) / nright
        sk = sk_left + sk_right

        split_pos = sk.argmin()
        split_score = sk[split_pos]
        split_idx = a + split_pos
        thr = 0.5 * (uniq[split_idx] + uniq[split_idx + 1])
        results[feature_idx] = (split_score, thr)
    best_split_idx = results[:, 0].argmin()
    best_score = results[best_split_idx, 0]
    if best_score < node_score:
        best_thr = results[best_split_idx, 1]
        # NB: node_score is actually RSS
        mse = node_score / m
        improvement = (node_score - best_score) / m
        return (best_split_idx, best_thr, mse, improvement)
    else:
        return NO_SPLIT
        
