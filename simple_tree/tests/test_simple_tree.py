'''
Test code for simple_tree_builder and SimpleTree using modified iris data.
These tests compare SimpleTree to sklearn.tree.DecisionTreeClassifier
at its default parameter values. 

Note that while SimpleTree is deterministic, DecisionTreeClassifier is not,
and it will sometimes return different trees. I think setting random_state=0 
in the DecisionTreeClassifier gets us the tree that matches ours, 
but if a bunch of test start failing, that is where to look.


To run:

    >>> python -m pytest

or:

    >>> python -m pytest test_simple_tree.py

...from within the project or tests directory

author: David Thaler
date: August 2017
'''
import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from .. import simple_tree_builder as stb


# Set up some test data
iris = load_iris()
y = (iris.target == 2).astype(float)
x = iris.data
mytree = stb.build_tree(x, y)
dt = DecisionTreeClassifier(random_state=0)
dt.fit(x, y)
tree = dt.tree_
# Feature number is < 0 for leaf nodes on both trees
sk_internal_mask = tree.feature >= 0 
my_internal_mask = mytree[:, stb.FEATURE_COL] >= 0


def test_size():
    assert(tree.node_count == mytree.shape[0])

def test_counts():
    skl_cts = tree.n_node_samples
    my_cts = mytree[:, stb.CT_COL]
    assert (skl_cts==my_cts).all()

def test_positives():
    skl_pos = tree.value.squeeze()[:, 1]
    my_pos = mytree[:, stb.POS_COL]
    assert (skl_pos == my_pos).all()

def test_same_internal_nodes():
    assert (my_internal_mask == sk_internal_mask).all()

def test_same_features():
    my_feature_idx = mytree[my_internal_mask, stb.FEATURE_COL]
    sk_feature_idx = tree.feature[sk_internal_mask]
    assert (my_feature_idx == sk_feature_idx).all()

def test_same_thresholds():
    my_thr = mytree[my_internal_mask, stb.THR_COL]
    sk_thr = tree.threshold[sk_internal_mask]
    assert np.allclose(my_thr, sk_thr, rtol=1e-4, atol=1e-4)

def test_left_children():
    assert (mytree[:, stb.CHILD_LEFT_COL] == tree.children_left).all()

def test_right_children():
    assert (mytree[:, stb.CHILD_RIGHT_COL] == tree.children_right).all()

def test_apply():
    # use the DecisionTree, not tree, due to dtype issue
    sk_leaf_idx = dt.apply(x)
    my_leaf_idx = stb.apply(mytree, x)
    assert(my_leaf_idx == sk_leaf_idx).all()

def test_predict():
    dt.fit(x[1::2], y[1::2])
    mytree = stb.build_tree(x[1::2], y[1::2])
    sk_pred = dt.predict(x)
    my_pred = stb.predict(mytree, x)
    assert (sk_pred==my_pred).all()
