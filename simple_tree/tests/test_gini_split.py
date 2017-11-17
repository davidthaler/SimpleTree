'''
First we test the gini_split function on toy data, focusing on edge cases.
Then we test the gini_split function by comparing the results of gini_split
to some of the data in the tree_ field of a fitted, depth 1 scikit-learn
decision tree.

author: David Thaler
date: November 2017
'''
import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ..datasets import load_iris, load_spam
from ..simple_splitter import gini_split
from collections import namedtuple

'''
Test Data:

Each column of the data excercizes one case:
    0: no split possible
    1: 1 split, but no progress possible
    2, 3: normal, imperfect split possible, both are same
    4: perfect split possible
'''
y = np.array([0., 0., 1., 1.])
x = np.array([[2., 2., 3., 3., 2.], 
              [2., 3., 3., 4., 2.], 
              [2., 2., 3., 4., 3.],
              [2., 3., 4., 4., 3.]])

Split = namedtuple('Split', ['feature', 'threshold', 'impurity', 'improvement'])
PLACES = 4
IRIS = load_iris()
SPAM = load_spam()[:2]      # spam has train/test data, use train

def extract_stump(x, y, **kwargs):
    '''Get data from the tree_ field from a depth 1 decision tree'''
    dt = DecisionTreeClassifier(max_depth=1, **kwargs)
    dt.fit(x, y)
    t = dt.tree_
    return Split(t.feature[0], t.threshold[0], t.impurity[0], -1)

class TestGiniSplit(unittest.TestCase):

    # Toy data
    
    def test_one_row(self):
        '''Check for no split/no error on one-row data'''
        split = Split(*gini_split(x[:1], y[:1], 1))
        self.assertEqual(split.feature, -1.)

    def test_pure_node(self):
        '''Test for no split/no error on pure node'''
        split = Split(*gini_split(x[:2], y[:2], 1))
        self.assertEqual(split.feature, -1.)

    def test_pure_x(self):
        '''Test for no split/no error if x has only one value'''
        # x is a 2-D array of column 0 only
        split = Split(*gini_split(x[:, :1], y, 1))
        self.assertEqual(split.feature, -1.)

    def test_no_improve(self):
        '''Test no split/no error if no improvement possible'''
        split = Split(*gini_split(x[:, 1:2], y, 1))
        self.assertEqual(split.feature, -1.)

    def test_perfect(self):
        '''Find correct split if perfect split possible'''
        split = Split(*gini_split(x, y, 1))
        self.assertEqual(split.feature, 4)
        self.assertAlmostEqual(split.threshold, 2.5)

    def test_imperfect(self):
        '''Check for imperfect split in column 2'''
        split = Split(*gini_split(x[:, :-2], y, 1))
        self.assertEqual(split.feature, 2)
        self.assertAlmostEqual(split.threshold, 3.5)

    # Spam and iris data 

    def test_success(self):
        '''Test for the main success scenario on spam data'''
        x, y = SPAM
        split = Split(*gini_split(x, y, 1))
        gold = extract_stump(x, y)
        self.assertEqual(split.feature, gold.feature)
        self.assertAlmostEqual(split.threshold, gold.threshold, places=PLACES)
        self.assertAlmostEqual(split.impurity, gold.impurity, places=PLACES)

    def test_min_samples_leaf(self):
        '''Test for same split when min_samples_leaf != 1'''
        x, y = IRIS
        split = Split(*gini_split(x[::10], y[::10], 4))
        gold = extract_stump(x[::10], y[::10], min_samples_leaf=4)
        self.assertEqual(split.feature, gold.feature)
        self.assertAlmostEqual(split.feature, gold.feature, places=PLACES)
        self.assertAlmostEqual(split.impurity, gold.impurity, places=PLACES)


    