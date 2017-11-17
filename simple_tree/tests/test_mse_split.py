'''
We test the mse_split function by comparing its output to some of the data
in the tree_ field of a fitted, depth 1 scikit-learn decision tree.

author: David Thaler
date: November 2017
'''
import unittest
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from ..datasets import load_mtcars, load_als
from ..simple_splitter import mse_split
from collections import namedtuple


PLACES = 4
CARS = load_mtcars()
ALS = load_als()[:2]    # ALS data has train/test, use train

Split = namedtuple('Split', ['feature', 'threshold', 'impurity', 'improvement'])

def extract_stump(x, y, **kwargs):
    '''Get data from the tree_ field from a depth 1 decision tree'''
    dt = DecisionTreeRegressor(max_depth=1, **kwargs)
    dt.fit(x, y)
    t = dt.tree_
    return Split(t.feature[0], t.threshold[0], t.impurity[0], -1)

class TestMseSplit(unittest.TestCase):

    def test_mtcars(self):
        '''Test main success scenario on mtcars data'''
        x, y = CARS
        split = Split(*mse_split(x, y, 1))
        gold = extract_stump(x, y)
        self.assertEqual(split.feature, gold.feature)
        self.assertAlmostEqual(split.threshold, gold.threshold, places=PLACES)
        self.assertAlmostEqual(split.impurity, gold.impurity, places=PLACES)

    def test_min_leaf(self):
        '''Test for same result when min_samples_leaf > 1'''
        x, y = CARS
        split = Split(*mse_split(x, y, 7))
        gold = extract_stump(x, y, min_samples_leaf=7)
        self.assertEqual(split.feature, gold.feature)
        self.assertAlmostEqual(split.threshold, gold.threshold, places=PLACES)
        self.assertAlmostEqual(split.impurity, gold.impurity, places=PLACES)

    def test_als(self):
        '''Test main success scenario on ALS data'''
        x, y = ALS
        split = Split(*mse_split(x, y, 1))
        gold = extract_stump(x, y)
        self.assertEqual(split.feature, gold.feature)
        self.assertAlmostEqual(split.threshold, gold.threshold, places=PLACES)
        self.assertAlmostEqual(split.impurity, gold.impurity, places=PLACES)