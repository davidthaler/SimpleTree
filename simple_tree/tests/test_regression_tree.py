'''
RegressionTree is tested by comparing its output to that of
sklearn.DecisionTreeRegressor. If equally good splits exist, the output
of these models is non-deterministic. This is mainly a problem when there
is a small amount of data in the nodes, so we test on a larger dataset
and limit the depth.

author: David Thaler
date: November 2017
'''
import unittest
import numpy as np
from .. import RegressionTree
from sklearn.tree import DecisionTreeRegressor
from .. datasets import load_als

class TestRegressionTree(unittest.TestCase):

    def test_success(self):
        '''Test for the same predictions at several shallow depths'''
        xtr, ytr, xte, yte = load_als()
        for d in range(1, 4):
            with self.subTest(depth=d):
                dt = DecisionTreeRegressor(max_depth=d)
                dt.fit(xtr, ytr)
                pred = dt.predict(xte)
                mytree = RegressionTree(max_depth=d)
                mytree.fit(xtr, ytr)
                mypred = mytree.predict(xte)
                self.assertTrue(np.allclose(pred, mypred))