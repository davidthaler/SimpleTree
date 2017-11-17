'''
We test ClassificationTree by comparing its output to that of a scikit-learn
DecisionTreeClassifer. Neither algorithm is deterministic if any split lacks
a unique best value. We use the spam data and limit the max_depth so that the
best splits will be unique.

author: David Thaler
date: November 2017
'''
import unittest
from .. import ClassificationTree
from sklearn.tree import DecisionTreeClassifier
from .. datasets import load_spam


class TestClassificationTree(unittest.TestCase):

    def test_success(self):
        '''Test that we get the same predictions with depth in 1..5'''
        xtr, ytr, xte, yte = load_spam()
        for d in range(1, 6):
            with self.subTest(depth=d):
                dt = DecisionTreeClassifier(max_depth=d)
                dt.fit(xtr, ytr)
                pred = dt.predict(xte)
                mytree = ClassificationTree(max_depth=d)
                mytree.fit(xtr, ytr)
                mypred = mytree.predict(xte)
                self.assertTrue((pred == mypred).all())
