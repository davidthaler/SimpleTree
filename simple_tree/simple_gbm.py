'''
Module simple_gbm.py implements least-squares and bernoulli loss 
gradient boosting models using the regression tree model from simple tree.

author: David Thaler
date: November 2017
'''
import numpy as np
from scipy.special import logit, expit
from . import RegressionTree


class SimpleGBM():
    '''
    SimpleGBM is a base class for gradient boosting models.
    The subclasses must define start_gbm, update_leaves and update_residual
    methods.

    Args:
        n_trees: (int) number of trees to fit
        learn_rate: the step size for the model
        max_depth: (int) the maximum depth of the trees grown.
    '''

    def __init__(self, n_trees=100, learn_rate=0.1, max_depth=3):
        self.n_trees = n_trees
        self.learn_rate = learn_rate
        self.max_depth = max_depth

    def fit(self, x, y):
        '''
        The fit() method fits gradient boosting models for regressors and
        classifiers. The subclasses need to implement the following methods
        for this to work: start_gbm, update_leaves and update_residual.

        Args:
            x: Training data features; ndarray of shape (n_samples, n_features)
            y: Training set labels; shape is (n_samples, )
        
        Returns:
            self, the fitted model
        '''
        n, p = x.shape
        self.estimators_ = []
        self.f0, r = self.start_gbm(y)
        f = self.f0 * np.ones_like(y)
        for k in range(self.n_trees):
            model = RegressionTree(max_depth=self.max_depth)
            model.fit(x, r)
            self.estimators_.append(model)
            leaves = model.apply(x)
            model.values = self.update_leaves(leaves, y, r)
            f += self.learn_rate * model.predict(x)
            r = self.update_residual(y, f)
        return self

    def decision_function(self, x):
        '''
        Returns the decision function for each row in x. 
        In a regression model, this is the estimate of the targets. 
        In a classification model, it is the estimated log-odds of the
        positive class.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) decision function for each row in x
        '''
        pred = np.zeros(len(x)) + self.f0
        for model in self.estimators_:
            pred += self.learn_rate * model.predict(x)
        return pred

    def __repr__(self):
        '''
        String representation contains actual class name and all
        constructor parameters.

        Returns:
            a string representation of this model
        '''
        name = self.__class__.__name__
        params = (name, self.n_trees, self.learn_rate, self.max_depth)
        return '%s(n_trees=%s, learn_rate=%s, max_depth=%s)' % params


class GBRegressor(SimpleGBM):
    '''
    GBRegressor implements gradient boosting for least squares regression.
    It extends SimpleGBM and implements the required methods: start_gbm, 
    update_leaves and update_residual. It also implements predict.
    '''

    def start_gbm(self, y):
        '''
        Compute the initial pseudo-residual, r, and the base prediction,
        f0, needed to begin gradient boosting.

        Args:
            y: the regression targets

        Returns:
            2-tuple of f0 (a float) and r (a numpy array like y)
        '''
        f0 = y.mean()
        r = y - f0
        return f0, r

    def update_leaves(self, leaves, y, r):
        '''
        Update the base learner leaf values. NB: For this model the new values
        are the mean of the data in each node, which is the same as the values
        learned by the tree model itself.

        Args:
            leaves: an array of leaf indices for each row of training data
            y: the regression targets
            r: the current residual

        Returns:
            an array of new node values to assign in the current base learner
        '''
        ct = np.bincount(leaves)
        ct = np.where(ct==0, 1, ct)
        ytot = np.bincount(leaves, weights=r)
        return ytot / ct

    def update_residual(self, y, f):
        '''
        Compute the new residual, after a new base learner has been
        added to the working response.

        Args:
            y: the regression targets
            f: the working response

        Returns:
            the new residual (numpy array like y and f)
        '''
        return y - f

    def predict(self, x):
        '''
        Estimates target value for each row in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) of estimates of target for each row in x
        '''
        return self.decision_function(x)


class GBClassifier(SimpleGBM):
    '''
    GBClassifier extends SimpleGBM to implement a binary gradient boosting
    classifier (bernoulli loss). It implements the required methods: start_gbm,
    update_leaves and update_residual. It also implements predict and
    predict_proba.
    '''

    def start_gbm(self, y):
        '''
        Compute the initial pseudo-residual, r, and the base prediction,
        f0, needed to begin gradient boosting.

        Args:
            y: 0-1 classification targets

        Returns:
            2-tuple of f0 (a float) and r (a numpy array like y)
        '''
        p = y.mean()
        f0 = logit(p)
        r = y - p
        return f0, r

    def update_leaves(self, leaves, y, r):
        '''
        Update the base learner leaf values on the log-odds scale.

        Args:
            leaves: an array of leaf indices for each row of training data
            y: 0-1 classification targets
            r: the current pseudo-residual

        Returns:
            an array of new node values to assign in the current base learner
        '''
        num = np.bincount(leaves, weights=r)
        den_vals = (y - r) * (1 - y + r)
        raw_den = np.bincount(leaves, weights=den_vals)
        den0idx = (np.abs(raw_den) < 1e-100)
        den = np.where(den0idx, 1., raw_den)
        vals = np.where(den0idx, 0., num/den)
        return vals

    def update_residual(self, y, f):
        '''
        Compute the new pseudo-residual, after a new base learner has been
        added to the working response.

        Args:
            y: 0-1 classification targets
            f: the working response, on the log-odds scale

        Returns:
            the new pseudo-residual (numpy array like y and f)
        '''
        return y - expit(f)

    def predict_proba(self, x):
        '''
        Predicts probabilities of the positve class for each row in x

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples,) of probabilities for class 1.
        '''
        return expit(self.decision_function(x))

    def predict(self, x):
        '''
        Predicts a 0-1 target label for each row in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) of estimates of target for each row in x
        '''
        return (self.decision_function(x) > 0).astype(int) 