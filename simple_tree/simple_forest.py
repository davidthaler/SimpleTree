'''
Module simple_forest implements random forest models for regression and 
binary classification using decision trees from simple_tree.
Class SimpleForest is a base class for regression and classification models.
Class RFClassifier is a concrete class for binary classification.
Class RFRegressor is an RF regression model.

author: David Thaler
date: Novermber 2017
'''
import numpy as np
from . import RegressionTree, ClassificationTree

class SimpleForest():
    '''
    SimpleForest is a base class for random forest models based 
    on the trees in simple_tree.

    Args:
        n_trees: number of trees to use in model
        other: same as underlying tree type
    '''
    def __init__(self, n_trees=30, min_samples_leaf=1, max_features=None,
                    max_depth=None):
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth

    def fit(self, x, y):
        '''
        Fits a random forest model using the base estimator from a subclass. 
        Also sets the oob_decision_function_ attribute.

        Args:
            x: Training data features; ndarray of shape (n_samples, n_features)
            y: Training set labels; shape is (n_samples, )

        Returns:
            Returns self, the fitted estimator
        '''
        m, n = x.shape
        self.estimators_ = []
        all_idx = np.arange(m)
        oob_ct = np.zeros(m)
        oob_tot_dv = np.zeros(m)
        for k in range(self.n_trees):
            model = self._get_base_estimator(n)
            boot_idx = np.random.randint(m, size=m)
            oob_idx = np.setdiff1d(all_idx, boot_idx)
            model.fit(x[boot_idx], y[boot_idx])
            self.estimators_.append(model)
            oob_ct[oob_idx] += 1
            oob_tot_dv[oob_idx] += model.decision_function(x[oob_idx])
        self.oob_decision_function = oob_tot_dv / oob_ct
        return self

    def decision_function(self, x):
        '''
        Returns the decision function for each row in x.
        For regression models, this is the prediction.
        For classifcation models, it is the class 1 probability.
        In either case, it is the average over the trees in this forest.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) decision function for each row in x
        '''
        dv = np.zeros(len(x))
        for model in self.estimators_:
            dv += model.decision_function(x)
        return dv / self.n_trees

    def __repr__(self):
        '''Return a string representation of this model'''
        name = self.__class__.__name__
        return ('%s(n_trees=%s, min_samples_leaf=%s, max_features=%s, max_depth=%s)' %
                (name, self.n_trees, self.min_samples_leaf,
                    self.max_features, self.max_depth))


class RFClassifier(SimpleForest):
    '''
    Concrete class extending SimpleForest as a classification model.
    It does single-output, binary classification and requires 0-1 labels.

    Args:
        see SimpleForest
    '''

    def _get_base_estimator(self, n_features):
        '''
        Return configured decision tree classifier.
        Default for max_features is sqrt(number of features)

        Args:
            n_features: number of columns in x (x.shape[1])

        Returns:
            Configured ClassifcationTree for this RF model.
        '''
        max_features = (int(np.ceil(np.sqrt(n_features)))
                        if self.max_features is None else self.max_features)
        return ClassificationTree(min_samples_leaf=self.min_samples_leaf,
                                  max_features=max_features,
                                  max_depth=self.max_depth)

    def predict_proba(self, x):
        '''
        Predicts probabilities of the positive class for each row in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples,) of probabilities for class 1.
        '''
        return self.decision_function(x)

    def predict(self, x):
        '''
        Predicts class membership for the rows in x. 

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array of shape (n_samples, ) of class labels for each row
        '''
        return (self.predict_proba(x) > 0.5).astype(int)


class RFRegressor(SimpleForest):
    '''
    Concrete class extending SimpleForest as a regression model.

    Args:
        see SimpleForest
    '''
    def _get_base_estimator(self, n_features):
        '''
        Return configured decision tree regressor.
        Default for max_features is (number of features) / 3

        Args:
            n_features: number of columns in x (x.shape[1])

        Returns:
            Configured RegressionTree for this RF model.
        '''
        max_features = (int(np.ceil(n_features / 3))
                        if self.max_features is None else self.max_features)
        return RegressionTree(min_samples_leaf=self.min_samples_leaf,
                              max_features=max_features,
                              max_depth=self.max_depth)

    def predict(self, x):
        '''
        Estimates target for each row in x.

        Args:
            x: Test data to predict; ndarray of shape (n_samples, n_features)

        Returns:
            array (n_samples,) of estimates of target for each row in x
        '''
        return self.decision_function(x)