# SimpleTree
SimpleTree is a python-only package of decision trees and tree ensemble models.
It has decision tree, random forest and gradient boosting models for 
binary classification and regression.

## Goals
This project aims to produce python-only tree and tree ensemble models that are:   

* readable and hackable by average programmers
* as accurate as scikit-learn DecisionTreeClassifier
* fast enough to be used on real data

## Requirements

* numpy
* pandas is used for loading the included sample datasets
* [scikit-learn](http://scikit-learn.org/stable/index.html) is used in the examples.

## Installation
This package is not on pypi, so you need to clone from github. Then you can install with pip.

    git clone https://github.com/davidthaler/SimpleTree.git
    cd SimpleTree
    pip install -e .

The `pip install -e .` installation allows you to hack on `simple_tree` and still have your changes 
show up when you `import simple_tree`.

## Usage

There are sample datasets in the `datasets/` package. 
These are loaded with named functions.
Here we load the spam data from the [CASI book](https://web.stanford.edu/~hastie/CASI_files/data.html).

    >> from simple_tree.datasets import load_spam
    >> xtr, ytr, xte, yte = load_spam()

The decision tree, random forest and gradient boosting models implement 
fit/predict methods as in sklearn.
Fitting and predicting with a `ClassificationTree`:

    >> from simple_tree import ClassificationTree
    >> mytree = ClassifcationTree(min_samples_leaf=5)
    >> mytree.fit(xtr, ytr)
    >> pred = mytree.predict(xte)

Fitting and predicting with an `RFClassifier`:

    >> from simple_tree import RFClassifier
    >> myrf = RFClassifier(n_trees=30, min_samples_leaf=5)
    >> myrf.fit(xtr, ytr)
    >> pred = mytree.predict(xte)

Using a `GBClassifier`:

    >> from simple_tree import GBClassifier
    >> mygbm = GBClassifier()
    >> mygbm.fit(xtr, ytr)
    >> pred = mygbm.predict(xte)

The usage for regression models is similar. 
See `help(model)` for the parameters of each particular model. 
There are more extensive usage examples in the `examples/` directory. 
The examples are run as:

    >> python spam_example.py

## Limitations
The classifiers only do single-output binary classification with 0-1 labels.
Regression models only do single output least-squares regression.
Models do not take sample weights or class weights.
Models in SimpleTree implement part of the scikit-learn estimator API,
but not `get_param/set_params` or `score`, so they do not interoperate 
with hyperparameter search functions in sklearn.model_selection.

## Accuracy
Models in SimpleTree generally match the corresponding models in 
scikit-learn for accuracy. Here we compare a `DecisionTreeClassifier` 
from scikit-learn to `ClassificationTree`.

    >> from sklearn.tree import DecisionTreeClassifier
    >> from sklearn.metrics import ConfusionMatrix as cm
    >> dt = DecisionTreeClassifier(min_samples_leaf=5)
    >> dt.fit(xtr, ytr)
    >> sk_pred = dt.predict(xte)
    >> cm(yte, sk_pred)
    array([[875,  66],
           [ 65, 530]])

We get a similar result from `ClassificationTree`:

    >> from simple_tree import ClassificationTree
    >> st = ClassificationTree(min_samples_leaf=5)
    >> st.fit(xtr, ytr)
    >> st_pred = st.predict(xte)
    >> cm(yte, st_pred)
    array([[874,  67],
           [ 61, 534]])

Next we compare a `GradientBoostingClassifier` from scikit-learn 
to a `GBClassifier` from SimpleTree:

    >> from sklearn.metrics import log_loss
    >> from sklearn.ensemble import GradientBoostingClassifier
    >> gbm = GradientBoostingClassifer()
    >> gbm.fit(xtr, ytr)
    >> cm(yte, gbm.predict(xte))
    array([[903,  38],
           [ 44, 551]])
    >> log_loss(yte, gbm.predict_proba(xte))
    0.15015159848399615
    >> from simple_tree import GBClassifier
    >> mygbm = GBClassifier()
    >> mygbm.fit(xtr, ytr)
    >> cm(yte, mygbm.predict(xte))
    array([[904,  37],
           [ 44, 551]])
    >> log_loss(yte, mygbm.predict_proba(xte))
    0.14861460234660964

Finally we compare a `RandomForestClassifier` to an `RFClassifier`:

    >> from sklearn.ensemble import RandomForestClassifier
    >> rf = RandomForestClassifier(n_estimators=30, min_samples_leaf=5)
    >> rf.fit(xtr, ytr)
    >> cm(yte, rf.predict(xte))
    array([[907,  34],
           [ 59, 536]])
    >> from simple_tree import RFClassifier
    >> myrf = RFClassifier(n_trees=30, min_samples_leaf=5)
    >> myrf.fit(xtr, ytr)
    >> cm(yte, myrf.predict(xte))
    array([[910,  31],
           [ 50, 545]])

The examples in the `examples/` directory output accuracy/confusion matrix
 or mean squared error.

## Performance

For the spam data training set (3065 x 57). Fitting a `ClassificationTree`:

    >> %timeit st.fit(xtr, ytr)
    1 loop, best of 3: 605 ms per loop

Predicting on the spam test set (1536 x 57):

    >> %timeit st.predict(xte)
    1000 loops, best of 3: 3.59 ms per loop

So it fits a realistic, smaller data set in about 0.7s and prediction is quite fast.
For the gradient boosting model, with 100 trees and depth 3, we get:

    >> %timeit mygbm.fit(xtr, ytr)
    1 loop, best of 3: 5.74 s per loop
    >> %timeit mygbm.predict(xte)
    10 loops, best of 3: 83.2 ms per loop

Note, however, that these times are considerably slower that scikit-learn, 
which makes extensive use of compiled (cython) extensions.
There are wall-clock timings on the scripts in the `examples` folder.

## Extending GBM
The gradient boosting models provided in SimpleTree implement single-output, 
binary classification and least-squares regression. To extend the model to 
other objective functions, subclass `simple_tree.simple_gbm.SimpleGBM` and 
implement the three functions:

* `start_gbm`: compute the base estimate and initial pseudo-residual.
* `update_leaves`: update the leaf values after learning each base estimator
* `update_residual`: compute the new pseudo-residual.

You will also need to implement `predict` and similar, but the base class has
`decision_function`.

## Tests
There are some tests in the `tests/` directory. They require scikit-learn. 
To run them, go to the project directory and enter:

    >> python -m unittest -v
