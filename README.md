# SimpleTree
SimpleTree is a python-only decision tree classifier.

## Goals
This project aims to produce an all-python decision tree classifier that is:   

* readable and hackable by average programmers
* as accurate as scikit-learn DecisionTreeClassifier
* fast enough to actually be used on real data

## Requirements
* numpy is used throughout
* pytest is needed if you want to run the tests.
* [scikit-learn](http://scikit-learn.org/stable/index.html) is used in the tests.

## Installation
This package is not on pypi, so you need to clone from github. Then you can install with pip.

    git clone https://github.com/davidthaler/SimpleTree.git
    cd SimpleTree
    pip install -e .

The `pip install -e .` installation allows you to hack on `simple_tree` and still have your changes 
show up when you `import simple_tree`.

## Usage

Using the sklearn-compatible estimator class:

    from simple_tree import SimpleTree
    st = SimpleTree()
    st.fit(x_train, y_train)
    pred = st.predict(x_test)

Or, using the functions in `simple_tree_builder` directly:     

    from simple_tree import *
    mytree = build_tree(x_train, y_train)
    pred = predict(mytree, x_test)

The `build_tree` function returns a numpy array that describes the tree.
Then that is passed to `predict` or `apply`, along with new data for prediction.

## Design
The code here has three parts:

* splitter: `split()` in `simple_splitter.py` takes in data and returns the best feature and threshold to split on.
* tree builder: `simple_tree_builder.py` contains `build_tree()`, which builds the tree recursively.
* sklearn interface: `simple_tree.py` contains a simple class to make an scikit-learn 
compatible estimator from this code.

All of the code is in python.
To get reasonable speed, the numpy code is vectorized as far as possible.
To keep this code as simple as possible, it only performs binary classification.
It uses [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) as its split criterion. 
Trees are grown out full, so there are no capacity control parameters.

### Data Format
The functions in `simple_tree_builder` create and consume the decision tree in an array format,
in which each column of the array contains one type of data about the tree and each row contains 
all of the data for one node.   

The column definitions are:

0. Column index in x of the split feature, -1 if leaf
1. Split threshold, 0.0 if leaf
2. Number of data points in this node
3. Number of positives in this node
4. Node number of this node
5. Node number of left child, -1 if this node is a leaf
6. Node number of right child, -1 if this node is a leaf 

For example, tree[1, 1] holds the split threshold for node number 1.
The nodes are numbered in [preorder](https://en.wikipedia.org/wiki/Tree_traversal#Pre-order) (root, left subtree, right subtree). 
Node numbers start at 0, which is the root.    

This data format is masked if you use the sklearn-compatible estimator.

## Accuracy

On the spam data from the [CASI book](https://web.stanford.edu/~hastie/CASI_files/data.html) we get this comparison:

    >> from sklearn.tree import DecisionTreeClassifier
    >> from sklearn.metrics import ConfusionMatrix as cm
    >> dt = DecisionTreeClassifier()
    >> dt.fit(xtr, ytr)
    >> sk_pred = dt.predict(xte)
    >> cm(yte, sk_pred)
    array([[861,  80],
       [ 71, 524]])

Versus SimpleTree:

    >> from simple_tree import SimpleTree
    >> st = SimpleTree()
    >> st.fit(xtr, ytr)
    >> st_pred = st.predict(xte)
    >> cm(yte, st_pred)
    array([[870,  71],
       [ 57, 538]])

...which is about the same. SimpleTree is deterministic, while DecisionTreeClassifier considers 
the features in random order, so they won't give exactly the same results.

## Performance

The spam data training set is of size 3065 x 57:

    >> %timeit st.fit(xtr, ytr)
    1 loop, best of 3: 707 ms per loop

Predicting on the spam test set (size 1536 x 57)

    >> %timeit st.predict(xte)
    1000 loops, best of 3: 405 µs per loop

So it fits a realistic, smaller data set in about 0.7s and prediction is quite fast.

## Tests
There is a small test suite in the `tests` directory. The tests require pytest and sklearn.
You can run the tests with:

    >> python -m pytest

from the project root or the tests directory.
