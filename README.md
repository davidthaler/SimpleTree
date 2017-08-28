# SimpleTree
SimpleTree is an all-python decision tree classifier with key sections accelerated using Numba.

## Goals
This project aims to produce a decision tree classifier that is:   

* readable and understandable by average programmers
* hackable by average programmers
* as accurate as scikit-learn DecisionTreeClassifier
* fast enough to actually be used (within 2x of scikit-learn)

## Requirements
* [Numba](http://numba.pydata.org/numba-doc/dev/index.html) Numba is a jit compiler for python that we use to accelerate key code sections.
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
    pred = predict(x_test)

## Design
The code here has three parts:

* splitter: `split()` in `simple_splitter.py` takes in data and returns the best feature and threshold to split on.
* tree builder: `simple_tree_builder.py` contains `build_tree()`, which builds the tree recursively.
* sklearn interface: `simple_tree.py` contains a simple class to make an scikit-learn 
compatible estimator from this code.

All of the code is in python.
Numba is used to accelerate the splitter and the apply function.
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

## Performance

On my machine, fitting an sklearn DecisionTreeClassifier on the iris data gives:

    >> dt = DecisionTreeClassifier()
    >> %timeit dt.fit(x, y)
    1000 loops, best of 3: 336 µs per loop

Using SimpleTree :

    >> st = SimpleTree()
    >> %timeit st.fit(x, y)
    1000 loops, best of 3: 499 µs per loop

Using the `build_tree` function gives:

    >> %timeit build_tree(x, y)
    1000 loops, best of 3: 520 µs per loop

So SimpleTree runs in about 1.5x the running time of sklearn's DecisionTreeClassifier on this data.

## Numba
Numba is a just-in-time compiler (jit) for python code.
It supports a only subset of standard python and the pydata stack (numpy/scipy/etc. ...), 
so it can't be used everywhere.
See numba's doc on [python suppport](http://numba.pydata.org/numba-doc/dev/reference/pysupported.html) for the standard language support
Unlike [pypy](https://pypy.org/), numba does support numpy arrays and a large number of numpy functions. 
See numba's doc on [numpy suppport](http://numba.pydata.org/numba-doc/dev/reference/numpysupported.html) for what parts are/are not supported.
Support for recursion and OOP is experimental/unstable.
This project uses it to accelerate key sections of the code, like the splitter.
Numba can be difficult to get installed. 
I recommend using the [Anaconda](https://docs.continuum.io/anaconda/) python distribution, which includes it.

## Tests
There is a small test suite in the `tests` directory. The tests require pytest and sklearn.
You can run the tests with:

    >> python -m pytest

from the project root or the tests directory.
