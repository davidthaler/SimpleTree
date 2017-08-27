'''
A few tests to verify behavior of the simple_splitter.
NB: the splitter also gets excersized by the tree tests on real data (irises).

To run:

    >>> python -m pytest

or:

    >>> python -m pytest test_splitter.py

...in this directory

author: David Thaler
date: August 2017
'''
import pytest
import numpy as np
from simple_tree.simple_splitter import split

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

def test_one_row():
    # Test for no split, but also no error, if only one row in data.
    result = split(x[:1], y[:1])
    assert(result == (-1, 0.0))

def test_pure_node():
    # Test for no split, but also no error, if node is pure.
    result = split(x[:2], y[:2])
    assert(result == (-1, 0.0))

def test_uniform_x():
    # Test for no split, but also no error, if x has only one value.
    result = split(x[:, :1], y)
    assert(result == (-1, 0.0))

def test_no_improve():
    # Test for no split, but also no error, no improvement possible.
    result = split(x[:, 1:2], y)
    assert(result == (-1, 0.0))

def test_perfect():
    # Find perfect split in column 4 with threshold of 2.5
    result = split(x, y)
    assert(result == (4, 2.5))

def test_imperfect():
    # Find first of 2 (equivalent) imperfect splits in columns 2 and 3.
    result = split(x[:, :-1], y)
    assert(result == (2, 3.5))
