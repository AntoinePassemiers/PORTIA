# -*- coding: utf-8 -*-
# test-la.py
# author: Antoine Passemiers

import numpy as np
from sklearn.datasets import make_regression
import pytest

import portia as pt


def test_all_linregs():
    _lambda = 0.01
    X, _ = make_regression(n_targets=0)
    n_genes = X.shape[1]
    B1 = pt.la.all_linear_regressions_naive(X, _lambda=_lambda)
    B2 = pt.la.all_linear_regressions(X, _lambda=_lambda)
    np.testing.assert_array_almost_equal(B1, B2)


def test_partial_inv():
    mat = np.asarray([
        [4.5, 0.1, -2.1, -1.2],
        [0.1, 7.3, 1.8, 0.4],
        [-2.1, 1.8, 5.4, -0.3],
        [-1.2, 0.4, -0.3, 4.2]])
    inv = np.linalg.inv(mat)

    idx = np.asarray([0, 1, 2, 3])
    np.testing.assert_array_almost_equal(pt.la.partial_inv(mat, idx)[idx, :], inv[idx, :])

    idx = np.asarray([0, 1, 2])
    np.testing.assert_array_almost_equal(pt.la.partial_inv(mat, idx)[idx, :], inv[idx, :])

    idx = np.asarray([0, 2, 3])
    np.testing.assert_array_almost_equal(pt.la.partial_inv(mat, idx)[idx, :], inv[idx, :])

    idx = np.asarray([1, 2, 3])
    np.testing.assert_array_almost_equal(pt.la.partial_inv(mat, idx)[idx, :], inv[idx, :])

    idx = np.asarray([0, 3])
    np.testing.assert_array_almost_equal(pt.la.partial_inv(mat, idx)[idx, :], inv[idx, :])
