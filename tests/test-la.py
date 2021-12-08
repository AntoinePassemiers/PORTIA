# -*- coding: utf-8 -*-
# test-la.py
# author: Antoine Passemiers

import numpy as np
import pytest

from grnportia.la import partial_inv


def test_partial_inv():
    mat = np.asarray([
        [4.5, 0.1, -2.1, -1.2],
        [0.1, 7.3, 1.8, 0.4],
        [-2.1, 1.8, 5.4, -0.3],
        [-1.2, 0.4, -0.3, 4.2]])
    inv = np.linalg.inv(mat)

    idx = np.asarray([0, 1, 2, 3])
    np.testing.assert_array_almost_equal(partial_inv(mat, idx)[idx, :], inv[idx, :])

    idx = np.asarray([0, 1, 2])
    np.testing.assert_array_almost_equal(partial_inv(mat, idx)[idx, :], inv[idx, :])

    idx = np.asarray([0, 2, 3])
    np.testing.assert_array_almost_equal(partial_inv(mat, idx)[idx, :], inv[idx, :])

    idx = np.asarray([1, 2, 3])
    np.testing.assert_array_almost_equal(partial_inv(mat, idx)[idx, :], inv[idx, :])

    idx = np.asarray([0, 3])
    np.testing.assert_array_almost_equal(partial_inv(mat, idx)[idx, :], inv[idx, :])
