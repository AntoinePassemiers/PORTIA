# -*- coding: utf-8 -*-
# la.py: Linear algebra
# author: Antoine Passemiers

import numpy as np


def partial_inv(mat, tf_idx):
    n_rows = mat.shape[0]
    _Theta = np.zeros((n_rows, n_rows), dtype=mat.dtype)
    _J = np.eye(n_rows)[:, tf_idx]
    _Theta[tf_idx, :] = np.linalg.solve(mat, _J).T
    return _Theta
