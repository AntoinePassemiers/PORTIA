# -*- coding: utf-8 -*-
# la.py: Linear algebra
# author: Antoine Passemiers

import numpy as np


def all_linear_regressions(X, _lambda=0.05):
    n_genes = X.shape[1]
    S = _lambda * np.eye(n_genes) + (1. - _lambda) * np.cov(X.T)
    T = np.linalg.inv(S)
    B = np.zeros((n_genes, n_genes))
    u = np.zeros(n_genes)
    for j in range(n_genes):

        # Remove i-th row of covariance matrix using Sherman-Morrison formula
        u[:] = 0
        u[j] = 1
        v = -S[:, j]
        v[j] = -(S[j, j] - 1)
        aiu = np.dot(T, u)
        vai = np.dot(v, T)
        T_j = T - np.outer(aiu, vai) / (1. + np.dot(v, aiu))

        # Remove i-th column of covariance matrix using Sherman-Morrison formula
        v[j] = 0
        aiu = np.dot(T_j.T, u)
        vai = np.dot(v, T_j.T)
        T_j = T_j - np.outer(aiu, vai) / (1. + np.dot(v, aiu))

        # Remove j-th variable (explained variable) from the set of explanatory variables
        T_j[:, j] = 0
        T_j[j, :] = 0

        # Compute OLS estimator (see Gauss-Markov theorem)
        B[:, j] = T_j.dot(X.T).dot(X[:, j])
    np.fill_diagonal(B, 0)
    return B


def all_linear_regressions_naive(X, _lambda=0.8):
    n_genes = X.shape[1]
    beta = []
    for j in range(n_genes):
        mask = np.ones(n_genes, dtype=bool)
        mask[j] = 0
        X_j = X[:, mask]
        y_j = X[:, j]
        cov = _lambda * np.eye(n_genes - 1) + (1. - _lambda) * np.cov(X_j.T)
        w = np.zeros(n_genes)
        w[mask] = np.linalg.inv(cov) @ X_j.T @ y_j
        assert not np.any(np.isnan(w))
        beta.append(w)
    beta = np.asarray(beta).T
    np.fill_diagonal(beta, 0)
    return beta


def partial_inv(mat, tf_idx):
    n_rows = mat.shape[0]
    _Theta = np.zeros((n_rows, n_rows), dtype=mat.dtype)
    _J = np.eye(n_rows)[:, tf_idx]
    _Theta[tf_idx, :] = np.linalg.solve(mat, _J).T
    return _Theta
