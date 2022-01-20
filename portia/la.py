# -*- coding: utf-8 -*-
#
#  la.py
#  
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import numpy as np


def all_linear_regressions(X, _lambda=0.05):
    """
    aiu2 = u^T @ T_j
         = u^T @ T - u^T @ aiun @ vai^T
         = (T^T @ u)^T - u^T @ aiun @ vai^T
         = aiu - u^T @ aiun @ vai^T
    vai2 = T_j @ v
         = T @ v - aiun @ vai^T @ v
         = vai - aiun @ vai^T @ v
    """
    n_genes = X.shape[1]
    S = _lambda * np.eye(n_genes) + (1. - _lambda) * np.cov(X.T)
    T = np.linalg.inv(S)
    B = T @ X.T @ X
    u = np.zeros(n_genes)

    V = -S - np.eye(n_genes)
    VAI = np.dot(V.T, T)
    den1 = 1. + np.sum(V * T, axis=0)

    B -= T * (np.sum(np.dot(X, VAI) * X, axis=0) / den1)[np.newaxis, :]
    AIU2 = T - VAI * (np.diag(T) / den1)[np.newaxis, :]
    VAIV = np.sum(VAI * V, axis=0)
    XAIU2X = np.sum(np.dot(X, AIU2) * X, axis=0)

    B -= (VAI - T * (VAIV / den1)[np.newaxis, :]) * (XAIU2X / (1. + np.sum(V * AIU2, axis=0)))[np.newaxis, :]

    np.fill_diagonal(B, 0)
    return B


def all_linear_regressions_slow(X, _lambda=0.05):
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
        T_j = (T_j.T - np.outer(aiu, vai) / (1. + np.dot(v, aiu))).T

        # Remove j-th variable (explained variable) from the set of explanatory variables
        T_j[:, j] = 0
        T_j[j, :] = 0
        T_j[j, j] = 0

        # Compute OLS estimator (see Gauss-Markov theorem)
        B[:, j] = T_j.dot(X.T).dot(X[:, j])
    np.fill_diagonal(B, 0)
    return B


def all_linear_regressions_naive(X, _lambda=0.05):
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
