# -*- coding: utf-8 -*-
#
#  topology.py
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

from portia.gt.causal_structure import CausalStructure


def _all_connected(A, tf_mask):
    """Naive algorithm for finding all pairs of connected vertices.

    The performance of this algorithm will strongly depend on the topology
    of the input graph, but it proved to be quite efficient in practice.

    Note: possible improvement, compute accessibility matrix (e.g. with Floyd-Warshall algorithm)
    """
    C_old = np.copy(np.asarray(A, dtype=np.uint8))
    while True:
        C_new = np.logical_or(C_old, np.dot(C_old, C_old))
        if np.all(C_new == C_old):
            break
        C_old = C_new
    return C_new


def _evaluate(A, A_pred, C, CU, tf_mask):
    A = np.asarray(A, dtype=bool)
    A_pred = np.asarray(A_pred, dtype=bool)
    C = np.asarray(C, dtype=bool)
    CU = np.asarray(CU, dtype=bool)
    T = np.zeros(A.shape, np.uint8)
    n_genes = A.shape[0]

    assert A.shape == (n_genes, n_genes)
    assert A_pred.shape == (n_genes, n_genes)
    assert C.shape == (n_genes, n_genes)
    assert CU.shape == (n_genes, n_genes)
    assert tf_mask.shape == (n_genes,)

    mask = np.logical_and(A_pred, ~A)

    M = C
    T[np.logical_and(mask, M)] = CausalStructure.CHAIN
    np.logical_and(mask, ~M, out=mask)

    M = C.T
    T[np.logical_and(mask, M)] = CausalStructure.CHAIN_REVERSED
    np.logical_and(mask, ~M, out=mask)

    M = np.dot(C.T, C).astype(bool)
    T[np.logical_and(mask, M)] = CausalStructure.FORK
    np.logical_and(mask, ~M, out=mask)

    M = np.dot(C, C.T).astype(bool)
    T[np.logical_and(mask, M)] = CausalStructure.COLLIDER
    np.logical_and(mask, ~M, out=mask)

    M = CU
    T[np.logical_and(mask, M)] = CausalStructure.UNDIRECTED
    np.logical_and(mask, ~M, out=mask)

    T[mask] = CausalStructure.SPURIOUS_CORRELATION

    T[~tf_mask, :] = CausalStructure.TRUE_NEGATIVE
    return T
