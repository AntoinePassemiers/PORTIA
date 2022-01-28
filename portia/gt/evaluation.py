# -*- coding: utf-8 -*-
#
#  evaluation.py
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

import os
import sys

import numpy as np

from portia.gt.causal_structure import CausalStructure
from portia.gt.grn import GRN
from portia.gt.utils import nan_to_min
from portia.gt.topology import _evaluate, _all_connected


def graph_theoretic_evaluation(filepath, G_target, G_pred, tf_mask=None, method='top'):

    method = method.strip().lower()
    assert method in {'top', 'auto'}

    # Check TF mask
    if tf_mask is None:
        n_genes = len(G_target)
        tf_mask = np.ones(n_genes, dtype=bool)

    # Ensure GRNs
    if not isinstance(G_target, GRN):
        tf_idx = np.where(tf_mask)[0]
        G_target = GRN(G_target, tf_idx)
    if not isinstance(G_pred, GRN):
        tf_idx = np.where(tf_mask)[0]
        G_pred = GRN(G_pred, tf_idx)

    # Goldstandard adjacency matrix
    A = np.copy(G_target.asarray())

    # Goldstandard undirected (symmetric) adjacency matrix
    AU = G_target.as_symmetric().asarray()

    # Convert predicted scores to binary adjacency matrix
    if method == 'top':
        use_top = True
    else:
        use_top = (len(np.unique(G_target.asarray())) > 2)
    if use_top:
        n_edges = G_target.n_edges
        A_binary_pred = G_pred.binarize(n_edges).asarray()
        assert int(np.sum(A_binary_pred)) == n_edges
    else:
        A_binary_pred = (G_pred.asarray() >= np.max(G_pred.asarray()))
        n_edges = np.sum(A_binary_pred)

    # Fill missing values
    np.nan_to_num(A, nan=0, copy=False)
    np.nan_to_num(AU, nan=0, copy=False)
    A_binary_pred = nan_to_min(A_binary_pred)

    if (filepath is not None) and os.path.exists(filepath):
        data = np.load(filepath)
        C = data['C']
        CU = data['CU']
    else:
        # Find all pairs of connected vertices
        C = _all_connected(A, tf_mask)

        # Find all pairs of connected vertices (undirected edges)
        CU = _all_connected(AU, np.ones(AU.shape[0]))

        # If `i` indirectly regulates `j`, then an undirected regulatory relationship
        # exists between `j` and `i`, and vice versa
        # This step is necessary because we didn't compute node connectivity when tf_mask[i]
        # is False, leading to missing values in `CU`
        CU = np.maximum(CU, CU.T)

        # Save regulatory relationship matrices
        if filepath is not None:
            np.savez(filepath, C=C, CU=CU)
    assert np.all(CU >= C)

    # No self-regulation
    np.fill_diagonal(A, 0)
    np.fill_diagonal(C, 0)
    np.fill_diagonal(CU, 0)

    print(A.astype(int))
    print(A_binary_pred.astype(int))

    # Categorise predictions based on local causal structures
    results = {'T': _evaluate(A, A_binary_pred, C, CU, tf_mask)}

    # Convert the matrix of causal structures into a relevance matrix
    relevance = CausalStructure.array_to_relevance(results['T'])

    # Filter predictions and keep only meaningful ones:
    # no self-regulation, and no prediction for TFs for which there is no
    # experimental evidence, based on the goldstandard
    mask = G_target.get_mask()
    y_relevance = relevance[mask]
    y_pred = G_pred[mask]

    # Fill missing predictions with the lowest score.
    # For fair comparison of the different GRN inference methods,
    # the same number of predictions should be reported.
    # If a method reports less regulatory links than what is present
    # in the goldstandard, missing values will be consumed until
    # the right number of predictions (`n_edges`) is reached.
    y_pred = nan_to_min(y_pred)

    # Get the indices of the first `n_edges` predictions,
    # sorted by decreasing order of importance
    idx = np.argsort(y_pred)[-n_edges:][::-1]

    # Compute importance weights
    weights = 1. / np.log2(1. + np.arange(1, len(idx) + 1))

    # Discounted Cumulative Gain
    dcg = np.sum(weights * y_relevance[idx])

    # Ideal Discounted Cumulative Gain
    idcg = 4. * np.sum(weights)

    # Normalized Discounted Cumulative Gain (ranges between 0 and 1)
    ndcg = dcg / idcg

    # Store NDCG score
    results['score'] = ndcg
    return results


def plot_fp_types(ax, G_target, G_pred, T, tf_mask=None, n_pred=300):
    import matplotlib.pyplot as plt  # pylint: disable=import-error

    # Check TF mask
    if tf_mask is None:
        n_genes = len(G_target)
        tf_mask = np.ones(n_genes, dtype=bool)

    # Ensure GRNs
    n_genes = len(G_target)
    if not isinstance(G_target, GRN):
        tf_idx = np.arange(n_genes)
        G_target = (G_target, tf_idx)
    if not isinstance(G_pred, GRN):
        tf_idx = np.arange(n_genes)
        G_target = (G_pred, tf_idx)

    n_pred = int(n_pred)
    n_links = G_target.n_edges

    plt.rcParams['figure.figsize'] = [12, 4]

    A = G_target.asarray()
    S = G_pred.asarray()

    mask = G_target.get_mask()
    _as = A[mask]
    ys = np.nan_to_num(S[mask], nan=0)
    types = T[mask]

    idx = np.argsort(ys)[::-1][:n_pred]
    types = types[idx]
    ys = ys[idx]
    _as = _as[idx]
    assert not np.any(np.isnan(types))
    assert not np.any(np.isnan(ys))
    assert not np.any(np.isnan(_as))
    for i in range(len(ys)):
        if not _as[i]:
            if types[i] > 0:
                if types[i] == CausalStructure.CHAIN:
                    color = 'green'
                elif types[i] == CausalStructure.FORK:
                    color = 'orange'
                elif types[i] == CausalStructure.CHAIN_REVERSED:
                    color = 'red'
                elif types[i] == CausalStructure.COLLIDER:
                    color = 'cyan'
                elif types[i] == CausalStructure.UNDIRECTED:
                    color = 'purple'
                elif types[i] == CausalStructure.SPURIOUS_CORRELATION:
                    color = 'black'
                elif types[i] == CausalStructure.TRUE_POSITIVE:
                    color = 'grey'
                else:
                    color = 'none'
                if color != 'none':
                    ax.bar(i, ys[i], width=0.4, color=color)
        if i + 1 >= n_links:
            break
    # plt.axvline(x=n_links, linewidth=1, linestyle='--', color='gray', alpha=0.5)
    plt.tick_params(labelleft=False)
    ax.set_yscale('log')
    ax.set(yticklabels=[" "])
    ax.axes.yaxis.set_ticklabels([" "])
    ax.axes.yaxis.set_visible(False)
