# -*- coding: utf-8 -*-
#
#  dataset.py
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

from portia.insilico.variable import Variable


def make_dataset(n, n_genes, n_tfs, ko=0, mko=0, sparsity=0.997):
    """Generate a GRN and gene expression dataset.

    Args:
        n (int): Number of gene expression measurements.
        n_genes (int): Total number of genes (GRN size).
        n_tfs (int): Number of putative transcripion factors.
            Should be lower than the number of genes.
        ko (int, optional): Number of single-gene KO experiments.
            The sum of `ko` and `mko` should not exceed `n`.
        mko (int, optional): Number of multiple-gene KO experiments.
            The sum of `ko` and `mko` should not exceed `n`.
        sparsity (float, optional): The percentage of gene pairs that
            do not form a regulatory relationship.

    Returns:
        X (np.ndarray): Gene expression matrix, of shape (n, n_genes).
        Z (np.ndarray): Binary matrix of shape (n, n_genes), indicating which genes
            have been knocked out in each experiment.
        A (np.ndarray): Goldstandard GRN, a binary matrix of shape (n, n_genes).
        tf_mask (np.ndarray): A binary vector indicating which genes are putative
            transcription factors.
    """
    assert n_tfs <= n_genes
    assert 0 < sparsity < 1
    assert 0 <= ko
    assert 0 <= mko
    assert ko + mko <= n
    A = (np.random.rand(n_genes, n_genes) < 1. - sparsity)
    np.fill_diagonal(A, 0)
    A[n_tfs:, :] = 0
    A[:n_tfs, :n_tfs] = np.triu(A[:n_tfs, :n_tfs])

    genes = [Variable() for _ in range(n_genes)]
    for i in range(n_tfs):
        for j in range(n_genes):
            if A[i, j]:
                coefficient = np.random.normal(0, 1)
                genes[j].add_parent(genes[i], coefficient)

    X = []
    Z = []
    for i in range(n):

        z = np.zeros(n_genes, dtype=bool)

        if ko > 0:
            j = np.random.randint(0, n_tfs)
            genes[j].knock_out()
            z[j] = 1
            ko -= 1
        elif mko > 0:
            idx = np.random.randint(0, n_tfs, size=10)
            for j in idx:
                genes[j].knock_out()
                z[j] = 1
            mko -= 1
        Z.append(z)

        X.append([gene.sample() for gene in genes])
        for gene in genes:
            gene.reset()
    X = np.nan_to_num(X)
    Z = np.asarray(Z, dtype=bool)
    tf_mask = np.zeros(n_genes, dtype=bool)
    tf_mask[:n_tfs] = True

    assert np.all(X >= 0)

    return X, Z, A, tf_mask
