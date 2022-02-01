# -*- coding: utf-8 -*-
#
#  run.py: Run PORTIA as a standalone script
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

import argparse

import numpy as np
from matplotlib import pyplot as plt

import portia
from portia.gt import graph_theoretic_evaluation
from portia.insilico.dataset import make_dataset
from portia.exceptions import BadlyFormattedFile


def load_expression_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header = lines[0].rstrip()
    lines = lines[1:]
    gene_names = header.split('\t')
    X = []
    for line in lines:
        line = line.rstrip()
        elements = line.split('\t')
        if len(elements) > 1:
            X.append([float(value) for value in elements])
    return np.asarray(X), gene_names


def load_tf_idx(filepath, gene_names):
    tf_mask = np.zeros(len(gene_names), dtype=bool)
    gene_dict = {gene_name: i for i, gene_name in enumerate(gene_names)}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        if len(line) > 0:
            i = gene_dict[line]
            tf_mask[i] = True
    return np.where(tf_mask)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expression', type=str, help='Path to the expression data file')
    parser.add_argument('--tfs', type=str, help='Path to the file containing a list of putative TFs')
    parser.add_argument('--kos', type=str, help='Path to the KO data file (see examples)')
    parser.add_argument('--out', type=str, help='Output path')
    parser.add_argument('--method', choices=['fast', 'end-to-end', 'no-transform'], default='fast',
                        help='Data transformation method ("fast", "end-to-end" or "no-transform"')
    parser.add_argument('--signed', action='store_true', default=False,
                        help='Whether to return signed scores (enhancement vs. inhibition)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalise data based on the median')
    parser.add_argument('--lambda1', default=0.8, type=float,
                        help='Shrinkage parameter for covariance matrix estimation')
    parser.add_argument('--lambda2', default=0.05, type=float,
                        help='Shrinkage parameter for solving the linear regression problems')
    args = parser.parse_args()

    # Load gene expression data
    X, gene_names = load_expression_data(args.expression)

    # Load TF list
    if args.tfs:
        tf_idx = load_tf_idx(args.tfs, gene_names)
    else:
        tf_idx = None

    # Load KO information
    if args.kos:
        Z, gene_names2 = load_expression_data(args.kos)
        Z = Z.astype(bool)
        if gene_names != gene_names2:
            raise BadlyFormattedFile(
                'Gene names lists should be identical in expression and KO data files')
    else:
        Z = np.zeros(X.shape, dtype=bool)

    # Run PORTIA
    dataset = portia.GeneExpressionDataset()
    for exp_id, data in enumerate(X):
        dataset.add(portia.Experiment(exp_id, data, knockout=np.where(Z[exp_id, :])[0]))
    res = portia.run(dataset, tf_idx=tf_idx, method='fast', lambda1=args.lambda1,
        lambda2=args.lambda2, n_iter=100, normalize=args.normalize,
        return_sign=args.signed, verbose=False)
    if args.signed:
        M_bar, sign = res
        M_bar = M_bar * sign
    else:
        M_bar = res

    out_filepath = args.out if args.out else 'out.txt'
    with open(out_filepath, 'w') as f:
        for gene_a, gene_b, score in portia.rank_scores(M_bar, gene_names):
            f.write(f'{gene_a}\t{gene_b}\t{score}\n')


if __name__ == '__main__':
    main()
