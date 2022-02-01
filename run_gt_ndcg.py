# -*- coding: utf-8 -*-
#
#  run_gt_ndcg.py: Graph-Theoretic Normalised Discounted Cumulative Gain
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
import argparse
import numpy as np
import matplotlib.pyplot as plt

import portia
from portia.gt.causal_structure import CausalStructure
from portia.gt.grn import GRN
from portia.gt.evaluation import graph_theoretic_evaluation, plot_fp_types


def load_gene_names(filepath):
    gene_names = set()
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        elements = line.rstrip().split('\t')
        if len(elements) >= 2:
            gene_names.add(elements[0])
            gene_names.add(elements[1])
    return gene_names


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
    parser.add_argument('inferred', type=str, help='Path to the inferred GRN file')
    parser.add_argument('goldstandard', type=str, help='Path to the goldstandard GRN file')
    parser.add_argument('--tfs', type=str, help='Path to the file containing a list of putative TFs')
    parser.add_argument('--out', type=str, help='Output folder where to save results')
    args = parser.parse_args()

    gene_names = load_gene_names(args.inferred)
    gene_names.union(load_gene_names(args.goldstandard))
    gene_names = list(gene_names)

    # Load TF list
    if args.tfs:
        tf_idx = load_tf_idx(args.tfs, gene_names)
    else:
        tf_idx = np.arange(len(gene_names))
    tf_mask = np.zeros(len(gene_names), dtype=bool)
    tf_mask[tf_idx] = True

    G_target = GRN.load_network(args.goldstandard, gene_names, tf_idx)
    G_pred = GRN.load_network(args.inferred, gene_names, tf_idx)

    res = graph_theoretic_evaluation(None, G_target, G_pred, tf_mask=tf_mask, method='top')

    if args.out is None:
        out_folder = '.'
    else:
        out_folder = args.out
        os.makedirs(out_folder)

    T = res['T']
    s  = f'Number of TPs: {np.sum(T == int(CausalStructure.TRUE_POSITIVE))}\n'
    s += f'Number of FPs in chains: {np.sum(T == int(CausalStructure.CHAIN))}\n'
    s += f'Number of FPs in forks: {np.sum(T == int(CausalStructure.FORK))}\n'
    s += f'Number of FPs in reserved chains: {np.sum(T == int(CausalStructure.CHAIN_REVERSED))}\n'
    s += f'Number of FPs in colliders: {np.sum(T == int(CausalStructure.COLLIDER))}\n'
    s += f'Number of FPs in undirected paths: {np.sum(T == int(CausalStructure.UNDIRECTED))}\n'
    s += f'Number of FPs due to spurious correlations only: {np.sum(T == int(CausalStructure.SPURIOUS_CORRELATION))}\n'
    s += '\n'
    s += f'Number of TNs: {np.sum(T == int(CausalStructure.TRUE_NEGATIVE))}\n'
    s += f'Number of FNs: {np.sum(T == int(CausalStructure.FALSE_NEGATIVE))}\n'
    s += '\n'
    s += f'Overall gt-NDCG score: {res["score"]}'
    print(s)
    with open(os.path.join(out_folder, 'results.txt'), 'w') as f:
        f.write(s)

    ax = plt.subplot(1, 1, 1)
    plot_fp_types(ax, G_target, G_pred, T, tf_mask=tf_mask, n_pred=G_target.n_edges,
        remove_labels=False, legend=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'false-positives.png'))


if __name__ == '__main__':
    main()
