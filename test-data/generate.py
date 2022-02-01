# -*- coding: utf-8 -*-
#
#  generate.py
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
import numpy as np

from portia.insilico.dataset import make_dataset


ROOT = os.path.dirname(os.path.abspath(__file__))


def save_expression_data(filepath, X, gene_names):
    assert X.shape[1] == len(gene_names)
    with open(filepath, 'w') as f:
        f.write('\t'.join(gene_names) + '\n')
        for i in range(X.shape[0]):
            f.write('\t'.join([str(x) for x in X[i, :]]) + '\n')


def save_goldstandard(filepath, A, gene_names, tf_mask):
    assert A.shape[0] == len(gene_names)
    with open(filepath, 'w') as f:
        for i in range(A.shape[0]):
            if tf_mask[i]:
                for j in range(A.shape[1]):
                    if not np.isnan(A[i, j]):
                        f.write(f'{gene_names[i]}\t{gene_names[j]}\t{int(A[i, j])}\n')


def save_tf_list(filepath, tf_mask, gene_names):
    assert tf_mask.shape[0] == len(gene_names)
    with open(filepath, 'w') as f:
        for i in range(len(gene_names)):
            if tf_mask[i]:
                f.write(f'{gene_names[i]}\n')


def main():

    # Dataset with no KO nor TF list
    n = 80
    n_genes = 100
    n_tfs = 100
    gene_names = [f'G{i + 1}' for i in range(n_genes)]
    X, _, A, tf_mask = make_dataset(n, n_genes, n_tfs, sparsity=0.95)
    save_expression_data(os.path.join(ROOT, 'dataset1.expression.txt'), X, gene_names)
    save_goldstandard(os.path.join(ROOT, 'dataset1.goldstandard.txt'), A, gene_names, tf_mask)

    # Dataset with TF list and multiple-genes KOs
    n = 80
    n_genes = 300
    n_tfs = 20
    ko = 5
    mko = 1
    gene_names = [f'G{i + 1}' for i in range(n_genes)]
    X, Z, A, tf_mask = make_dataset(n, n_genes, n_tfs, ko=ko, mko=mko, sparsity=0.95)
    save_expression_data(os.path.join(ROOT, 'dataset2.expression.txt'), X, gene_names)
    save_expression_data(os.path.join(ROOT, 'dataset2.kos.txt'), Z.astype(int), gene_names)
    save_goldstandard(os.path.join(ROOT, 'dataset2.goldstandard.txt'), A, gene_names, tf_mask)    
    save_tf_list(os.path.join(ROOT, 'dataset2.tfs.txt'), tf_mask, gene_names)


if __name__ == '__main__':
    main()
