# -*- coding: utf-8 -*-
# test-ndcg.py
# author: Antoine Passemiers

import numpy as np

from portia.gt.evaluation import graph_theoretic_evaluation


def test_ndcg_random():
    n_genes = 10
    G_pred = np.random.rand(n_genes, n_genes)
    G_target = np.random.randint(0, 2, size=(n_genes, n_genes))
    graph_theoretic_evaluation(None, G_target, G_pred, tf_mask=None)
