# -*- coding: utf-8 -*-
# test-ndcg.py
# author: Antoine Passemiers

import numpy as np

from portia.gt.evaluation import graph_theoretic_evaluation
from portia.gt.topology import _all_connected
from portia.gt.causal_structure import CausalStructure


def test_accessibility_matrix():
    G = np.asarray([
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
    C_target = np.asarray([
        [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
    tf_mask = np.ones(G.shape[0], dtype=bool)
    C = _all_connected(G, tf_mask)
    assert np.all(C == C_target)


def test_causal_structure_enum():
    assert CausalStructure.TRUE_POSITIVE == 1
    assert CausalStructure.CHAIN == 2
    assert CausalStructure.FORK == 3
    assert CausalStructure.CHAIN_REVERSED == 4
    assert CausalStructure.COLLIDER == 5
    assert CausalStructure.UNDIRECTED == 6
    assert CausalStructure.SPURIOUS_CORRELATION == 7
    assert CausalStructure.TRUE_NEGATIVE == 8
    assert CausalStructure.FALSE_NEGATIVE == 9


def test_ndcg():
    G_target = np.asarray([
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
    G_pred = np.asarray([
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=bool)
    T_target = np.asarray([
        [8, 1, 1, 2, 8, 5, 8, 8, 8, 9, 8],
        [8, 8, 9, 1, 8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8],
        [4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 1, 8, 8, 8, 8, 6, 8, 7],
        [8, 8, 8, 8, 9, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 1, 8, 9, 8, 8],
        [8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 8, 8, 9, 8, 8],
        [8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8]], dtype=int)
    res = graph_theoretic_evaluation(None, G_target, G_pred, tf_mask=None, method='auto')
    T = res['T']
    assert np.all(T == T_target)
