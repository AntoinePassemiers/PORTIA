# -*- coding: utf-8 -*-
# test-grn-inference.py
# author: Antoine Passemiers

import numpy as np
import pytest

import portia
from portia.core import partial_inv, rank_scores


def test_run():
    dataset = portia.GeneExpressionDataset()
    dataset.add(portia.Experiment(0, [4.1, 0.2, 0, 3.4], knockout=[2]))
    dataset.add(portia.Experiment(1, [2.3, 0.25, 0.1, 2.1]))
    dataset.add(portia.Experiment(2, [6.1, 3.2, 0.003, 3.1]))

    scores = portia.run(dataset, tf_idx=None, method='fast', verbose=False)
    assert not np.any(np.isnan(scores))
    assert np.all(np.logical_and(scores >= 0, scores <= 1))
    assert len(np.unique(scores)) > 1

    scores = portia.run(dataset, tf_idx=None, method='end-to-end', verbose=False)
    assert not np.any(np.isnan(scores))
    assert np.all(np.logical_and(scores >= 0, scores <= 1))
    assert len(np.unique(scores)) > 1


def test_numerical_stability():
    dataset = portia.GeneExpressionDataset()
    dataset.add(portia.Experiment(0, [3, 3, 3, 3]))
    dataset.add(portia.Experiment(1, [3, 3, 3, 3]))
    dataset.add(portia.Experiment(2, [3, 3, 3, 3]))

    scores = portia.run(dataset, tf_idx=None, method='fast', verbose=False)
    assert not np.any(np.isnan(scores))
    assert np.all(np.logical_and(scores >= 0, scores <= 1))

    scores = portia.run(dataset, tf_idx=None, method='end-to-end', verbose=False)
    assert not np.any(np.isnan(scores))
    assert np.all(np.logical_and(scores >= 0, scores <= 1))


def test_score_ranking():
    scores = np.asarray([
        [0.45, 0, 0.21, 0.12],
        [0.30, 0.73, 0.18, 0],
        [0.20, 0.17, 0.18, 0.24],
        [0, 0, 0, 0.42]])
    gene_names = ['G1', 'G2', 'G3', 'G4']
    gen = rank_scores(scores, gene_names, tf_names=None, limit=5)
    assert next(gen) == ('G2', 'G1', 0.30)
    assert next(gen) == ('G3', 'G4', 0.24)
    assert next(gen) == ('G1', 'G3', 0.21)
    assert next(gen) == ('G3', 'G1', 0.20)
    assert next(gen) == ('G2', 'G3', 0.18)
    with pytest.raises(StopIteration):
        next(gen)

    gen = rank_scores(scores, gene_names, tf_names=['G1', 'G2'], limit=6)
    assert next(gen) == ('G2', 'G1', 0.30)
    assert next(gen) == ('G1', 'G3', 0.21)
    assert next(gen) == ('G2', 'G3', 0.18)
    assert next(gen) == ('G1', 'G4', 0.12)
    with pytest.raises(StopIteration):
        next(gen)

    gen = rank_scores(scores, gene_names, tf_names=None, limit=5, diagonal=True)
    assert next(gen) == ('G2', 'G2', 0.73)
    assert next(gen) == ('G1', 'G1', 0.45)
    assert next(gen) == ('G4', 'G4', 0.42)
    assert next(gen) == ('G2', 'G1', 0.30)
    assert next(gen) == ('G3', 'G4', 0.24)
    with pytest.raises(StopIteration):
        next(gen)
