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


class Experiment:

    def __init__(self, _id, expression, knockout=None, knockdown=None, time=0):
        if knockdown is None:
            knockdown = []
        if knockout is None:
            knockout = []
        self._id = _id
        self.expression = expression
        self.knockout = knockout
        self.knockdown = knockdown
        self.time = time

    @property
    def id(self):
        return self._id

    @property
    def n_genes(self):
        return len(self.expression)


class GeneExpressionDataset:

    def __init__(self):
        self.experiments = dict()
        self.knockout = []
        self.knockdown = []
        self.X = []
        self.K = []

    def add(self, experiment):
        assert isinstance(experiment, Experiment)
        self.X.append(experiment.expression)
        k = np.zeros(len(experiment.expression), dtype=bool)
        k[experiment.knockout] = 1
        self.K.append(k)
        if self.n_genes > 0:
            assert self.n_genes == experiment.n_genes
        if experiment.id not in self.experiments:
            self.experiments[experiment.id] = list()
        if len(self.experiments[experiment.id]) > 0:
            assert(self.experiments[experiment.id][0].time <= experiment.time)
        self.experiments[experiment.id].append(experiment)
        
        # Add to knockout experiments
        gene_ids = tuple(experiment.knockout)
        if len(gene_ids) > 0:
            self.knockout.append(experiment)

        # Add to knockdown experiments
        gene_ids = tuple(experiment.knockdown)
        if len(gene_ids) > 0:
            self.knockdown.append(experiment)

    def compute_null_mutant_zscores(self, mean=None, std=None):

        # Knock-out experiments
        S = np.zeros((self.n_genes, self.n_genes))
        counts = np.zeros((self.n_genes, self.n_genes), dtype=int)

        # Compute statistics
        X = []
        for experiment in self.knockout:
            X.append(experiment.expression)
        X = np.asarray(X)
        if mean is None:
            if len(X) > 0:
                mean = np.mean(X, axis=0)
            else:
                mean = np.zeros(self.n_genes)
        if std is None:
            if len(X) > 0:
                std = np.std(X, axis=0) + 1e-50
            else:
                std = np.ones(self.n_genes)

        # Compute Z-scores
        for experiment in self.knockout:
            gene_ids = experiment.knockout
            if len(gene_ids) == 1:
                i = gene_ids[0]
                S[i, :] += np.abs(experiment.expression - mean) / std
                counts[i, :] += 1
        if np.sum(counts) > 0:
            mask = np.logical_or(counts == 0, counts.T == 0)
            S[~mask] = S[~mask] / counts[~mask]
            mean = np.mean(S[~mask])
            if mean == 0:
                S[~mask] = 1
            else:
                S[~mask] /= np.median(S[~mask])
            S[mask] = 1
            if np.sum(S) <= 0:
                S = np.ones((self.n_genes, self.n_genes))
        else:
            S = np.ones((self.n_genes, self.n_genes))

        np.fill_diagonal(S, 0)
        return S

    def asarray(self):
        return np.asarray(self.X)

    @property
    def n_samples(self):
        return len(self.X)

    @property
    def n_features(self):
        if len(self.X) == 0:
            return 0
        else:
            return len(self.X[0])

    @property
    def n_genes(self):
        return self.n_features

    def has_ko(self):
        return len(self.knockout) > 0
