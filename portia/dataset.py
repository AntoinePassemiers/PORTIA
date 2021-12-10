# -*- coding: utf-8 -*-
# dataset.py
# author: Antoine Passemiers

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
        self.Z = []

    def add(self, experiment):
        self.X.append(experiment.expression)
        z = np.zeros(len(experiment.expression), dtype=bool)
        z[experiment.knockout] = 1
        self.Z.append(z)
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
        counts = np.zeros((self.n_genes, self.n_genes), dtype=np.int)

        # Compute statistics
        X = list()
        for experiment in self.knockout:
            X.append(experiment.expression)
        X = np.asarray(X)
        if mean is None:
            mean = np.mean(X, axis=0)
        if std is None:
            std = np.std(X, axis=0) + 1e-50

        # Get the set of genes that have been knocked-out at least once
        ko_genes = set()
        for experiment in self.knockout:
            gene_ids = experiment.knockout
            if len(gene_ids) == 1:
                ko_genes.add(list(gene_ids)[0])

        # Compute Z-scores
        for i in ko_genes:
            for experiment in self.knockout:
                gene_ids = experiment.knockout
                if (len(gene_ids) == 1) and (i in gene_ids):
                    S[i, :] += np.abs(experiment.expression - mean) / std
                    counts[i, :] += 1
        if np.sum(counts) > 0:
            mask = np.logical_or(counts == 0, counts.T == 0)
            S[~mask] = S[~mask] / counts[~mask]
            mean = np.mean(S[~mask])
            if mean != 0:
                S[~mask] /= mean
            else:
                S[mask] = 1
            S[mask] = np.quantile(S[~mask], 0.5)
            np.fill_diagonal(S, 0)
            if np.sum(S) <= 0:
                S = np.ones((self.n_genes, self.n_genes))
        else:
            S = np.ones((self.n_genes, self.n_genes))

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
