# -*- coding: utf-8 -*-
#
#  core.py
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

import enum
import collections.abc
import warnings

import scipy.spatial
from sklearn.preprocessing import PowerTransformer, StandardScaler

from portia.correction import apply_correction
from portia.dataset import GeneExpressionDataset
from portia.exceptions import *
from portia.la import *


class Method(enum.Enum):

    # Fast power transforms
    FAST = enum.auto()

    # End-to-end inference of the optimal parameters for the power transforms
    END_TO_END = enum.auto()

    # No power transform
    NO_TRANSFORM = enum.auto()

    @staticmethod
    def from_string(name):
        formatted_name = name.strip().lower().replace('-', ' ').replace('_', ' ')
        if formatted_name == 'fast':
            return Method.FAST
        elif formatted_name == 'end to end':
            return Method.END_TO_END
        elif formatted_name == 'no transform':
            return Method.NO_TRANSFORM
        else:
            raise InvalidParameterError(
                f'Unknown method name "{name}". '
                f'Method should be "fast", "end-to-end" or "no-transform".')


def run(dataset, tf_idx=None, method='fast', lambda1=0.8, lambda2=0.05, n_iter=1000,
        normalize=True, return_sign=False, verbose=True):
    """Infers a Gene Regulatory Network (GRN) from gene expression data.

    The output network is a score matrix of shape (n_genes, n_genes).

    Args:
        dataset (:obj:`portia.GeneExpressionDataset`): Gene expression dataset.
        tf_idx (iterable): List of non-redundant 0-based indices.
            Each index in `tf_idx` corresponds to a potential transcription factor,
            and should be in the range [0, n_genes - 1]. The list should contain at
            most n_genes elements. If None, then all genes are considered as potentially
            regulating genes.
        method (str): Non-linear transformation of gene expression data.
            Should be either "fast", "end-to-end" or "no-transform".
        lambda1 (float): Shrinkage parameter for the estimation of the covariance matrix.
            The higher, the smaller the condition number of the covariance matrix will be.
        lambda2 (float): Tiknonov regularization rate for inferring directional information
            from gene expression exclusively.
        n_iter (int): Maximum number of iterations ("end-to-end" method only).
        normalize (bool): Whether to normalize each experiment separately or not.
            Normalization is based on the median.
        return_sign (bool): Whether to return the signs of inferred regulatory links.
        verbose (bool): Whether to print information to the standard output.

    Returns:
        :obj:`np.ndarray`: A real-valued estimation of the adjacency matrix of the GRN.
            The scores range from 0 to 1.
        :obj:`np.ndarray`: Returned only when `return_sign` is True. An integer matrix
            of same shape as the score matrix, where 1 indicates a positive regulatory relation,
            -1 a negative regulatory relation, and 0 no relation at all.
    """
    
    # Arbitrarily-small value used for numerical stability purposes
    epsilon = 1e-50

    # Check parameters
    method = Method.from_string(method)
    if not isinstance(dataset, GeneExpressionDataset):
        raise InvalidParameterError('Parameter "dataset" should be of type `portia.GeneExpressionDataset`.')
    if tf_idx is None:
        tf_idx = np.arange(0, dataset.n_genes)
    if not isinstance(tf_idx, collections.abc.Iterable):
        raise InvalidParameterError('Parameter "tf_idx" should be an iterable (e.g. list or array), or None.')
    tf_idx = np.asarray(tf_idx)
    try:
        _min = np.min(tf_idx)
        _max = np.max(tf_idx)
        if (_min < 0) or (_max >= dataset.n_genes):
            raise InvalidParameterError('Parameter "tf_idx" should contain indices in [0, n_genes - 1].')
        if len(np.unique(tf_idx)) != tf_idx.shape[0]:
            raise InvalidParameterError('Parameter "tf_idx" should not contain redundant indices.')
    except:
        raise InvalidParameterError('Parameter "tf_idx" should contain 0-based indices.')
    if not (0 <= lambda1 <= 1):
        raise InvalidParameterError('Shrinkage parameter "lambda1" should be in [0, 1].')
    if not (0 <= lambda2 <= 1):
        raise InvalidParameterError('Shrinkage parameter "lambda2" should be in [0, 1].')
    if not isinstance(normalize, bool):
        raise InvalidParameterError('Parameter "normalize" should be of boolean type.')
    if not isinstance(verbose, bool):
        raise InvalidParameterError('Parameter "verbose" should be of boolean type.')

    # Get expression data
    _X = dataset.asarray()
    n_samples = _X.shape[0]
    n_genes = _X.shape[1]
    if verbose:
        print(f'Gene expression matrix of shape ({n_samples}, {n_genes})')

    # Check missing values
    if np.any(np.isnan(_X)):
        raise NotImplementedError('Missing values are not supported (yet).')

    # Check the presence of negative values
    mask = np.any(_X < 0, axis=0)
    if np.any(mask):
        warnings.warn(
            'Expression data contain negative values. '
            'It is recommended to provide raw data instead.')
        _X[:, mask] -= (np.min(_X[:, mask], axis=0) - 1e-2)

    # Normalize experiments
    if normalize:
        quantiles = np.quantile(_X, 0.5, axis=0)[np.newaxis, :]
        quantiles[quantiles == 0] = 1
        _X = _X / quantiles

    # Compute sample weights (reduce redundancy)
    if n_samples >= n_genes:
        _P = 1. - scipy.spatial.distance.cdist(_X, _X, metric='correlation')
        theta = 0.8
        counts = np.zeros(n_samples)
        idx = np.where(_P > theta)
        np.add.at(counts, idx[0], 1)
        np.add.at(counts, idx[1], 1)
        weights = 1.0 / (1. + np.asarray(counts, dtype=np.float))
    else:
        weights = np.ones(n_samples) / n_samples

    # Box-Cox transform
    mask = np.asarray([(len(np.unique(_X[:, i])) > 1) for i in range(n_genes)], dtype=np.bool_)
    if method == Method.FAST:
        if np.sum(mask) > 0:
            power_transform = PowerTransformer(method='box-cox', standardize=True)
            _X_transformed = _X
            _X_transformed[:, mask] = power_transform.fit_transform(_X[:, mask] + epsilon)
        else:
            _X_transformed = _X + epsilon
    elif method == Method.END_TO_END:
        if np.sum(mask) > 0:
            _X_transformed = _X
            from portia.end_to_end import apply_optimal_transform
            _X_transformed[:, mask] = apply_optimal_transform(
                    _X[:, mask] + 1e-45, aweights=weights, _lambda=lambda1,
                    max_n_iter=100, verbose=verbose)
        else:
            _X_transformed = _X
    else:
        _X_transformed = _X

    # Centering and scaling
    scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
    _X_transformed = scaler.fit_transform(_X_transformed)

    # Compute empirical covariance matrix
    _S = np.cov(_X_transformed.T, ddof=1, aweights=weights)

    # Apply covariance matrix shrinkage
    _S_bar = lambda1 * np.eye(n_genes) + (1. - lambda1) * _S

    # Compute precision matrix
    if (tf_idx is not None) and (len(tf_idx) < 0.5 * n_genes):
        _Theta = partial_inv(_S_bar, tf_idx)
    else:
        _Theta = np.linalg.inv(_S_bar)

    # Gene bias correction
    _M = np.abs(_Theta)
    if tf_idx is not None:
        mask = np.zeros((n_genes, n_genes))
        mask[tf_idx, :] = 1
        np.fill_diagonal(mask, 0)
        _M *= mask
        _M[tf_idx, :] = apply_correction(_M[tf_idx, :], method='rcw')
    else:
        _M = apply_correction(_M, method='rcw')

    # No gene self-regulation
    np.fill_diagonal(_M, 0)

    # Break symmetry using linear regressions.
    # All OLS estimators are efficiently computed from a single
    # covariance matrix, using Sherman-Morrison formula.
    beta = all_linear_regressions(_X_transformed, _lambda=lambda2)
    beta = np.abs(beta)
    np.fill_diagonal(beta, 0)
    div = np.maximum(beta, beta.T)
    div_mask = (div > 0)
    beta[div_mask] /= div[div_mask]
    _M = _M * beta

    # Compute null-mutant Z-scores (if KOs are available)
    if dataset.has_ko():
        _F = dataset.compute_null_mutant_zscores()

        # Combine corrected precision matrix and Z-scores
        norm = np.linalg.norm(_F)
        if norm > 0:
            _F = _F / norm
        _M_bar = _M * _F ** 2
    else:
        _M_bar = _M

    # Promote scores from "hub" genes
    _M_bar *= np.std(_M_bar, axis=1)[:, np.newaxis]

    # Ensure that scores are in the range [0, 1]
    _range = _M_bar.max() - _M_bar.min()
    if _range > 0:
        _M_bar = (_M_bar - _M_bar.min()) / _range

    if return_sign:
        signs = np.sign(_Theta)
        np.fill_diagonal(_Theta, 0)
        return _M_bar, signs
    else:
        return _M_bar


def rank_scores(scores, gene_names, tf_names=None, limit=np.inf, diagonal=False):
    """Generator that iterates over top-scoring regulatory links by decreasing order of score.

    Args:
        scores (:obj:`np.ndarray`): A real-valued estimation of the adjacency matrix of the GRN.
            This can be, for example, the output of `portia.run` function.
        gene_names (iterable): List of gene names, in the same order as the corresponding 
            rows and columns in `scores`.
        tf_names (iterable, optional): List of regulating genes. Each gene name should necessarily
            be in `gene_names`. If None, then all genes are considered as potentially regulating genes.
        limit (int, optional): Maximum number of top-scoring links to generate.
        diagonal (bool, optional): Whether to include the diagonal of the score matrix.

    Yields:
        tuple: The next regulatory link to be reported. It is represented as a tuple containing
            three elements: the name of the regulating gene, the name of the regulated gene,
            and the corresponding score in the real-valued adjacency matrix.
    """

    if tf_names is None:
        tf_names = set(gene_names)

    idx = np.argsort(scores.flatten(order='C'))[::-1]
    idx = np.unravel_index(idx, scores.shape, order='C')

    n_predictions = 0

    for k in range(len(idx[0])):
        i = idx[0][k]
        j = idx[1][k]

        if gene_names[i] in tf_names:

            # Discard scores on diagonal if not needed
            if (not diagonal) and (i == j):
                continue

            if scores[i, j] > 0:
                yield gene_names[i], gene_names[j], scores[i, j]
                n_predictions += 1
                if n_predictions >= limit:
                    break
