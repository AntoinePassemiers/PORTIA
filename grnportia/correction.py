# -*- coding: utf-8 -*-
# correction.py
# author: Antoine Passemiers

import numpy as np


def apply_correction(C, method='rcw', eps=1e-50, ignore_diag=True):
    is_square = (C.shape[0] == C.shape[1])
    C = np.copy(C)
    if is_square and ignore_diag:
        np.fill_diagonal(C, 0)
    if min(C.shape[0], C.shape[1]) < 2:
        return C
    else:
        if ignore_diag:
            n_nonzero_genes_i = np.sum(C != 0, axis=1)
            n_nonzero_genes_i = np.maximum(n_nonzero_genes_i, 1)
            n_nonzero_genes_j = np.sum(C != 0, axis=0)
            n_nonzero_genes_j = np.maximum(n_nonzero_genes_j, 1)
            Fi = C.sum(axis=1) / n_nonzero_genes_i
            Fj = C.sum(axis=0) / n_nonzero_genes_j
            F = C.sum() / (np.sum(n_nonzero_genes_i) + eps)
        else:
            Fi = np.mean(C, axis=1)
            Fj = np.mean(C, axis=0)
            F = np.mean(C)
        if method == 'apc':
            # Average product correction, as proposed by
            # S.D. Dunn et al.
            apc = Fi[:, np.newaxis] * Fj[np.newaxis, :]
            if F != 0:
                apc = apc / F
            C -= apc
        elif method == 'asc':
            # Average sum correction, as proposed by
            # S.D. Dunn et al.
            asc = Fi[:, np.newaxis] + Fj[np.newaxis, :] - F
            C -= asc
        else:
            # Row-column weighting
            rcw = Fi[:, np.newaxis] * Fj[np.newaxis, :]
            C /= (2. * (rcw + eps))
        return C
