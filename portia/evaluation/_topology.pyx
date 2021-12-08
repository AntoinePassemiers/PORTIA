# -*- coding: utf-8 -*-
# _topology.pyx
# author : Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport libc.math
from libc.stdlib cimport *
from libc.string cimport memcpy


cdef cnp.float32_t NAN = <cnp.float32_t>np.nan


def _evaluate(_A, _A_pred, _C, _CU, tf_mask):
    cdef cnp.uint8_t[:, :] C = np.asarray(_C, dtype=np.uint8)
    cdef cnp.uint8_t[:, :] CU = np.asarray(_CU, dtype=np.uint8)
    cdef cnp.uint8_t[:, :] A = np.asarray(_A, dtype=np.uint8)
    cdef cnp.uint8_t[:, :] A_pred = np.asarray(_A_pred, dtype=np.uint8)
    cdef cnp.uint8_t[:, :] T = np.zeros(_A.shape, np.uint8)
    cdef cnp.uint8_t[:] mask = np.asarray(tf_mask, dtype=np.uint8)
    cdef int n_genes = A.shape[0]
    cdef int i, j, k
    cdef int tp = 0
    cdef int total = 0
    cdef int fp_total = 0
    cdef int fp_spurious = 0
    cdef int fp_fork = 0
    cdef int n_fork = 0
    cdef int fp_chain = 0
    cdef int n_chain = 0
    cdef int fp_collider = 0
    cdef int n_collider = 0
    cdef int fp_chain_reversed = 0
    cdef int n_chain_reversed = 0
    cdef int fp_undirected = 0
    cdef bint found
    with nogil:

        for i in range(n_genes):
            if mask[i]:
                for j in range(n_genes):
                    if (not A[i, j]) and A_pred[i, j]:
                        fp_total += 1
                        total += 1

                        # Check chains
                        if C[i, j]:
                            fp_chain += 1
                            T[i, j] = 1
                            continue

                        # Check reversed chains
                        if C[j, i]:
                            fp_chain_reversed += 1
                            T[i, j] = 2
                            continue

                        # Check colliders
                        found = False
                        for k in range(n_genes):
                            if C[i, k] and C[j, k]:
                                found = True
                                break
                        if found:
                            fp_collider += 1
                            T[i, j] = 3
                            continue

                        # Check forks
                        found = False
                        for k in range(n_genes):
                            if mask[k]:
                                if C[k, i] and C[k, j]:
                                    found = True
                                    break
                        if found:
                            fp_fork += 1
                            T[i, j] = 4
                            continue

                        # Check undirected
                        if CU[i, j]:
                            fp_undirected += 1
                            T[i, j] = 5
                            continue

                        fp_spurious += 1
                        T[i, j] = 6
                        continue

                    elif A[i, j] and A_pred[i, j]:
                        total += 1
                        tp += 1

    if fp_total == 0:
        fp_total += 1
    return {
        'fork': fp_fork / float(total),
        'chain': fp_chain / float(total),
        'chain-reversed': fp_chain_reversed / float(total),
        'collider': fp_collider / float(total),
        'undirected': fp_undirected / float(total),
        'spurious': fp_spurious / float(total),
        'tp': tp / float(total),
        'total': fp_total,
        'T': np.asarray(T)
    }
