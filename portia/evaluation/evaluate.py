# -*- coding: utf-8 -*-
# evaluate.py
# author: Antoine Passemiers

from sklearn.metrics import matthews_corrcoef
import networkx as nx
from networkx.algorithms.connectivity.connectivity import node_connectivity
import numpy as np


def _evaluate(_A, _A_pred, _C, _CU, tf_mask):
    C = np.asarray(_C, dtype=np.uint8)
    CU = np.asarray(_CU, dtype=np.uint8)
    A = np.asarray(_A, dtype=np.uint8)
    A_pred = np.asarray(_A_pred, dtype=np.uint8)
    T = np.zeros(_A.shape, np.uint8)
    mask = np.asarray(tf_mask, dtype=np.uint8)
    n_genes = A.shape[0]
    tp = 0
    total = 0
    fp_total = 0
    fp_spurious = 0
    fp_fork = 0
    n_fork = 0
    fp_chain = 0
    n_chain = 0
    fp_collider = 0
    n_collider = 0
    fp_chain_reversed = 0
    n_chain_reversed = 0
    fp_undirected = 0

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


def create_edge_labels(A):
    labels = dict()
    for i, j in zip(*np.where(A > 0)):
        labels[(i, j)] = 1
    return labels


def evaluate(A, A_pred, tf_mask, with_kernels=False):
    n_genes = A.shape[0]

    results = dict()
    node_labels = {i: 1 for i in range(n_genes)}
    G1 = (A, node_labels, create_edge_labels(A))
    G2 = (A_pred, node_labels, create_edge_labels(A_pred))
    results['MCC'] = matthews_corrcoef(A.flatten(), A_pred.flatten())

    AU = np.maximum(A, A.T)

    G1 = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    G1U = nx.from_numpy_matrix(AU, create_using=nx.DiGraph)
    C = np.copy(A)
    CU = np.copy(A)
    for i in range(n_genes):
        for j in range(i):
            if not A[i, j]:
                C[i, j] = node_connectivity(G1, s=i, t=j)
            if not AU[i, j]:
                CU[i, j] = node_connectivity(G1U, s=i, t=j)
    results['MCC-indirect'] = matthews_corrcoef(C.flatten(), A_pred.flatten())
    results.update(_evaluate(A, A_pred, C, CU, tf_mask))

    return results


def find_best_threshold(y_test, y_proba):
    lb = np.min(y_proba)
    ub = np.max(y_proba)
    results = list()
    thresholds = np.linspace(lb, ub, 100)
    for alpha in thresholds:
        results.append(matthews_corrcoef(y_test, y_proba > alpha))
    return thresholds[np.argmax(results)]
