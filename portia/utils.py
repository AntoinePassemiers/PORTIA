# -*- coding: utf-8 -*-
# utils.py
# author: Antoine Passemiers

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef


def plot_fp_types(ax, A, S, T, n_pred=300):
    _as = A.flatten()
    ys = S.flatten()
    idx = np.argsort(ys)[::-1][:n_pred]
    types = T.flatten()[idx]
    ys = ys[idx]
    _as = _as[idx]
    for i in range(len(ys)):
        if (not _as[i]) and not (types[i]):
            ax.bar(i, ys[i], width=1, color='black')
        elif types[i]:
            color = ['black', 'green', 'red', 'cyan', 'orange', 'purple'][types[i]]
            ax.bar(i, ys[i], width=1, color=color)
    ax.set_xlim(0, n_pred)
    plt.tick_params(labelleft=False)
    ax.set_yscale('log')
    ax.set(yticklabels=[" "])
    ax.axes.yaxis.set_ticklabels([" "])
    ax.axes.yaxis.set_visible(False)
