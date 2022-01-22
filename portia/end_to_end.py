# -*- coding: utf-8 -*-
#
#  end_to_end.py
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
import torch

from portia.etel.box_cox import BoxCoxTransform
from portia.optim import LocalConvergence


def cov(X, ddof=0, aweights=None):
    """Differentiable covariance matrix estimation.

    Args:
        X (:obj:`torch.Tensor`): Matrix of observations, of shape (n_samples, n_genes).
        ddof (int, optional): When ddof=1, unbiased estimator is used.
        aweights (:obj:`torch.Tensor`): Array of sample weights, of shape (n_samples,).
            If None, all observations have equal weights.

    Returns:
        :obj:`torch.Tensor`: Estimated covariance matrix, of shape (n_genes, n_genes).
    """

    if (aweights is not None) and (not torch.is_tensor(aweights)):
        aweights = torch.DoubleTensor(aweights)

    if X.dim() == 1:
        X = X.view(-1, 1)

    if aweights is not None:
        weight_sum = torch.sum(aweights)
        mean = torch.sum(X * (aweights / weight_sum).unsqueeze(1), 0)
    else:
        mean = torch.mean(X, 0)

    if aweights is None:
        fact = X.shape[0] - ddof
    elif ddof == 0:
        fact = weight_sum
    elif aweights is None:
        fact = weight_sum - ddof
    else:
        sq_norm = torch.sum(aweights * aweights)
        fact = weight_sum - ddof * sq_norm / weight_sum

    centered = X.sub(mean.expand_as(X))

    if aweights is None:
        X_T = centered.t()
    else:
        X_T = torch.mm(torch.diag(aweights), centered).t()

    C = torch.mm(X_T, centered) / fact

    return C.squeeze()


def apply_optimal_transform(X, aweights=None, _lambda=0.8, max_n_iter=100, verbose=True):

    n_samples = X.shape[0]
    n_genes = X.shape[1]

    boxcox = BoxCoxTransform(n_genes)
    boxcox.init(X)
    X = torch.DoubleTensor(X)

    boxcox.bc_lambda_1.data = torch.clamp(boxcox.bc_lambda_1.data, -60, 60)

    optimizer = torch.optim.Adam([
        {'params': boxcox.parameters(), 'lr': 1e-2}
    ])
    convergence = LocalConvergence(5, max_n_iter=max_n_iter, tau=1e-7)

    if verbose:
        print('Starting gradient descent.')

    iteration = 1
    while not convergence():

        optimizer.zero_grad()

        X_prime, log_jac1 = boxcox(X)
        assert not np.isnan(X_prime.data.numpy()).any()

        C = cov(X_prime, aweights=aweights)
        C_reg = _lambda * torch.diag(torch.diagonal(C)) + (1. - _lambda) * C

        try:
            L = torch.linalg.cholesky(C_reg)
            Theta = torch.cholesky_inverse(L)
            logdet = -2. * torch.sum(torch.log(torch.diagonal(L)))
            # Theta = torch.inverse(C_reg)
            # logdet = torch.logdet(Theta)
        except RuntimeError:
            if verbose:
                print('Invalid value encountered. Falling back to sequential PORTIA.')
            boxcox.init(X)
            break

        ll = -0.5 * (n_samples - 1.) * torch.trace(torch.mm(Theta, C)) + 0.5 * n_samples * logdet
        # ll = -0.5 * n_samples * torch.trace(torch.mm(Theta, C)) + 0.5 * n_samples * logdet

        loss = -ll - log_jac1

        loss.backward()

        optimizer.step()
        boxcox.bc_lambda_1.data = torch.clamp(boxcox.bc_lambda_1.data, -60, 60)

        convergence.step(loss.item())

        if verbose:
            print(f'Loss at iteration {iteration}: {loss.item()}')
        iteration += 1

    boxcox.standardize = True
    X_prime, _ = boxcox(X)
    return X_prime.data.numpy()
