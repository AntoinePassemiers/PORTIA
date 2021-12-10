# -*- coding: utf-8 -*-
# end_to_end.py
# author: Antoine Passemiers

import numpy as np
import torch
from sklearn.preprocessing import PowerTransformer

from portia.optim import LocalConvergence


def cov(X, ddof=0, aweights=None):
    """Differentiable covariance matrix estimation.

    Args:
        X (:obj:`np.ndarray`): Matrix of observations, of shape (n_samples, n_genes).
        ddof (int, optional): When ddof=1, unbiased estimator is used.
        aweights (:obj:`np.ndarray`): Array of sample weights, of shape (n_samples,).
            If None, all observations have equal weights.

    Returns:
        :obj:`portia.GeneExpressionDataset`: Estimated covariance matrix, of shape (n_genes, n_genes).
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


class BoxCoxTransform(torch.nn.Module):
    """Differentiable Box-Cox transform.

    Attributes:
        n_variables (int): Number of genes.
        epsilon (float): Arbitrarily-small value used for numerical
            stability purposes.
        bc_lambda_1 (:obj:`torch.nn.Parameter`): Vector of Box-Cox parameters, where i-th
            component corresponds to the parameter of the transform
            associated to gene i.
        standardize (bool): Whether to center and scale the transformed data.
        _mu (:obj:`torch.nn.Parameter`): Mean vector, where i-th component is the average
            expression of gene i.
        _mu (:obj:`torch.nn.Parameter`): Standard deviation vector, where i-th component is the
            standard deviation of gene i.
    """

    def __init__(self, n_variables, epsilon=1e-45, standardize=False):
        torch.nn.Module.__init__(self)
        self.n_variables = n_variables
        self.epsilon = epsilon
        self.bc_lambda_1 = torch.nn.Parameter(torch.ones(1, self.n_variables))
        self.standardize = standardize

        self._mu = None
        self._std = None

    def init(self, X_arr):
        """Initializes Box-Cox parameters.

        Each transform is fitted based on the marginal likelihood associated
        to one gene, instead of the joint likelihood.

        Args:
            X_arr (:obj:`np.ndarray`): NumPy array containing the input data.
        """
        transformer = PowerTransformer(method='box-cox', standardize=False)
        transformer.fit_transform(X_arr + self.epsilon)
        self.bc_lambda_1.data = torch.FloatTensor(transformer.lambdas_.reshape(1, self.n_variables))
        self._mu = torch.FloatTensor(np.mean(X_arr, axis=0)).unsqueeze(0)
        self._std = torch.FloatTensor(np.std(X_arr, axis=0)).unsqueeze(0)

    def forward(self, X):
        X = X + self.epsilon
        y1 = ((X ** self.bc_lambda_1) - 1.) / self.bc_lambda_1
        y2 = torch.log(X)
        Y = torch.where(self.bc_lambda_1 != 0, y1, y2)

        if self.standardize:
            mu = torch.mean(Y, dim=0).unsqueeze(0)
            std = torch.std(Y, dim=0).unsqueeze(0) + 1e-15
            Y = Y - mu
            Y = Y / std
        return Y


def apply_optimal_transform(X, aweights=None, _lambda=0.8, max_n_iter=100,
                            epsilon=1e-50, verbose=True):

    n_samples = X.shape[0]
    n_genes = X.shape[1]

    boxcox = BoxCoxTransform(n_genes, standardize=False, epsilon=epsilon)
    boxcox.init(X)
    X = torch.DoubleTensor(X)

    boxcox.bc_lambda_1.data = torch.clamp(boxcox.bc_lambda_1.data, -60, 60)

    optimizer = torch.optim.Adam([
        {'params': boxcox.parameters(), 'lr': 0.5*1e-2}
    ])
    convergence = LocalConvergence(5, max_n_iter=max_n_iter, tau=1e-7)

    sum_log_x = torch.sum(torch.log(X + epsilon), dim=0)

    if verbose:
        print('Starting gradient descent.')

    iteration = 1
    while not convergence():

        optimizer.zero_grad()

        X_prime = boxcox(X)
        assert not np.isnan(X_prime.data.numpy()).any()

        C = cov(X_prime, aweights=aweights)
        C_reg = _lambda * torch.diag(torch.diagonal(C)) + (1. - _lambda) * C

        try:
            U = torch.linalg.cholesky(C_reg)
            Theta = torch.cholesky_inverse(U)
            logdet = -2. * torch.sum(torch.log(torch.diagonal(U)))
            # S = torch.inverse(C_reg)
            # logdet = torch.logdet(S)
        except RuntimeError:
            if verbose:
                boxcox.init(X)
                print('Invalid value encountered. Falling back to sequential PORTIA.')

        S = torch.inverse(C_reg)
        ll = -0.5 * (n_samples - 1.) * torch.trace(torch.mm(Theta, C)) + 0.5 * n_samples * logdet
        # ll = -0.5 * n_samples * torch.trace(torch.mm(Theta, C)) + 0.5 * n_samples * logdet

        jac1 = torch.sum((boxcox.bc_lambda_1 - 1.) * sum_log_x)
        loss = -ll - jac1

        loss.backward()

        optimizer.step()
        boxcox.bc_lambda_1.data = torch.clamp(boxcox.bc_lambda_1.data, -60, 60)

        convergence.step(loss.item())

        if verbose:
            print(f'Loss at iteration {iteration}: {loss.item()}')
        iteration += 1

    boxcox.standardize = True
    X_prime = boxcox(X)
    return X_prime.data.numpy()
