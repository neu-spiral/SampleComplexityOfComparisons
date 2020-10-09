"""
Data generation and reading related
"""

from random import choices

import numpy as np
from src.helpers import gen_tour


def gen_data(ld, d, N, M, beta):
    """
    Generates d dimensional gaussian vectors with random mean and covariance.
    Lambda d, i.e. minimum eigen value is ld, max is 1, other are u.a.r.
    Uniformly at random pairwise comparisons are chosen for M comparisons.
    Labels are generated with Bradley Terry model.
    """
    # Feature mean
    f_mean = np.random.rand(d)*10 - 5

    basis = np.random.randn(d, d)
    # Orthonormal basis as columns
    basis, _ = np.linalg.qr(basis)
    eigen_values = np.linspace(ld, 1, d)
    # Feature covariance with eigen value composition
    f_cov = basis*eigen_values @ basis.T

    # Data for statistics estimation, i.e. covariance estimation
    X = np.random.multivariate_normal(f_mean, f_cov, N)

    # Data for comparisons
    X1 = np.random.multivariate_normal(f_mean, f_cov, N)

    # Sample edges
    all_edges = gen_tour(N)
    for n in range(N):
        all_edges.append((n, n))
    # Uniformly at random edges from all possible edges
    edges = choices(all_edges, k=M)

    # An edge from u to v
    # implies v beats u
    u = [e[0] for e in edges]
    v = [e[1] for e in edges]

    # Comparison features
    XC = X1[v] - X1[u]

    # beta^T(X-Y)
    scores = XC @ beta
    # BTL Probability
    p = (1+np.exp(-scores))**-1
    # Uniform random variables
    urv = np.random.rand(M)
    # BTL Labels
    yn = np.sign(urv - p)
    # True labels
    y = np.sign(scores)

    return X, XC, yn, y
