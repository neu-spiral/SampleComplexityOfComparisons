"""
Data generation and reading related
"""
from random import choices
import numpy as np


def gen_data(N, M, beta, f_mean, f_cov):
    """
    Generates d dimensional gaussian vectors.
    Uniformly at random pairwise comparisons are chosen.
    Labels are generated with Bradley Terry model.
    """
    # Data for covariance estimation
    X = np.random.multivariate_normal(f_mean, f_cov, N)

    # Data for comparisons
    X1 = np.random.multivariate_normal(f_mean, f_cov, N)

    # Uniformly at random edges
    vertices = list(range(N))
    u = choices(vertices, k=M)
    v = choices(vertices, k=M)

    # An edge from u to v implies v beats u
    # Comparison features
    XC = X1[v] - X1[u]

    # beta^T(X-Y)
    scores = XC @ beta
    # BTL Probability
    p = (1+np.exp(-scores))**-1
    # Uniform random variables
    urv = np.random.rand(M)
    # BTL Labels
    yn = np.sign(p - urv)
    # True labels
    y = np.sign(scores)

    return X, XC, yn, y
