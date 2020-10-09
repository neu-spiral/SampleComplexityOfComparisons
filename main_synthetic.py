"""
Main file for running sample complexity experiments
"""
from argparse import ArgumentParser

import numpy as np
from src.helpers import get_NM
from src.data import gen_data


def parse_args():
    """
    Parse args and do simple computations if necessary
    """
    parser = ArgumentParser(description='Runs synthetic experiments.')
    parser.add_argument('seed', type=int, help='Random seed.')
    parser.add_argument('ld', type=float,
                        help='Lambda d. Min eig value of data' +
                        ' covariance where max is 1.')
    parser.add_argument('d', type=int, help='Dimensionality.')
    parser.add_argument('k', type=int, choices=[1, 2, 3, 4],
                        help='Defines M. 1 for N, 2 for NloglogN' +
                        ', 3 for NlogN, 4 for N*sqrt(N).')
    parser.add_argument('N1', type=int, help='Smalles N is 10**N1')
    parser.add_argument('N2', type=int, help='Largest N is 10**N2')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    ld = args.ld
    d = args.d
    k = args.k
    N1 = args.N1
    N2 = args.N2

    Ns, Ms = get_NM(k, N1, N2)

    beta = np.random.multivariate_normal(np.zeros(d), np.eye(d)*10)

    for i, N in enumerate(Ns):
        print(N)
        M = Ms[i]
        X, XC, yn, y = gen_data(ld, d, N, M, beta)

        # Estimated mean
        e_mean = X.mean(axis=0)
        # Estimated unbiased covariance
        X -= e_mean
        e_cov = X.T@X/(N-d-2)
