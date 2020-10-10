"""
Main file for running sample complexity experiments
"""
import random
from argparse import ArgumentParser
import numpy as np
from src.helpers import get_NM, get_f_stats
from src.data import gen_data
from src.estimators import estimate_beta
from src.loss import beta_error


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
    parser.add_argument('N1', type=int, help='Smalles N is 10**N1')
    parser.add_argument('N2', type=int, help='Largest N is 10**N2')
    parser.add_argument('k', type=int, choices=[1, 2, 3, 4],
                        help='Defines M. 1: N, 2: NloglogN' +
                        ', 3: NlogN, 4: N*sqrt(N).')
    parser.add_argument('method', type=int, choices=[1, 2],
                        help='Beta estimation method. 1: averaging, ' +
                        '2: logistic regression.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    ld = args.ld
    d = args.d
    k = args.k
    N1 = args.N1
    N2 = args.N2
    method = args.method

    # Get N and M values
    Ns, Ms = get_NM(k, N1, N2)

    # Beta and feature stats change with seed
    beta = np.random.multivariate_normal(np.zeros(d), np.eye(d)*10)
    f_mean, f_cov = get_f_stats(d, ld)

    # This for can be run embarrasingly parallel.
    # But I'm embarrassingly lazy to do that.
    for i, N in enumerate(Ns):
        M = Ms[i]
        # Sample data
        X, XC, yn, y = gen_data(N, M, beta, f_mean, f_cov)
        # Estimate beta
        e_beta = estimate_beta(X, XC, yn, method)
        # Calculate error of beta
        err_angle, err_norm = beta_error(e_beta, beta, f_cov)
        print(err_angle, err_norm)
        # Test beta on new data by kendall tau correlation
        # Write results to disk appropriately
