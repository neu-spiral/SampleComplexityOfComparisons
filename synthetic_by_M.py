"""
Main file for running
synthethic sample complexity experiments
"""
import random
from argparse import ArgumentParser
from collections import defaultdict
from time import time
import numpy as np
from src.helpers import get_f_stats, save_results, check_exp, get_alpha, \
        get_c1, get_NM
from src.data import get_data
from src.estimators import estimate_beta
from src.loss import beta_error, kt_distance


def parse_args():
    """
    Parse args and do simple computations if necessary
    """
    parser = ArgumentParser(description='Run synthetic experiments.')
    parser.add_argument('seed', type=int, help='Random seed.')
    parser.add_argument('ld', type=float,
                        help='Lambda d. Min eig value of data' +
                        ' covariance where max is 1.')
    parser.add_argument('pe', type=float, help='Probability of error.')
    parser.add_argument('d', type=int, help='Dimensionality.')
    parser.add_argument('N', type=int, help='N number of nodes.')
    parser.add_argument('M', type=int, help='Largest M.')
    parser.add_argument('method', type=int, choices=[1, 2, 3],
                        help='Beta estimation method. 1: averaging, ' +
                        '2: logistic regression, 3: RABF-LOG')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Get inputs (parameters)
    args = parse_args()
    ld = args.ld
    pe = args.pe
    d = args.d
    N = args.N
    M = args.M
    method = args.method
    # Set global variables
    np.random.seed(args.seed)
    np.seterr(over='ignore')
    random.seed(args.seed)
    # Outputs (results)
    results = defaultdict(dict)

    # Start Experiment if not already finished
    check_exp(args, 'synth_by_M')

    # Get M values to run for
    Ms, _ = get_NM(1, 300, M)
    # Beta and feature stats change with seed
    beta = np.random.multivariate_normal(np.zeros(d), np.eye(d)*10)
    f_mean, f_cov = get_f_stats(d, ld)
    # Get alpha that results in prob of error
    alpha = get_alpha(pe, beta, f_cov)
    e_c1 = get_c1(alpha, beta, f_cov)
    print('pe: %.1f | alpha: %.3f | c1: %.3f' % (pe, alpha, e_c1))

    # This for can be run embarrasingly parallel.
    # But I'm embarrassingly lazy to do that.
    for M in Ms:
        # Sample data
        X1, X2, u, v, XC, yn = get_data(N, M, beta, f_mean, f_cov, alpha)
        # Estimate beta
        t0 = time()
        e_beta = estimate_beta(X1, u, v, XC, yn, method)
        duration = time() - t0
        # Calculate error of beta
        err_angle, err_norm = beta_error(e_beta, beta, method, e_c1)
        # Test e_beta on new data for kendall tau
        test_X, _, _, _, _, _ = get_data(500, 1, beta, f_mean, f_cov, alpha)
        scores = test_X @ beta
        e_scores = test_X @ e_beta
        kt_dist = kt_distance(scores, e_scores)
        # Print and save results
        print('ld:%.3f | d:%3i | N:%6i | M:%6i | Ang:%.3f | Norm:%.3f | KT:%.3f'
              % (ld, d, N, M, err_angle, err_norm, kt_dist))
        results[M]['err_angle'] = err_angle
        results[M]['err_norm'] = err_norm
        results[M]['kt_dist'] = kt_dist
        results[M]['duration'] = duration 
    results['seed'] = args.seed
    results['ld'] = ld
    results['pe'] = pe
    results['d'] = d
    results['method'] = method
    results['Ns'] = np.array(N)
    results['Ms'] = Ms
    # Save results to disk
    save_results(results, args, 'synth_by_M')
