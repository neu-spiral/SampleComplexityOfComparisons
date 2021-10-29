"""
Main file for running
synthethic sample complexity experiments
"""
import random
from argparse import ArgumentParser
from collections import defaultdict
from time import time
import numpy as np
from src.helpers import get_NM, get_f_stats, save_results, check_exp, \
        get_alpha, get_c1
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
    parser.add_argument('N1', type=int, help='Smallest N.')
    parser.add_argument('N2', type=int, help='Largest N.')
    parser.add_argument('k', type=int, choices=[1, 2, 3, 4],
                        help='Defines M. 1: N, 2: NloglogN' +
                        ', 3: NlogN, 4: N*sqrt(N).')
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
    N1 = args.N1
    N2 = args.N2
    k = args.k
    method = args.method
    # Set global variables
    np.random.seed(args.seed)
    np.seterr(over='ignore')
    random.seed(args.seed)
    # Outputs (results)
    results = defaultdict(dict)

    # Start experiment if not already finished
    check_exp(args, 'synth')

    # Get N and M values
    Ns, Ms = get_NM(k, N1, N2)
    # Beta and feature stats change with seed
    beta = np.random.multivariate_normal(np.zeros(d), np.eye(d)*10)
    f_mean, f_cov = get_f_stats(d, ld)
    # Estimate alpha that gives required prob of error and c1
    alpha = get_alpha(pe, beta, f_cov)
    e_c1 = get_c1(alpha, beta, f_cov)
    print('pe: %.1f | alpha: %.3f | c1: %.3f' % (pe, alpha, e_c1))

    # This for can be run embarrasingly parallel.
    # But I'm embarrassingly lazy to do that.
    t0 = time()
    for i, N in enumerate(Ns):
        M = Ms[i]
        # Sample data
        X1, X2, u, v, XC, yn = get_data(N, M, beta, f_mean, f_cov, alpha)
        # Estimate beta
        e_beta = estimate_beta(X1, u, v, XC, yn, method)
        # Calculate error of beta
        err_angle, err_norm = beta_error(e_beta, beta, method, e_c1)
        # Test e_beta on new data for kendall tau
        test_X, _, _, _, _, _ = get_data(500, 1, beta, f_mean, f_cov, alpha)
        scores = test_X @ beta
        e_scores = test_X @ e_beta
        kt_dist = kt_distance(scores, e_scores)
        # Print and save results
        print('ld: %f | d:%3i | N:%6i | M:%6i | Ang:%.3f | Norm:%.3f | KT:%.3f'
              % (ld, d, N, M, err_angle, err_norm, kt_dist))
        results[N]['err_angle'] = err_angle
        results[N]['err_norm'] = err_norm
        results[N]['kt_dist'] = kt_dist
    results['seed'] = args.seed
    results['ld'] = ld
    results['pe'] = pe
    results['d'] = d
    results['k'] = k
    results['method'] = method
    results['Ns'] = Ns
    results['Ms'] = Ms
    # Save results to disk
    save_results(results, args, 'synth')
    print('Finished in %.2f seconds.' % (time() - t0))
