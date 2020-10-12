"""
Main file for running
sushi experiments
"""
import random
from argparse import ArgumentParser
from collections import defaultdict
from time import time
import numpy as np
from src.helpers import save_results, check_exp
from src.data import get_sushi_data, get_sushi_fs
from src.estimators import estimate_beta
from src.loss import kt_distance


def parse_args():
    """
    Parse args
    """
    parser = ArgumentParser(description='Run sushi experiments.')
    parser.add_argument('seed', type=int, help='Random seed.')
    parser.add_argument('k', type=int, choices=[1, 2, 3, 4],
                        help='Defines M. 1: N, 2: NloglogN' +
                        ', 3: NlogN, 4: N*sqrt(N).')
    parser.add_argument('method', type=int, choices=[1, 2],
                        help='Beta estimation method. 1: averaging, ' +
                        '2: logistic regression.')
    parser.add_argument('-gamma', type=float, default=0.01,
                        help='Feature covariance regularizor.')
    parser.add_argument('-K', type=int, default=5, help='CV split count.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Get inputs (parameters)
    args = parse_args()
    k = args.k
    method = args.method
    K = args.K
    gamma = args.gamma
    # Set global variables
    np.random.seed(args.seed)
    np.seterr(over='ignore')
    random.seed(args.seed)
    # Outputs (results)
    results = defaultdict(dict)

    # Start experiment if not already finished
    # check_sushi_exp(args)

    # Get all features and all scores
    a_feats, a_scores = get_sushi_fs()

    t0 = time()
    # iterate over splits with cross validation k
    for cvk in range(K):
        # Get comparison features, labels and
        # test item features and scores
        print('getting s data')
        X, XC, test_X, yn, scores = get_sushi_data(a_feats, a_scores, cvk)
        print('done')
        # Estimate beta
        e_beta = estimate_beta(X, XC, yn, method, gamma)
        # Compute kendall tau dist for
        # scores and estimated scores
        e_scores = X @ e_beta
        kt_dist = kt_distance(scores, e_scores)
        # Print and save results
        print('KT:%.3f' % (kt_dist))
        results['kt_dist'] = kt_dist
    results['seed'] = args.seed
    # Save results to disk
    # save_results(results, args)
    print('Finished in %.2f seconds.' % (time() - t0))
