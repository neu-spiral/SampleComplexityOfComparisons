"""
Main file for running
sushi experiments
"""
from argparse import ArgumentParser
from time import time
import numpy as np
from src.helpers import save_results, check_exp
from src.data import get_sushi_data
from src.estimators import estimate_beta
from src.loss import kt_distance


def parse_args():
    """
    Parse args
    """
    parser = ArgumentParser(description='Run sushi experiments.')
    parser.add_argument('method', type=int, choices=[1, 2],
                        help='Beta estimation method. 1: averaging, ' +
                        '2: logistic regression.')
    parser.add_argument('-gamma', type=float, default=1e-9,
                        help='Feature covariance regularizor.')
    parser.add_argument('-K', type=int, default=5, help='CV split count.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Get inputs (parameters)
    args = parse_args()
    method = args.method
    gamma = args.gamma
    K = args.K
    # Outputs (results)
    results = {}

    # Start experiment if not already finished
    check_exp(args, 'sushi')

    Ns = list(range(22, 100*(K-1)//K//2+1, 2))

    results['Ns'] = Ns
    results['K'] = K
    for N in Ns:
        results[N] = {}
        results[N]['acc'] = [[] for _ in range(K)]
        results[N]['kt_dist'] = [[] for _ in range(K)]

    t0 = time()
    # iterate over splits with cross validation k
    for cvk in range(K):
        for N in Ns:
            # Get comparison features, labels
            X, XC, test_X, test_XC, yn, test_yn, test_scores = \
                    get_sushi_data(cvk, N)
            # Estimate beta
            e_beta = estimate_beta(X, XC, yn, method, gamma)

            # Compute error metrics
            # Compute accuracy over given edges
            e_test_y = np.sign(test_XC @ e_beta)
            correct_test_y = e_test_y == test_yn
            acc = np.sum(correct_test_y)/len(correct_test_y)
            # Compute kendall tau dist for
            # scores and estimated scores
            e_scores = test_X @ e_beta
            kt_dist = kt_distance(test_scores, e_scores)
            # Print and save results
            print('cvk:%i | N:%2i | Acc: %.3f | KT:%.3f'
                  % (cvk, N, acc, kt_dist))
            results[N]['acc'][cvk] = acc
            results[N]['kt_dist'][cvk] = kt_dist
    # Save results to disk
    save_results(results, args, 'sushi')
    print('Finished in %.2f seconds.' % (time() - t0))
