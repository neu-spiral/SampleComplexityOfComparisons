# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import sys
import numpy as np

from ranking_dataset import ranking_dataset


def args_reader(args):
    fold = args.fold  # K-th fold in repeated cross validation
    estimate_method = args.estimate
    algorithm = args.algorithm
    algorithm = algorithm.replace("-", " ")
    correct_method = args.correct
    noise_type = args.noise_type
    noise_level = args.noise_level
    threshold = args.threshold
    seed = args.seed

    if threshold < 0 or threshold > 1:
        raise (BaseException('Threshold is not correct.'))
    if noise_level < 0 or noise_level > 1:
        raise (BaseException('Noise level is not correct.'))

    if algorithm == 'None':
        correct_method = 'None'
        threshold = 'None'

    return fold, algorithm, estimate_method, correct_method, noise_type, noise_level, threshold, seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "This file is used for running greedy cycle experiments on Netflix data.")
    parser.add_argument('-dataset',type=str,default='netflix')
    parser.add_argument('-fold', type=int, default=0, help="Split (fold) index in [0, 49].")
    parser.add_argument('-algorithm', type=str,  default='sortedGreedyCycle')
    parser.add_argument('-estimate', type=str, choices=['average', 'MLE'], default='MLE')
    parser.add_argument('-correct', type=str, choices=['flip', 'remove', 'None'], default='remove')
    parser.add_argument('-noisetype', type=str, choices=['iid', 'bradley'], default='iid',
                        help='Noise type added (only) to the training set.')
    parser.add_argument('-noiselevel', type=float, default=0.1, help='Fraction of flipped labels, in [0, 1].')
    parser.add_argument('-seed', type=int, default=0, help='Used in choosing labels that are flipped.')
    parser.add_argument('-n', type=int, default=None)
    parser.add_argument('-k', type=str, default='16')  # 0 means tournament
    parser.add_argument('-m', type=int, default=None)

    args = parser.parse_args()

    k = args.k
    # k =  2# degree per node

    time_list = []
    dataset = args.dataset
    fold = args.fold
    noise_type = args.noisetype
    estimate_method = args.estimate
    noise_level = args.noiselevel
    algorithm = args.algorithm
    algorithm = algorithm.replace("-", " ")
    correct_method = args.correct

    if args.k.isdigit():  # This cannot check the float number in string. If k is a number, then we are fixed either n or m.
        k = int(k)
        if not args.n is None:
            # We provide n and k
            n = args.n
            if k == 0:
                k = n-1
            m = n * k / 2
        elif not args.m is None:
            # We provide m and k
            m = args.m
            n = 2 * m / k
        else:
            sys.exit("we have n " + str(args.n) + " m " + str(args.m) + " k " + str(k))
    else:
        n = args.n
        if k == 'sqrt':
            # k is changing with \sqrt{n-1}. We provide n
            k = int(np.sqrt(n - 1))
        elif k == 'half':
            k = int(0.5*n)
        else:  # k = 'fix16'
            k_str = ''.join([char for char in k if not char.isalpha()])
            k = int(k_str)
        m = n * k / 2

    folder = "../result/"+dataset+"/" + algorithm + "/"
    name_base = dataset+"_" + algorithm  + "_m_" + str(m) + "_k_" + str(k) + "_n_" + str(n) + "_" + noise_type + "noiseLevel_" + str(
        noise_level).replace(".", "-") + "_fold_" + str(fold) + "_" + str(
        estimate_method) + '_' + correct_method
    file_name = folder + name_base +".p"

    # Check if the result folder for current algorithm exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    if  not os.path.isfile(file_name):
        if args.k == 'sqrt':
            k = args.k
        elif args.k == '0':
            k = 0
        elif args.k == 'half':
            k = args.k
        rank_data = ranking_dataset(dataset,n,k)
        greedy_algorithms = ['greedyCycle','greedyCycleBeta','greedyCycleUncertain','greedyCycleFlipRemove','greedyMulitCycleBeta','greedyMultiCycle','sortedGreedyCycle','sortedGreedyCycleBeta','sortedGreedy']
        if algorithm in greedy_algorithms:
            nte, nfe, cycles_size, removed_likelihood, beta_est_iter= rank_data.estimate(
                algorithm, estimate_method, noise_level, correct_method,cv_fold_index=fold)
            error = [beta_est_iter]
        else:
            beta_est_list, error_list, nte, nfe = rank_data.estimate(
            algorithm,estimate_method, noise_level,correct_method,cv_fold_index=fold)
            error = error_list[0]
        out_dict = {'m': m, 'k': k, 'n':n,
                    'error_method': noise_type, 'algorithm': algorithm, 'estimate_method': estimate_method,
                    'noise_level': noise_level, 'correct_method':correct_method,
                    'error':error,'nte':nte,'nfe':nfe,}
        if algorithm in greedy_algorithms:
            out_dict['cycles_size'] = cycles_size
            out_dict['removed_likelihood'] = removed_likelihood
            out_dict['beta_error_iter'] = beta_est_iter
        if not os.path.exists(folder):
            os.makedirs(folder)
        #savemat("../result/Synthetic_CV_noiseless/" + error_method + "/" + algorithm + "/" + name_base + ".mat", out_dict)
        pickle.dump(out_dict, open(file_name, 'wb'))
    print "done"
