from synthetic import k_by_n,m_by_n_k,generate_comparison_ij,synthetic_independent_comparisons, generate_comparison_fixed_degree ,generate_tournament
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from time import time
import pickle
import argparse
import os
from scipy.io import savemat
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='synthetic data independent samples instead of independent comparisons')
    parser.add_argument('-d', metavar='d', type=int, default=100,
                        help='dimensionality')
    parser.add_argument('-error',type=str,choices=['iid','BT','no'],default='iid')
    parser.add_argument('-noiselevel', type=float,default=0.2)
    parser.add_argument('-algorithm',type=str,default='sortedGreedyCycle')
    parser.add_argument('-estimate',type=str,choices=['average','MLE'],default='MLE')
    parser.add_argument('-correct',type=str,choices=['flip','remove','no'],default='remove')
    parser.add_argument('-stoprule',type=float,default=1e-4)
    parser.add_argument('-repeatInd', type=int, default=0,help='repeat index')
    # parser.add_argument('-cv',type=int,default=0)
    parser.add_argument('-n', type=int, default=None)
    parser.add_argument('-k', type=str, default='16')  # 0 means tournament
    parser.add_argument('-m', type=int, default=None)
    args = parser.parse_args()

    # n_list = list(np.arange(1e3, 1e5, 4e3).astype(np.int))
    k = args.k
    # k =  2# degree per node

    time_list = []
    d = args.d
    error_method = args.error
    # algorithms = ['No Correction','Oracle','0.7 Oracle','Repeated MLE']  #'Repeated MLE'
    # algorithms = 'greedyCycle'  #'Repeated MLE'
    # args.algorithm
    estimate_method = args.estimate
    noise_level = args.noiselevel
    algorithm = args.algorithm
    algorithm = algorithm.replace("-"," ")
    stop_criteria = args.stoprule
    correct_method = args.correct
    # ind = args.ind
    repeat_ind= args.repeatInd
    # cv_index = args.cv
    num_repeat = 1


    if args.k.isdigit(): # This cannot check the float number in string. If k is a number, then we are fixed either n or m.
        k = int(k)
        if not args.n is None:
            # We provide n and k

            n = args.n
            m = n * k / 2
        elif not args.m is None:
            # We provide m and k
            m = args.m
            n = 2*m/k
        else:
            sys.exit("we have n "+str(args.n)+" m "+str(args.m)+" k "+str(k))
    else:
        n = args.n
        if k == 'sqrt':
            # k is changing with \sqrt{n-1}. We provide n
            k = int(np.sqrt(n-1))
            if n*k % 2 != 0:
                k -= 1
        elif k=='tournament':
            k = n-1
        else: #k = 'fix16'
            k_str = ''.join([char for char in k if not char.isalpha()])
            k = int(k_str)
        m = n * k / 2
    n_test = 200


    name_base = "randomSplit_" + algorithm +"_d_" + str(
        d)+"_m_"+str(m)+ "_k_" + str(k) + "_n_"+ str(n) + "_" + error_method + "noiseLevel_" + str(
        noise_level).replace(".", "-") + "_repeatInd_" + str(repeat_ind)  + "_" + str(
        estimate_method) + '_' + correct_method
    file_name = "../result/Synthetic_CV_noiseless/" + error_method + "/" + algorithm + "/" + name_base + ".mat"
    if  True:#not os.path.isfile(file_name):
        t1 = time()
        if k !=n-1:
            synthetic_train_pairs = generate_comparison_fixed_degree(n, k, repeat_ind)
            synthetic_test_pairs = generate_comparison_fixed_degree(n_test, n_test-1, repeat_ind+100)
        else:
            synthetic_train_pairs = generate_tournament(n)
            synthetic_test_pairs = generate_tournament(n_test)
        synthetic = synthetic_independent_comparisons(n, n_test,k, d, synthetic_train_pairs,synthetic_test_pairs,seeds=(repeat_ind,100))
        greedy_algorithms = ['greedyCycle','greedyCycleBeta','greedyCycleUncertain','greedyCycleFlipRemove','greedyMulitCycleBeta','greedyMultiCycle','sortedGreedyCycle','sortedGreedyCycleBeta','sortedGreedy']
        if algorithm in greedy_algorithms:
            beta_est, error_list, nte, nfe, cycles_size, removed_likelihood, beta_error_iter = synthetic.estimate(
                error_method, algorithm, estimate_method, noise_level, correct_method)
        else:
            beta_est, error_list, nte, nfe = synthetic.estimate(
            error_method, algorithm,estimate_method, noise_level,correct_method)
        error = error_list[0]
        t1 = time()
        out_dict = {'m': m, 'k': k, 'n':n, 'd': d, 'n_test':n_test,
                    'error_method': error_method, 'algorithm': algorithm, 'estimate_method': estimate_method,
                    'noise_level': noise_level, 'stop_criteria': stop_criteria,'correct_method':correct_method,
                    'error':error,'nte':nte,'nfe':nfe,
                    'repeat_ind':repeat_ind}
        if algorithm in greedy_algorithms:
            out_dict['cycles_size'] = cycles_size
            out_dict['removed_likelihood'] = removed_likelihood
            out_dict['beta_error_iter'] = beta_error_iter
        if not os.path.exists("../result/Synthetic_CV_noiseless/"+error_method+"/"+algorithm+"/"):
            os.makedirs("../result/Synthetic_CV_noiseless/"+error_method+"/"+algorithm+"/")
        #savemat("../result/Synthetic_CV_noiseless/" + error_method + "/" + algorithm + "/" + name_base + ".mat", out_dict)
        pickle.dump(out_dict, open("../result/Synthetic_CV_noiseless/"+error_method+"/"+algorithm+"/" + name_base + ".p", 'wb'))
    print "done"
