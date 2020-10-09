from Admission import Admission
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from time import time
import pickle
import argparse
import os
from scipy.io import savemat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='synthetic data independent samples instead of independent comparisons')
    parser.add_argument('-noiselevel', type=float,default=0.2)
    parser.add_argument('-algorithm',type=str,default='greedyCycle')
    parser.add_argument('-estimate',type=str,choices=['average','MLE'],default='MLE')
    parser.add_argument('-correct',type=str,choices=['flip','remove','no'],default='remove')
    parser.add_argument('-num',type = int, default=0, help='use which one in the n_list (log scale logspace(3, 5, 15) space)')
    parser.add_argument('-cv',type=int,default=-1)
    parser.add_argument('-k',type=str, default="0")
    args = parser.parse_args()

    time_list = []
    # algorithms = ['No Correction','Oracle','0.7 Oracle','Repeated MLE']  #'Repeated MLE'
    # algorithms = 'greedyCycle'  #'Repeated MLE'
    # args.algorithm
    estimate_method = args.estimate
    noise_level = args.noiselevel
    algorithm = args.algorithm
    algorithm = algorithm.replace("-"," ")
    correct_method = args.correct
    sub_num = args.num
    cv_index = args.cv
    k = args.k
    if k.isdigit():
        k = eval(k)
    num_repeat = 1
    name_base = "itemCV_"+algorithm  +"_k_"+str(k)+ "_error_" + "noiseLevel_" + str(
        noise_level).replace(".", "-") + "_sub_" + str(sub_num) + '_cv_'+str(cv_index)+"_" + str(estimate_method) + '_' + correct_method
    # file_name = "../result/Admissions/"+algorithm+"/" + name_base + ".mat"

    if  not os.path.isfile("../result/Admissions/"+algorithm+"/" + name_base + ".p"):
        t1 = time()
        ad = Admission(sub_num,k)
        out_dict = {'algorithm': algorithm, 'estimate_method': estimate_method,
                    'noise_level': noise_level, 'correct_method': correct_method,'k':k}
        greedy_algorithms = ['greedyCycle','greedyCycleBeta','greedyCycleUncertain','greedyCycleFlipRemove','greedyMulitCycleBeta','greedyMultiCycle','sortedGreedyCycle','sortedGreedyCycleBeta','sortedGreedy']
        if algorithm == 'Oracle':
            test_auc, test_noiseless_auc = ad.estimate(algorithm,estimate_method,noise_level,correct_method,cv_fold_index=cv_index)
            out_dict['test_auc'] = test_auc
            out_dict['test_noiseless_auc'] = test_noiseless_auc
        elif algorithm == "No Correction":
            test_auc, test_noiseless_auc, num_true_error, num_false_error = ad.estimate(algorithm, estimate_method, noise_level, correct_method, cv_fold_index=cv_index)
            out_dict['test_auc'] = test_auc
            out_dict['test_noiseless_auc'] = test_noiseless_auc
        elif algorithm in greedy_algorithms:
            num_true_error,num_false_error, cycles_size, removed_likelihood, beta_est_iter = ad.estimate(
                algorithm, estimate_method, noise_level, correct_method, cv_fold_index=cv_index)
            out_dict['num_true_error'] = num_true_error
            out_dict['num_false_error'] = num_false_error
            out_dict['cycles_size'] = cycles_size
            out_dict['removed_likelihood'] = removed_likelihood
            out_dict['beta_est_iter'] = beta_est_iter
        else:
            beta_est_list,error_list, num_true_error,num_false_error = ad.estimate(
            algorithm, estimate_method, noise_level, correct_method, cv_fold_index=cv_index)
        t1 = time()

        if not os.path.exists("../result/Admissions/"+algorithm+"/"):
            os.makedirs("../result/Admissions/"+algorithm+"/")
        savemat("../result/Admissions/" + algorithm + "/" + name_base + ".mat", out_dict)
        pickle.dump(out_dict, open("../result/Admissions/"+algorithm+"/" + name_base + ".p", 'wb'))
    print "done"