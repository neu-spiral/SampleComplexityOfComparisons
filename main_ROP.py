from ROP import ROP
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
    parser.add_argument('-fold', type=int,default = 0)
    parser.add_argument('-nums',type=int,default=80,help='Number of sub sampled images')
    parser.add_argument('-algorithm',type=str,default='greedyCycle')
    parser.add_argument('-estimate',type=str,choices=['average','MLE'],default='MLE')
    parser.add_argument('-correct',type=str,choices=['flip','remove','no'],default='remove')
    args = parser.parse_args()
    greedy_algorithms = ['greedyCycle', 'greedyCycleBeta', 'greedyCycleUncertain', 'greedyCycleFlipRemove',
                         'greedyMulitCycleBeta', 'greedyMultiCycle',
                         'sortedGreedyCycle', 'sortedGreedyCycleBeta', 'sortedGreedy']
    # n_list = list(np.arange(1e3, 1e5, 4e3).astype(np.int))
    fold = args.fold # K-th fold
    nums = args.nums
    # algorithms = ['No Correction','Oracle','0.7 Oracle','Repeated MLE']  #'Repeated MLE'
    # algorithms = 'greedyCycle'  #'Repeated MLE'
    # args.algorithm
    estimate_method = args.estimate
    algorithm = args.algorithm
    algorithm = algorithm.replace("-"," ")
    correct_method = args.correct
    name_base = "ROP_"+str(nums)+"_"+algorithm +  '_' + str(
        estimate_method) +"_fold_" + str(fold)+ '_' + correct_method
    folder = "../result/ROP/"+algorithm+"/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_name = folder + name_base + ".mat"
    if  True:#not os.path.isfile(file_name):
        ROP_data = ROP(lam=0, nums=nums) # MLE
        t1 = time()
        out_dict = {'fold': fold, 'algorithm': algorithm, 'estimate_method': estimate_method,
                    'correct_method': correct_method}
        if algorithm == 'No Correction':
            auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp = ROP_data.train_test(fold,algorithm,estimate_method)
        elif algorithm in greedy_algorithms:
            auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp, ind_error_labels, cycles_size, removed_likelihood, beta_est_iter = ROP_data.train_test(fold,algorithm,estimate_method,correct_method=correct_method)
            out_dict['ind_error_labels'] =ind_error_labels
            out_dict['cycles_size'] = cycles_size
            out_dict['removed_likelihood'] = removed_likelihood
            out_dict['beta_est_iter'] = beta_est_iter
        else:
            auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp, ind_error_labels = ROP_data.train_test(fold, algorithm, estimate_method,correct_method=correct_method)
            out_dict['ind_error_labels'] = ind_error_labels
        out_dict['auc_class_plus']= auc_class_plus
        out_dict['acc_class_plus'] = acc_class_plus
        out_dict['auc_class_normal']= auc_class_normal
        out_dict['acc_class_normal']= acc_class_normal
        out_dict['auc_cmp']= auc_cmp
        out_dict['acc_cmp'] = acc_cmp
        out_dict['auc_maj_cmp'] = auc_maj_cmp
        out_dict['acc_maj_cmp'] = acc_maj_cmp
        savemat(file_name,out_dict)
        pickle.dump(out_dict,open(file_name[:-3]+"p",'wb'))
    print "done"
