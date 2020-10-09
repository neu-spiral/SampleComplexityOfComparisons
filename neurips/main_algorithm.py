from synthetic import k_by_n,m_by_n_k,synthetic_independent_comparisons, generate_comparison_fixed_degree ,generate_tournament
import numpy as np
import pickle
import argparse
import os
from scipy.io import savemat
import sys
from networkx.generators.random_graphs import random_regular_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='synthetic data independent samples instead of independent comparisons')
    parser.add_argument('-d', metavar='d', type=int, default=100,
                        help='dimensionality')
    parser.add_argument('-noiselevel', type=float,default=0.2)
    parser.add_argument('-repeatInd', type=int, default=0,help='repeat index')
    parser.add_argument('-n', type=int, default=None, help='number of samples, if provided, -m will be ignored.')
    parser.add_argument('-k', type=str, default='16', help='number of times a sample is involved in a comparison')  # 0 means tournament
    parser.add_argument('-m', type=int, default=None, help='number of comparisons.')
    parser.add_argument('-output',type=str,default='./',help='output file folder')
    args = parser.parse_args()


    k = args.k
    time_list = []
    d = args.d
    noise_level = args.noiselevel
    repeat_ind= args.repeatInd
    output=args.output
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
        elif k=='tournament':
            k = n-1
        elif k =='nlogn3':
            k = 2*(int(np.log(n)**3)+1)
            if n*k %2 != 0:
                k += 1
        elif k == 'nsqrt':
            k = 2*(int(np.sqrt(n))+1)
            if n*k %2 != 0:
                k += 1
        else: #k = 'fix16'
            k_str = ''.join([char for char in k if not char.isalpha()])
            k = int(k_str)
        m = n * k / 2


    name_base = "randomSplit_d_" + str(
        d)+"_m_"+str(m)+ "_k_" + str(k) + "_n_"+ str(n)  + "_noiseLevel_" + str(
        noise_level).replace(".", "-") + "_repeatInd_" + str(repeat_ind)
    file_name = output+ name_base + ".p"
    if  not os.path.isfile(file_name):
        
        if k !=n-1:
            synthetic_train_pairs = generate_comparison_fixed_degree(n, k, repeat_ind)
        else:
            synthetic_train_pairs = generate_tournament(n)
        synthetic = synthetic_independent_comparisons(n,k, d, synthetic_train_pairs,seed=repeat_ind)
        beta_est, error_list, nte, nfe = synthetic.estimate(noise_level)
        error = error_list[0]
        
        out_dict = {'m': m, 'k': k, 'n':n, 'd': d,
                    'noise_level': noise_level, 
                    'error':error,'nte':nte,'nfe':nfe,
                    'repeat_ind':repeat_ind}
        if not os.path.exists(output):
            os.makedirs(output)
       
        pickle.dump(out_dict, open(file_name, 'wb'))
