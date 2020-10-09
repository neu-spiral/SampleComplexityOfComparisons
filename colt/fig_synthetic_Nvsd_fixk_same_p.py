import pickle
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os.path
from scipy.io import loadmat


def nonparametric(algorithm,k_set,d,error_method,noise_level,n_list,estimate_method,correct_method):
    error_array = np.zeros((len(n_list),5))
    error_array[:] = np.nan
    error_normalized_array  = np.zeros((len(n_list),5))
    error_normalized_array[:] = np.nan
    test_auc_array = np.zeros((len(n_list),5))
    test_auc_array[:] = np.nan
    test_noiseless_auc_array = np.zeros((len(n_list), 5))
    test_noiseless_auc_array[:] = np.nan
    missing_files = []
    counter = 0
    for n in n_list:
        if k_set == 'nlogn3':
            k = 2 * (int(np.log(n) ** 3) + 1)
        elif k_set == 'nsqrt':
            k = 2 * (int(np.sqrt(n)) + 1)
        else:
            k = k_set
        if n * k % 2 != 0:
            k += 1
        m = n*k/2
        for repeat_ind in range(5):
            name_base = "randomSplit_" + algorithm + "_d_" + str(d) + "_m_" + str(m) + "_k_" + str(k) + "_n_" + str(n) + "_" + error_method + "noiseLevel_" + str(noise_level).replace(".", "-") + "_repeatInd_" + str(repeat_ind) + "_" + str(estimate_method) + '_' + correct_method
            file_name = "../data/result/" + error_method + "/" + algorithm + "/" + name_base + ".p"
            if os.path.isfile(file_name):
                    pickle_file = pickle.load(open(file_name, 'rb'))
                    k = pickle_file['k']
                    error_array[counter, repeat_ind] = 1 * pickle_file['error'][0][0]
                    error_normalized_array[counter, repeat_ind] = 1 * pickle_file['error'][0][1]
                    test_auc_array[counter, repeat_ind] = 1 * pickle_file['error'][1]
                    test_noiseless_auc_array[counter, repeat_ind] = 1 * pickle_file['error'][2]
            else:
                missing_files.append(file_name)
        counter += 1
    return error_array, error_normalized_array, test_auc_array, test_noiseless_auc_array

def organize_result_for_figures(noise_level,error_method,correct_method,epsilons, k, n_list, d_list):
    algorithms = ['No Correction']
    estimate_method = 'average'
    estimate_method_dict = {'MLE': 'MLE', 'average': "Stein's Estimate"}
    num_points = len(d_list)
    d_list = list(d_list)
    d_list_displayed = np.array(d_list)

    N_normalized_epsilon_list = []
    N_epsilon_list = []
    for epsilon in epsilons:
        N_normalized_epsilon = np.zeros(d_list_displayed.shape)
        N_normalized_epsilon[:] = np.nan
        N_epsilon = np.zeros(d_list_displayed.shape)
        N_epsilon[:] = np.nan
        for d_counter, d in enumerate(d_list):
            error_array, error_normalized_array, test_auc_array, test_noiseless_auc_array = nonparametric(
                "No Correction", k, d, error_method, noise_level, n_list, estimate_method, correct_method)
            error_array_ave = np.nanmean(error_array, axis=1)
            ind_N_error_array = np.where(error_array_ave <= epsilon)[0]
            if len(ind_N_error_array) != 0:
                N_epsilon[d_counter] = 1 * n_list[ind_N_error_array[0]]
            error_normalized_array_ave = np.nanmean(error_normalized_array, axis=1)
            ind_N_normalized_error_array = np.where(error_normalized_array_ave <= epsilon)[0]
            if len(ind_N_normalized_error_array) != 0:
                N_normalized_epsilon[d_counter] = 1 * n_list[ind_N_normalized_error_array[0]]
        N_epsilon_list.append(1 * N_epsilon)
        N_normalized_epsilon_list.append(1 * N_normalized_epsilon)
    saved_dict = {"N_epsilon_list":N_epsilon_list,'N_normalized_epsilon_list':N_normalized_epsilon_list,'d_list_displayed':d_list_displayed}
    pickle.dump(saved_dict,open("../data/result/N_vs_d_fix_k_same_p_nl"+str(noise_level)+"_k_"+str(k)+".p",'wb'))
    return


def produce_figure(noise_level,epsilons, k):
    f = pickle.load(open("../data/result/N_vs_d_fix_k_same_p_nl"+str(noise_level)+"_k_"+str(k)+".p",'rb'))
    # N_epsilon_list = f["N_epsilon_list"]
    N_normalized_epsilon_list = f["N_normalized_epsilon_list"]
    d_list_displayed = f["d_list_displayed"]
    epsilon_legends = [r"$\epsilon=" + str(epsilon) + "$" for epsilon in epsilons]
    common_fig_name_base = "p" + str(noise_level)[-1] + "_" + str(k)
    ########################################################################################################################
    # Normalized Error Fig
    fig_name_base = "d_vs_n_normalized_"+common_fig_name_base
    sns_palette = sns.color_palette()
    sns_cmap = ListedColormap(sns_palette.as_hex())
    linestyes = ['-', '--', '-.', ':', '']
    markers = [".", 's', '^', 'D', 'p', '8','*','x','h','1','2','3','4']
    plt.switch_backend('agg')
    fig_error_normalized = plt.figure()

    for i in range(len(N_normalized_epsilon_list)):
        plt.plot( N_normalized_epsilon_list[i], d_list_displayed, linestyle=linestyes[i],
                 marker=markers[i],
                 markersize=5, linewidth=1.5,
                 c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
    plt.xlabel(r'$N$', fontsize=30)
    plt.ylabel(r'$d$', fontsize=30)
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.legend(epsilon_legends, fontsize=20)
    if k =='nlogn3':
        m_title = "$M=N\log(N)^3$"
    elif k == 'nsqrt':
        m_title = "$M=N\sqrt{N}$"
    plt.show()
    fig_error_normalized.savefig("../fig/" + fig_name_base + ".pdf", bbox_inches='tight')
    return


if __name__=="__main__":
    error_methods = ['iid']
    correct_methods = ['flip']
    epsilons = [0.1*i for i in range(1,4)]
    n_list = [i * 1000 for i in range(1, 16, 1)]
    d_list = [i*10 for i in range(1,11,1)]
    noise_level = 0.2
    k_list = ["nlogn3", "nsqrt"]
    # for k in k_list:   # This only needs to be used once for organizing results.
    #     organize_result_for_figures(noise_level,error_methods[0],correct_methods[0],epsilons,k,n_list,d_list)
    count = 0
    for error_method in error_methods:
        for correct_method in correct_methods:
            for k in k_list:
                temp = produce_figure(noise_level,epsilons,k)

    print("done")

