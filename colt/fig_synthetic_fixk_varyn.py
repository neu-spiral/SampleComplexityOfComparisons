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

def organize_result_for_figures(error_method,correct_method,noise_level, k, n_list, d_list):
    algorithms = ['No Correction']
    estimate_method = 'average'
    estimate_method_dict = {'MLE': 'MLE', 'average': "Stein's Estimate"}
    test_auc_array_list, test_noiseless_auc_array_list, error_array_list, error_normalized_array_list = [], [], [], []
    num_points = len(n_list)
    n_list_displayed = np.array(n_list)  # np.sqrt(np.array(n_list))
    d_list = list(d_list)
    for d in d_list:
        error_array, error_normalized_array, test_auc_array, test_noiseless_auc_array = nonparametric("No Correction",
                                                                                                      k, d,
                                                                                                      error_method,
                                                                                                      noise_level,
                                                                                                      n_list,
                                                                                                      estimate_method,
                                                                                                      correct_method)
        error_array_list.append(1 * error_array)
        error_normalized_array_list.append(1 * error_normalized_array)
        test_auc_array_list.append(1 * test_auc_array)
        test_noiseless_auc_array_list.append(1 * test_noiseless_auc_array)
    saved_dict = {"error_array_list": error_array_list, 'error_normalized_array_list': error_normalized_array_list,
                  'test_auc_array_list': test_auc_array_list,'n_list_displayed':n_list_displayed,'estimate_method':estimate_method}
    pickle.dump(saved_dict,
                open("../data/result/fix_k_vary_n_nl" + str(noise_level) + "_k_" + str(k) + ".p", 'wb'))

def produce_figure(noise_level, k, d_list):
    # k_dict = {16: r"k=16",0:r"tournament", "sqrt":r"$k=\sqrt{m-1}$"}
    # d = 100
    # m = n*2/k
    f = pickle.load(open("../data/result/fix_k_vary_n_nl" + str(noise_level) + "_k_" + str(k) + ".p", 'rb'))
    # error_array_list = f['error_array_list']
    error_normalized_array_list = f['error_normalized_array_list']
    n_list_displayed = f['n_list_displayed']
    estimate_method = f['estimate_method']

    num_points = len(n_list_displayed)
    d_legends = ["d="+str(d) for d in d_list]
    common_fig_name_base = '_' + str(
        estimate_method) + str(noise_level).replace(".", "-")[-1]+"_"+str(k)
    # varyn_normalized_error_average0nlogn3
    ########################################################################################################################
    # Error Fig normalized \|beta-cbeta\|

    fig_name_base = "varyn_normalized_error"+common_fig_name_base
    sns_palette = sns.color_palette()
    sns_cmap = ListedColormap(sns_palette.as_hex())
    linestyes = ['-', '--', '-.', ':', '']
    markers = [".", 's', '^', 'D', 'p', '8','*','x','h','1','2','3','4']
    plt.switch_backend('agg')
    fig_error = plt.figure()
    linestyle_counter = 0
    line_counter = 0
    for i in range(len(error_normalized_array_list)):
        if line_counter >= len(markers):
            line_counter = 0
            linestyle_counter += 1
        plt.plot(n_list_displayed[:num_points], np.nanmean(error_normalized_array_list[i][:num_points, :], axis=1), linestyle=linestyes[line_counter],
                 marker=markers[i],
                 markersize=5, linewidth=1.5,
                 c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
    # plt.xscale("log",basex=2)
    plt.xlabel(r'$N$', fontsize=30)
    plt.ylim((0,0.4))
    xmin, xmax, ymin, ymax = plt.axis()
    ref_x = np.arange(xmin, xmax, 10)
    ref_y = 0.1 * np.ones(ref_x.shape)
    plt.plot(ref_x, ref_y, '--', c='k')
    plt.ylabel(r'$\|\|\frac{\hat{\beta}}{\|\|\hat{\beta}\|\|}-\frac{\beta}{\|\|\beta\|\|}\|\|$', fontsize=30)
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    # algorithms_legend = ['No Correction', 'Oracle', '0.7 Oracle', 'GreedyCycleCorrection']
    if noise_level == 0:
        plt.legend(d_legends, fontsize=20)


    for i in range(len(error_normalized_array_list)):
        plt.fill_between(n_list_displayed[:num_points],
                         np.nanmean(error_normalized_array_list[i][:num_points, :], axis=1) - np.nanstd(
                             error_normalized_array_list[i][:num_points, :], axis=1),
                         np.nanmean(error_normalized_array_list[i][:num_points, :], axis=1) + np.nanstd(
                             error_normalized_array_list[i][:num_points, :], axis=1),
                         color=sns_palette[i], alpha=0.3)
    fig_error.savefig("../fig/" + fig_name_base + ".pdf", bbox_inches='tight')

    return


if __name__=="__main__":
    error_methods = ['iid']
    correct_methods = ['flip']
    noise_levels_dict = {'iid':[0.0,0.2,0.4],'BT':[0.41,0.185,0.095,0.042]}
    n_list = [i*1000 for i in range(1,16,)] # MLE uses this.
    d_list = [i*10 for i in range(1,11,2)]
    k_list = ["nlogn3","nsqrt"]
    total_loops = len(error_methods)*len(correct_methods)*(len(noise_levels_dict['iid'])*len(k_list))
    count = 0


    # Use once
    for error_method in error_methods:
        noise_levels = noise_levels_dict[error_method]
        for noise_level in noise_levels:
            for k in k_list:
                organize_result_for_figures(error_method,correct_methods[0],noise_level,k,n_list,d_list)
    # Use once ends here

    for error_method in error_methods:
        noise_levels = noise_levels_dict[error_method]
        for noise_level in noise_levels:
            for correct_method in correct_methods:
                for k in k_list:
                    count += 1
                    print("Processing "+str(count)+"/"+str(total_loops))
                    temp = produce_figure(noise_level,k,d_list)
    print("done")

