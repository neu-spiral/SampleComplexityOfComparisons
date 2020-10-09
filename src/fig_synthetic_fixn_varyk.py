import pickle
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os.path
from scipy.io import loadmat


def nonparametric(algorithm,n,d,error_method,noise_level,k_list,estimate_method,correct_method):
    error_array = np.zeros((len(k_list),5))
    error_array[:] = np.nan
    error_normalized_array  = np.zeros((len(k_list),5))
    error_normalized_array[:] = np.nan
    test_auc_array = np.zeros((len(k_list),5))
    test_auc_array[:] = np.nan
    test_noiseless_auc_array = np.zeros((len(k_list), 5))
    test_noiseless_auc_array[:] = np.nan
    missing_files = []
    counter = 0
    for k in k_list:
        if n * k % 2 != 0:
            k += 1
        m = int(n*k/2)
        for repeat_ind in range(5):
            name_base = "randomSplit_" + algorithm + "_d_" + str(d) + "_m_" + str(m) + "_k_" + str(k) + "_n_" + str(n) + "_" + error_method + "noiseLevel_" + str(noise_level).replace(".", "-") + "_repeatInd_" + str(repeat_ind) + "_" + str(estimate_method) + '_' + correct_method
            file_name = "../data/result/" + error_method + "/" + algorithm+"(ForM)"+ "/" + name_base + ".p"
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

def organize_result_for_figures(error_method,correct_method,noise_level, n, k_list, d_list):
    # k_dict = {16: r"k=16",0:r"tournament", "sqrt":r"$k=\sqrt{m-1}$"}
    # d = 100
    # m = n*2/k
    algorithms = ['No Correction']
    estimate_method = 'average'
    estimate_method_dict = {'MLE': 'MLE', 'average': "Stein's Estimate"}
    test_auc_array_list, test_noiseless_auc_array_list, error_array_list, error_normalized_array_list = [], [], [], []
    m_list_displayed = np.array(k_list) * n / 2
    d_list = list(d_list)
    for d in d_list:
        error_array, error_normalized_array, test_auc_array, test_noiseless_auc_array = nonparametric("No Correction",
                                                                                                      n, d,
                                                                                                      error_method,
                                                                                                      noise_level,
                                                                                                      k_list,
                                                                                                      estimate_method,
                                                                                                      correct_method)
        error_array_list.append(1 * error_array)
        error_normalized_array_list.append(1 * error_normalized_array)
        test_auc_array_list.append(1 * test_auc_array)
        test_noiseless_auc_array_list.append(1 * test_noiseless_auc_array)
    saved_dict = {"error_array_list": error_array_list, 'error_normalized_array_list': error_normalized_array_list,
                  'test_auc_array_list': test_auc_array_list, 'm_list_displayed': m_list_displayed,
                  'estimate_method': estimate_method}
    pickle.dump(saved_dict,
                open("../data/result/fix_n_vary_k_nl" + str(noise_level) + ".p", 'wb'))

def produce_figure(noise_level, n, d_list):
    f = pickle.load(open("../data/result/fix_n_vary_k_nl" + str(noise_level)  + ".p", 'rb'))
    # error_array_list = f['error_array_list']
    error_normalized_array_list = f['error_normalized_array_list']
    m_list_displayed = f['m_list_displayed']
    estimate_method = f['estimate_method']

    num_points = len(m_list_displayed)

    d_legends = ["d="+str(d) for d in d_list]
    common_fig_name_base = '_' + str(
        estimate_method) + str(noise_level).replace(".", "-")[-1] + '_n_'+str(n)


    ########################################################################################################################
    # Error Fig normalized \|beta-cbeta\|

    fig_name_base = "varyk_normalized_error"+common_fig_name_base
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
        plt.plot(m_list_displayed[:num_points], np.nanmean(error_normalized_array_list[i][:num_points, :], axis=1), linestyle=linestyes[linestyle_counter],
                 marker=markers[i],
                 markersize=5, linewidth=1.5,
                 c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
    plt.xscale("log",basex=10)
    plt.xlabel(r'$M$', fontsize=30)
    plt.ylim((0,0.55))
    xmin, xmax, ymin, ymax = plt.axis()
    # ref_x = np.arange(xmin, xmax, 10)
    # ref_y = 0.1 * np.ones(ref_x.shape)
    # plt.plot(ref_x, ref_y, '--', c='k')
    plt.vlines(n * np.log(n) ** 3, ymin, ymax, linestyles='dashed')
    plt.ylabel(r'$\|\|\frac{\hat{\beta}}{\|\|\hat{\beta}\|\|}-\frac{\beta}{\|\|\beta\|\|}\|\|$', fontsize=20)
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.legend(d_legends, fontsize=20)

    for i in range(len(error_normalized_array_list)):
        plt.fill_between(m_list_displayed[:num_points],
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
    noise_levels_dict = {'iid':[0.0,0.1,0.2],'BT':[0.41,0.185,0.095,0.042]}
    k_list = [i for i in range(2,10)]+[10*i for i in range(1,10)] +[100*i for i in range(1,11)] # + [100*i for i in range(1,6)]
    d_list = [10*i for i in range(1,11,2)]
    total_loops = len(error_methods)*len(correct_methods)*(len(noise_levels_dict['iid']))
    count = 0
    n = 1001
    ## Use once
    for error_method in error_methods:
        noise_levels = noise_levels_dict[error_method]
        for noise_level in noise_levels:
            for correct_method in correct_methods:
                temp = organize_result_for_figures(error_method,correct_method,noise_level, n, k_list, d_list)
    ## Use once ends here


    for error_method in error_methods:
        noise_levels = noise_levels_dict[error_method]
        for noise_level in noise_levels:
            for correct_method in correct_methods:
                count += 1
                print("Processing "+str(count)+"/"+str(total_loops))
                temp = produce_figure(noise_level,n,d_list)

    print("done")

