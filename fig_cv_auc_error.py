import numpy as np
import pickle
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os


def change_hyperparamter(algorithm,k,d,error_method,noise_level,ind,estimate_method,correct_method, thresholds ,repeat_ind = 0):
    missing_files = []
    threshold_error_array = np.zeros((len(thresholds), 5))
    threshold_test_auc_array = np.zeros((len(thresholds), 5))
    threshold_test_noiseless_auc_array = np.zeros((len(thresholds), 5))
    threshold_error_array[:] = np.nan
    threshold_test_auc_array[:] = np.nan
    threshold_test_noiseless_auc_array[:] = np.nan
    for ind_thre in range(len(thresholds)):
        for cv_index in range(5):
            name_base = algorithm + "_k_" + str(k) + "_is_d_" + str(d) + "_" + error_method + "noiseLevel_" + str(
                noise_level).replace(".", "-") + "_ind_" + str(ind) + "_repeatInd_" + str(repeat_ind) + '_cv_' + str(
                cv_index) + "_" + str(estimate_method) + '_' + correct_method
            file_name = "../result/Synthetic_CV_noiseless/" + error_method + "/" + algorithm + "/" + name_base + ".p"
            if os.path.isfile(file_name):
                    try:
                        pickle_file = pickle.load(open(file_name, 'rb'))
                    except:
                        pickle_file = loadmat(file_name[:-1]+"mat")
                    error_auc_array_temp = np.array(1 * pickle_file['beta_error_iter'])
                    removed_likelihood = np.array(1*pickle_file['removed_likelihood'])
                    try:
                        ind_row_best_test_auc = np.where(removed_likelihood<=thresholds[ind_thre])[0][-1]
                    except:
                        ind_row_best_test_auc = 0
                    threshold_error_array[ind_thre,cv_index] = error_auc_array_temp[ind_row_best_test_auc,0]
                    threshold_test_auc_array[ind_thre,cv_index] = error_auc_array_temp[ind_row_best_test_auc, 1]
                    threshold_test_noiseless_auc_array[ind_thre,cv_index] = error_auc_array_temp[ind_row_best_test_auc, 2]
            else:
                missing_files.append(file_name)
    return threshold_error_array, threshold_test_auc_array, threshold_test_noiseless_auc_array


def no_correction(k,d,error_method,noise_level,ind,estimate_method,correct_method ,repeat_ind = 0):
    algorithm = "No Correction"
    missing_files = []
    threshold_error_array = np.zeros((1, 5))
    threshold_test_auc_array = np.zeros((1, 5))
    threshold_test_noiseless_auc_array = np.zeros((1, 5))
    threshold_error_array[:] = np.nan
    threshold_test_auc_array[:] = np.nan
    threshold_test_noiseless_auc_array[:]= np.nan
    for cv_index in range(5):
        name_base = algorithm + "_k_" + str(k) + "_is_d_" + str(d) + "_" + error_method + "noiseLevel_" + str(
            noise_level).replace(".", "-") + "_ind_" + str(ind) + "_repeatInd_" + str(repeat_ind) + '_cv_' + str(
            cv_index) + "_" + str(estimate_method) + '_' + correct_method
        file_name = "../result/Synthetic_CV_noiseless/" + error_method + "/" + algorithm + "/" + name_base + ".p"
        if os.path.isfile(file_name):
            pickle_file = pickle.load(open(file_name, 'rb'))
            error_auc_tuple_temp = 1 * pickle_file['error']
            threshold_error_array[0, cv_index] = error_auc_tuple_temp[0]
            threshold_test_auc_array[0, cv_index] = error_auc_tuple_temp[1]
            threshold_test_noiseless_auc_array[0,cv_index] = error_auc_tuple_temp[2]
        else:
            missing_files.append(file_name)
    return threshold_error_array, threshold_test_auc_array, threshold_test_noiseless_auc_array



def produce_figure(error_method,correct_method,noise_level,ind,repeat_ind = 0):
    k, d = 0, 100
    estimate_method="MLE"
    algorithms = ['sortedGreedy', 'sortedGreedyCycle']
    BT_alpha_Perror = {0.042: 0.4021, 0.095: 0.3011, 0.185: 0.2004, 0.41: 0.1029}
    thresholds = np.arange(0.01,0.51,0.02)
    sns_palette = sns.color_palette()
    sns_cmap = ListedColormap(sns_palette.as_hex())
    linestyes = ['-', '--', '-.', '--', '']
    markers = [".", 's', '^', 'D', 'p']
    nc_error_array, nc_test_auc_array, nc_test_noiseless_auc_array = no_correction(k,d,error_method,noise_level,ind,estimate_method,correct_method)
    error_list = []
    auc_list = []
    auc_noiseless_list =[]
    for algorithm in algorithms:
        threshold_error_array, threshold_test_auc_array, threshold_test_noiseless_auc_array = change_hyperparamter(algorithm,k,d,error_method,noise_level,ind,estimate_method,correct_method, thresholds)
        error_list.append(1*threshold_error_array)
        auc_list.append(1*threshold_test_auc_array)
        auc_noiseless_list.append(1*threshold_test_noiseless_auc_array)
    fig_name_base = error_method + "noiseLevel_" + str(noise_level).replace(".", "-") + '_' + str(
        estimate_method) + '_' + correct_method + "_" + '-'.join(algorithms) + "_k_" + str(k) + "_is_d_" + str(
        d)
    #########################################################################################################
    # Fig Error
    plt.switch_backend('agg')
    fig_error = plt.figure()
    i = 0
    count = 0
    for i in [0]+range(2,2+len(algorithms)):
        if i == 0:
            plt.plot(thresholds,np.nanmean(nc_error_array) * np.ones_like(thresholds),linestyle='-',
                 marker=markers[i],
                 markersize=5, linewidth=1.5,
                 c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
        else:
            plt.plot(thresholds, np.nanmean(error_list[count], axis=1),
                 linestyle='-',
                 marker=markers[i],
                 markersize=5, linewidth=1.5,
                 c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
            count += 1
    plt.xlabel('Thresholds', fontsize=20)
    plt.ylabel(r'$\|\|\beta-\hat{\beta}\|\|$', fontsize=20)
    algorithms_legend = ["No Correction"] + algorithms
    noise_digit = "%.1f" % BT_alpha_Perror[noise_level] if error_method == 'BT' else "%.1f" % noise_level
    plt.title(error_method + ' Noise (P=' + noise_digit + r') at n = 10^4 ' + 'MLE with ' + correct_method, fontsize=15)
    plt.legend(algorithms_legend,fontsize=10)
    count = 0
    for i in [0]+range(2,2+len(algorithms)):
        if i == 0:
            plt.fill_between(np.array(thresholds),
                         np.ones(len(thresholds),)*(np.nanmean(nc_error_array, axis=1) - np.nanstd(nc_error_array, axis=1)),
                         np.ones(len(thresholds),)*(np.nanmean(nc_error_array, axis=1) + np.nanstd(nc_error_array, axis=1)),
                         color=np.array(sns_palette[i]), alpha=0.3)
        else:
            plt.fill_between(np.array(thresholds),
                             np.nanmean(error_list[count], axis=1) - np.nanstd(error_list[count], axis=1),
                             np.nanmean(error_list[count], axis=1) + np.nanstd(error_list[count], axis=1),
                             color=np.array(sns_palette[i]), alpha=0.3)
            count += 1
    fig_error.savefig("../fig/"+fig_name_base+ "cv_prob_error"+".pdf", bbox_inches='tight')
    plt.close(fig_error)

    #########################################################################################################
    # Test AUC
    plt.switch_backend('agg')
    fig_auc = plt.figure()
    i = 0
    count = 0
    for i in [0] + range(2, 2 + len(algorithms)):
        if i == 0:
            plt.plot(thresholds, np.nanmean(nc_test_auc_array) * np.ones_like(thresholds), linestyle='--',
                     marker=markers[i],
                     markersize=5, linewidth=1.5,
                     c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])

        else:
            plt.plot(thresholds, np.nanmean(auc_list[count], axis=1),
                     linestyle='--',
                     marker=markers[i],
                     markersize=5, linewidth=1.5,
                     c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
            count += 1
    plt.xlabel('Thresholds', fontsize=20)
    plt.ylabel(r'AUC on Noisy Labels', fontsize=20)
    algorithms_legend = ["No Correction"] + algorithms
    noise_digit = "%.1f" % BT_alpha_Perror[noise_level] if error_method == 'BT' else "%.1f" % noise_level
    plt.title(error_method + ' Noise (P=' + noise_digit + r') at n = 10^4 ' + 'MLE with ' + correct_method, fontsize=15)
    plt.legend(algorithms_legend, fontsize=10)
    count = 0
    for i in [0] + range(2, 2 + len(algorithms)):
        if i == 0:
            plt.fill_between(np.array(thresholds),
                             np.ones(len(thresholds),)*(np.nanmean(nc_test_auc_array, axis=1) - np.nanstd(nc_test_auc_array, axis=1)),
                             np.ones(len(thresholds),)*(np.nanmean(nc_test_auc_array, axis=1) + np.nanstd(nc_test_auc_array, axis=1)),
                             color=sns_palette[i], alpha=0.3)
        else:
            plt.fill_between(thresholds,
                             np.nanmean(auc_list[count], axis=1) - np.nanstd(auc_list[count], axis=1),
                             np.nanmean(auc_list[count], axis=1) + np.nanstd(auc_list[count], axis=1),
                             color=sns_palette[i], alpha=0.3)
            count += 1

    fig_auc.savefig("../fig/"+fig_name_base+ "cv_prob_test_auc"+".pdf", bbox_inches='tight')
    plt.close(fig_auc)
    #########################################################################################################
    # Noiseless AUC
    plt.switch_backend('agg')
    fig_noiseless_auc = plt.figure()
    i = 0
    count = 0
    for i in [0] + range(2, 2 + len(algorithms)):
        if i == 0:
            plt.plot(np.array(thresholds), np.nanmean(nc_test_noiseless_auc_array) * np.ones_like(thresholds), linestyle='--',
                     marker=markers[i],
                     markersize=5, linewidth=1.5,
                     c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])

        else:
            plt.plot(thresholds, np.nanmean(auc_noiseless_list[count], axis=1),
                     linestyle='--',
                     marker=markers[i],
                     markersize=5, linewidth=1.5,
                     c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
            count += 1
    plt.xlabel('Thresholds', fontsize=20)
    plt.ylabel(r'AUC on Noiseless Labels', fontsize=20)
    algorithms_legend = ["No Correction"] + algorithms
    noise_digit = "%.1f" % BT_alpha_Perror[noise_level] if error_method == 'BT' else "%.1f" % noise_level
    plt.title(error_method + ' Noise (P=' + noise_digit + r') at n = 10^4 ' + 'MLE with ' + correct_method, fontsize=15)
    plt.legend(algorithms_legend, fontsize=10)
    count = 0
    for i in [0] + range(2, 2 + len(algorithms)):
        if i == 0:
            plt.fill_between(np.array(thresholds),
                             np.ones(len(thresholds),)*(np.nanmean(nc_test_noiseless_auc_array, axis=1) - np.nanstd(nc_test_noiseless_auc_array, axis=1)),
                             np.ones(len(thresholds),)*(np.nanmean(nc_test_noiseless_auc_array, axis=1) + np.nanstd(nc_test_noiseless_auc_array, axis=1)),
                             color=sns_palette[i], alpha=0.3)
        else:
            plt.fill_between(thresholds,
                             np.nanmean(auc_noiseless_list[count], axis=1) - np.nanstd(auc_noiseless_list[count], axis=1),
                             np.nanmean(auc_noiseless_list[count], axis=1) + np.nanstd(auc_noiseless_list[count], axis=1),
                             color=sns_palette[i], alpha=0.3)
            count += 1

    fig_noiseless_auc.savefig("../fig/" + fig_name_base + "cv_prob_test_noiseless_auc" + ".pdf", bbox_inches='tight')
    plt.close(fig_noiseless_auc)
    return


if __name__=="__main__":
    # BT_alpha_Perror = {0.042: 0.4021, 0.095: 0.3011, 0.185: 0.2004, 0.41: 0.1029}
    ind = range(10)[6]  # Only 10^4 considered.
    error_methods = ['iid','BT']
    correct_methods = ['remove','flip']
    noise_levels_dict = {'iid':[0.2,0.3,0.4],'BT':[0.185,0.095,0.042]}
    total_loops = len(error_methods)*len(correct_methods)*(len(noise_levels_dict['iid']))
    count = 0
    for error_method in error_methods:
        noise_levels = noise_levels_dict[error_method]
        for noise_level in noise_levels:
            for correct_method in correct_methods:
                count += 1
                print("Processing "+str(count)+"/"+str(total_loops))
                temp = produce_figure(error_method,correct_method,noise_level,ind)
                a = 1



    print "done"
