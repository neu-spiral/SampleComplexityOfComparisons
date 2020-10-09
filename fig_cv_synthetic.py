import pickle
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os.path
from scipy.io import loadmat


def nonparametric(algorithm,k,d,error_method,noise_level,ind_list,estimate_method,correct_method,repeat_ind = 0):
    error_array = np.zeros((len(ind_list),5))
    error_array[:] = np.nan
    test_auc_array = np.zeros((len(ind_list),5))
    test_auc_array[:] = np.nan
    test_noiseless_auc_array = np.zeros((len(ind_list), 5))
    test_noiseless_auc_array[:] = np.nan
    missing_files = []
    for ind in ind_list:
        for cv_index in range(5):
            name_base = algorithm + "_k_" + str(k) + "_is_d_" + str(d) + "_" + error_method + "noiseLevel_" + str(
                noise_level).replace(".", "-") + "_ind_" + str(ind) + "_repeatInd_" + str(repeat_ind) + '_cv_' + str(
                cv_index) + "_" + str(estimate_method) + '_' + correct_method
            file_name = "../result/Synthetic_CV_noiseless/" + error_method + "/" + algorithm + "/" + name_base + ".p"
            if os.path.isfile(file_name):
                    pickle_file = pickle.load(open(file_name, 'rb'))
                    n_accurate_list = pickle_file['n_list']
                    error_array[ind, cv_index] = 1 * pickle_file['error'][0]
                    test_auc_array[ind, cv_index] = 1 * pickle_file['error'][1]
                    test_noiseless_auc_array[ind, cv_index] = 1 * pickle_file['error'][2]
            else:
                missing_files.append(file_name)
    return error_array, test_auc_array, test_noiseless_auc_array, n_accurate_list

def best_hyperparamter(algorithm,k,d,error_method,noise_level,ind_list,estimate_method,correct_method, thresholds ,repeat_ind = 0):
    error_array = np.zeros((len(ind_list),5))
    test_auc_array = np.zeros((len(ind_list),5))
    test_noiseless_auc_array = np.zeros((len(ind_list),5))
    missing_files = []
    for ind in ind_list:
        threshold_error_array = np.zeros((len(thresholds), 5))
        threshold_test_auc_array = np.zeros((len(thresholds), 5))
        threshold_test_noiseless_auc_array = np.zeros((len(thresholds), 5))
        for ind_thre in range(len(thresholds)):
            for cv_index in range(5):
                name_base = algorithm + "_k_" + str(k) + "_is_d_" + str(d) + "_" + error_method + "noiseLevel_" + str(
                    noise_level).replace(".", "-") + "_ind_" + str(ind) + "_repeatInd_" + str(repeat_ind) + '_cv_' + str(
                    cv_index) + "_" + str(estimate_method) + '_' + correct_method
                file_name = "../result/Synthetic_CV_noiseless/" + error_method + "/" + algorithm + "/" + name_base + ".p"
                if os.path.isfile(file_name):
                        pickle_file = pickle.load(open(file_name, 'rb'))
                        n_accurate_list = pickle_file['n_list']
                        error_auc_array_temp = np.array(1 * pickle_file['beta_error_iter'])
                        removed_likelihood = np.array(1*pickle_file['removed_likelihood'])
                        try:
                            ind_row_best_test_auc = np.where(removed_likelihood<=thresholds[ind_thre])[0][-1]
                        except:
                            print(np.where(removed_likelihood[:, 0] <= thresholds[ind_thre])[0])
                            continue
                        threshold_error_array[ind_thre,cv_index] = error_auc_array_temp[ind_row_best_test_auc,0]
                        threshold_test_auc_array[ind_thre,cv_index] = error_auc_array_temp[ind_row_best_test_auc, 1]
                        threshold_test_noiseless_auc_array[ind_thre,cv_index] = error_auc_array_temp[ind_row_best_test_auc, 2]
                else:
                    missing_files.append(file_name)
        ind_best_threshold = np.argmax(np.nanmean(threshold_test_auc_array,axis=1))
        error_array[ind,:] = 1*threshold_error_array[ind_best_threshold,:]
        test_auc_array[ind,:] = 1*threshold_test_auc_array[ind_best_threshold,:]
        ind_best_noiseless_threshold = np.argmax(np.nanmean(threshold_test_noiseless_auc_array, axis=1))
        test_noiseless_auc_array[ind,:] = 1*threshold_test_noiseless_auc_array[ind_best_noiseless_threshold,:]
    return error_array, test_auc_array, test_noiseless_auc_array, n_accurate_list


def produce_figure(error_method,correct_method,noise_level):
    k = "sqrt"
    k_dict = {16: r"k=16",0:r"tournament", "sqrt":r"$k=\sqrt{m-1}$"}
    d = 100
    algorithms = ['No Correction', 'Oracle', 'sortedGreedy', 'sortedGreedyCycle']
    mat_algorithms = [' ']  # greedyMultiCycleBeta
    # error_methods = ['remove','flip']
    # noise_levels = [0.1,0.2,0.3,0.4,0.5,0.6] # iid
    # noise_levels = ["0.35" "0.15" "0.08" "0.03"] # BT
    # P error = [0.4036,0.3008,0.2018,0.1004]
    # alpha = [0.03,0.08,0.15,0.35]
    BT_alpha_Perror = {0.042: 0.4021, 0.095: 0.3011, 0.185: 0.2004, 0.41: 0.1029}
    ind_list = range(10)
    cv_ind_list = range(5)
    estimate_method = 'MLE'
    estimate_method_dict = {'MLE': 'MLE', 'average': "Stein's Estimate"}
    # correct_method = 'remove'
    # error_method = 'iid'
    # noise_level = 0.2
    num_points = 10
    test_auc_array_list, test_noiseless_auc_array_list,error_array_list = [], [], []
    missing_files = []
    count_alg = 0
    thresholds = np.arange(0.15, 0.6, 0.05)
    for algorithm in algorithms[:]:
        # print("Processing " + str(count_alg + 1) + "/" + str(len(algorithms)))
        if algorithm in ["No Correction", "Oracle"]:
            error_array, test_auc_array, test_noiseless_auc_array,n_accurate_list = nonparametric(algorithm, k, d, error_method, noise_level,
                                                                         ind_list, estimate_method, correct_method)
        else:
            error_array, test_auc_array, test_noiseless_auc_array,n_accurate_list = best_hyperparamter(algorithm, k, d, error_method,
                                                                              noise_level, ind_list, estimate_method,
                                                                              correct_method, thresholds)
        error_array_list.append(1 * error_array)
        test_auc_array_list.append(1 * test_auc_array)
        test_noiseless_auc_array_list.append(1*test_noiseless_auc_array)

    ########################################################################################################################
    # Test AUC Fig
    fig_name_base =  error_method + "noiseLevel_" + str(noise_level).replace(".", "-") + '_' + str(
        estimate_method) + '_' + correct_method+ "_"+'-'.join(algorithms) + "_k_" + str(k) + "_is_d_" + str(d) +"cv_test_auc"
    sns_palette = sns.color_palette()
    sns_cmap = ListedColormap(sns_palette.as_hex())
    linestyes = ['-', '--', '-.', ':', '']
    markers = [".", 's', '^', 'D', 'p']
    plt.switch_backend('agg')
    fig_test_auc = plt.figure()
    for i in range(len(test_auc_array_list)):
        plt.plot(n_accurate_list[:num_points], np.nanmean(test_auc_array_list[i][:num_points, :], axis=1),
                 linestyle='-',
                 marker=markers[i],
                 markersize=5, linewidth=1.5,
                 c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
    plt.xscale("log")
    plt.xlabel('n', fontsize=20)
    plt.ylabel(r'AUC on Noisy Labels', fontsize=20)
    # algorithms_legend = ['No Correction', 'Oracle', '0.7 Oracle', 'GreedyCycleCorrection']
    plt.legend(algorithms, fontsize=15)
    if error_method == 'iid':
        plt.title('IID Noise (P=' + str(noise_level) + ') and ' + estimate_method + ' in '+k_dict[k] + ' with ' + str(correct_method),
                  fontsize=12)
    elif error_method == 'BT':
        plt.title('Bradley Terry Noise (P=' + "%.1f" % BT_alpha_Perror[
            noise_level] + ') and ' + estimate_method + ' in '+k_dict[k]+ ' with ' + str(correct_method), fontsize=15)

    for i in range(len(test_auc_array_list)):
        plt.fill_between(n_accurate_list[:num_points],
                         np.nanmean(test_auc_array_list[i][:num_points, :], axis=1) - np.nanstd(
                             test_auc_array_list[i][:num_points, :], axis=1),
                         np.nanmean(test_auc_array_list[i][:num_points, :], axis=1) + np.nanstd(
                             test_auc_array_list[i][:num_points, :], axis=1),
                         color=sns_palette[i], alpha=0.3)
    fig_test_auc.savefig("../fig/" + fig_name_base + ".pdf", bbox_inches='tight')

    # ########################################################################################################################
    # # Test Noiseless AUC Fig
    # fig_name_base = error_method + "noiseLevel_" + str(noise_level).replace(".", "-") + '_' + str(
    #     estimate_method) + '_' + correct_method + "_" + '-'.join(algorithms) + "_k_" + str(k) + "_is_d_" + str(
    #     d) + "cv_test_noiseless_auc"
    # sns_palette = sns.color_palette()
    # sns_cmap = ListedColormap(sns_palette.as_hex())
    # linestyes = ['-', '--', '-.', ':', '']
    # markers = [".", 's', '^', 'D', 'p']
    # plt.switch_backend('agg')
    # fig_test_noiseless_auc = plt.figure()
    # for i in range(len(test_noiseless_auc_array_list)):
    #     plt.plot(n_accurate_list[:num_points], np.nanmean(test_noiseless_auc_array_list[i][:num_points, :], axis=1),
    #              linestyle='-',
    #              marker=markers[i],
    #              markersize=5, linewidth=1.5,
    #              c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
    # plt.xscale("log")
    # plt.xlabel('n', fontsize=20)
    # plt.ylabel(r'AUC on Noiseless Labels', fontsize=20)
    # # algorithms_legend = ['No Correction', 'Oracle', '0.7 Oracle', 'GreedyCycleCorrection']
    # plt.legend(algorithms, fontsize=15)
    # if error_method == 'iid':
    #     plt.title('IID Noise (P=' + str(noise_level) + ') and ' + estimate_method + ' with ' + str(correct_method),
    #               fontsize=15)
    # elif error_method == 'BT':
    #     # plt.title('Bradley Terry Noise (P=' + "%.1f"  % BT_alpha_Perror[noise_level] + ') and ' + estimate_method + ' with ' + str(correct_method), fontsize=15)
    #     plt.title('Bradley Terry Noise (P=' + "%.1f" % BT_alpha_Perror[
    #         noise_level] + ') and ' + estimate_method + ' with ' + str(correct_method), fontsize=15)
    #
    # for i in range(len(test_noiseless_auc_array_list)):
    #     plt.fill_between(n_accurate_list[:num_points],
    #                      np.nanmean(test_noiseless_auc_array_list[i][:num_points, :], axis=1) - np.nanstd(
    #                          test_noiseless_auc_array_list[i][:num_points, :], axis=1),
    #                      np.nanmean(test_noiseless_auc_array_list[i][:num_points, :], axis=1) + np.nanstd(
    #                          test_noiseless_auc_array_list[i][:num_points, :], axis=1),
    #                      color=sns_palette[i], alpha=0.3)
    # fig_test_noiseless_auc.savefig("../fig/" + fig_name_base + ".pdf", bbox_inches='tight')
    ########################################################################################################################
    # Error Fig

    fig_name_base = error_method + "noiseLevel_" + str(noise_level).replace(".", "-") + '_' + str(
        estimate_method) + '_' + correct_method + "_" + '-'.join(algorithms) + "_k_" + str(k) + "_is_d_" + str(
        d) + "cv_error"
    sns_palette = sns.color_palette()
    sns_cmap = ListedColormap(sns_palette.as_hex())
    linestyes = ['-', '--', '-.', ':', '']
    markers = [".", 's', '^', 'D', 'p']
    plt.switch_backend('agg')
    fig_error = plt.figure()
    for i in range(len(error_array_list)):
        plt.plot(n_accurate_list[:num_points], np.nanmean(error_array_list[i][:num_points, :], axis=1), linestyle='-',
                 marker=markers[i],
                 markersize=5, linewidth=1.5,
                 c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
    plt.xscale("log")
    plt.xlabel('n', fontsize=20)
    plt.ylabel(r'$\|\|\beta-\hat{\beta}\|\|$', fontsize=20)
    # algorithms_legend = ['No Correction', 'Oracle', '0.7 Oracle', 'GreedyCycleCorrection']
    plt.legend(algorithms, fontsize=15)
    if error_method == 'iid':
        plt.title('IID Noise (P=' + str(noise_level) + ') and ' + estimate_method+ ' in '+k_dict[k] + ' with ' + str(correct_method),
                  fontsize=12)
    elif error_method == 'BT':
        plt.title('Bradley Terry Noise (P=' + "%.1f" % BT_alpha_Perror[
            noise_level] + ') and ' + estimate_method+ ' in '+k_dict[k] + ' with ' + str(correct_method), fontsize=15)

    for i in range(len(error_array_list)):
        plt.fill_between(n_accurate_list[:num_points],
                         np.nanmean(error_array_list[i][:num_points, :], axis=1) - np.nanstd(
                             error_array_list[i][:num_points, :], axis=1),
                         np.nanmean(error_array_list[i][:num_points, :], axis=1) + np.nanstd(
                             error_array_list[i][:num_points, :], axis=1),
                         color=sns_palette[i], alpha=0.3)
    fig_error.savefig("../fig/" + fig_name_base + ".pdf", bbox_inches='tight')
    return


if __name__=="__main__":
    # BT_alpha_Perror = {0.042: 0.4021, 0.095: 0.3011, 0.185: 0.2004, 0.41: 0.1029}
    error_methods = ['iid','BT']
    correct_methods = ['remove','flip']
    noise_levels_dict = {'iid':[0.1,0.2,0.3,0.4],'BT':[0.41,0.185,0.095,0.042]}
    total_loops = len(error_methods)*len(correct_methods)*(len(noise_levels_dict['iid']))
    temp = produce_figure('iid', 'remove', 0.4)
    count = 0
    for error_method in error_methods:
        noise_levels = noise_levels_dict[error_method]
        for noise_level in noise_levels:
            for correct_method in correct_methods:
                count += 1
                print("Processing "+str(count)+"/"+str(total_loops))
                try:
                    temp = produce_figure(error_method,correct_method,noise_level)
                except:
                    continue
    print("done")
