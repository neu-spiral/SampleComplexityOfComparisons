import pickle
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os.path


if __name__=="__main__":
    k = 16
    d = 100
    algorithms = ['No Correction', 'Oracle', '0.7 Oracle', 'greedyCycle']
    #error_methods = ['remove','flip']
    #noise_levels = [0.1,0.2,0.3,0.4,0.5,0.6] # iid
    #noise_levels = ["0.35" "0.15" "0.08" "0.03"] # BT
    # P error = [0.4036,0.3008,0.2018,0.1004]
    # alpha = [0.03,0.08,0.15,0.35]
    BT_alpha_Perror = {0.03:0.4036,0.08:0.3008,0.15:0.2018,0.35:0.1004}
    ind_list = range(15)
    repeat_ind_list = range(10)
    estimate_method = 'MLE'
    correct_method = 'remove'
    error_method = 'iid'
    noise_level = 0.4
    num_points = 10
    error_array_list, nte_array_list, nfe_array_list = [], [], []
    for i in range(len(algorithms)):
        error_array_list.append(np.zeros((len(ind_list),len(repeat_ind_list))))
        error_array_list[i][:] = np.nan
        nte_array_list.append(np.zeros((len(ind_list),len(repeat_ind_list))))
        nte_array_list[i][:] = np.nan
        nfe_array_list.append(np.zeros((len(ind_list),len(repeat_ind_list))))
        nfe_array_list[i][:] = np.nan
    #error_array_list = [np.zeros((len(ind_list),len(repeat_ind_list))).fill(np.nan) for i in range(len(algorithms))]
    #nte_array_list = [np.zeros((len(ind_list),len(repeat_ind_list))).fill(np.nan) for i in range(len(algorithms))]
    #nfe_array_list = [np.zeros((len(ind_list),len(repeat_ind_list))).fill(np.nan) for i in range(len(algorithms))]

    missing_files = []
    count_alg = 0
    for algorithm in algorithms:
        print "Processing "+str(count_alg+1)+"/"+str(len(algorithms))
        for ind in ind_list:
            for repeat_ind in repeat_ind_list:
                name_base = algorithm + "_k_" + str(k) + "_is_d_" + str(d) + "_" + error_method + "noiseLevel_" + str(
                    noise_level).replace(".", "-") + "_ind_" + str(ind) + "_repeatInd_" + str(repeat_ind) + '_' + str(
                    estimate_method) + '_' + correct_method
                file_name = "../result/"+error_method+"/"+algorithm+"/" + name_base + ".p"
                if os.path.isfile(file_name):
                    pickle_file = pickle.load(open(file_name, 'rb'))
                    n_accurate_list = pickle_file['n_list']
                    error_array_list[count_alg][ind,repeat_ind] = 1*pickle_file['error']
                    nte_array_list[count_alg][ind,repeat_ind] = 1*pickle_file['nte']
                    nfe_array_list[count_alg][ind,repeat_ind] = 1*pickle_file['nfe']
                else:
                    missing_files.append(file_name)
        count_alg += 1


    fig_name_base = '-'.join(algorithms) + "_k_" + str(k) + "_is_d_" + str(d) + "_" + error_method + "noiseLevel_" + str(noise_level).replace(".", "-")+ '_' + str(estimate_method) + '_' + correct_method
    sns_palette = sns.color_palette()
    sns_cmap = ListedColormap(sns_palette.as_hex())
    linestyes = ['-', '--', '-.', ':', '']
    markers = [".", 's', '^', 'D', 'p']
    fig_ave = plt.figure()
    for i in range(len(error_array_list)):
        plt.plot(n_accurate_list[:num_points], np.nanmean(error_array_list[i][:num_points,:],axis=1), linestyle='-', marker=markers[i],
                 markersize=5, linewidth=1.5,
                 c=sns_palette[i], markeredgecolor=sns_palette[i], markerfacecolor=sns_palette[i])
    plt.xscale("log")
    plt.xlabel('n', fontsize=20)
    plt.ylabel(r'$\|\|\beta-\hat{\beta}\|\|$', fontsize=20)
    plt.legend(algorithms, fontsize=15)
    if error_method == 'iid':
        plt.title('IID Noise (P='+str(noise_level)+') and '+estimate_method +' with '+str(correct_method), fontsize=15)
    elif error_method == 'BT':
        plt.title('Bradley Terry Noise (P=' + "%.1f"  % BT_alpha_Perror[noise_level] + ') and ' + estimate_method + ' with ' + str(correct_method), fontsize=15)

    for i in range(len(error_array_list)):
        plt.fill_between(n_accurate_list[:num_points], np.nanmean(error_array_list[i][:num_points,:],axis=1)-np.nanstd(error_array_list[i][:num_points,:],axis=1), np.nanmean(error_array_list[i][:num_points,:],axis=1)+np.nanstd(error_array_list[i][:num_points,:],axis=1),
                 color=sns_palette[i], alpha=0.3)
    fig_ave.savefig("../fig/"+fig_name_base+".pdf", bbox_inches='tight')

    # algorithms = ['No Correction', 'Oracle', '0.7 Oracle', 'MLE']
    color_ind = [0] + range(2, len(algorithms) + 1)
    num_error_list = [nte_array_list[a] for a in [0]+range(2, len(algorithms))]+[nfe_array_list[-1]] # Skip second one (Oracle) and add the false positive of algorihtm.
    fig_num_error = plt.figure()
    for i in range(len(num_error_list)):
        plt.plot(n_accurate_list[:num_points], np.mean(num_error_list[i][:num_points,:],axis=1), linestyle='-', marker=markers[color_ind[i]],
                 markersize=5, linewidth=1.5,
                 c=sns_palette[color_ind[i]], markeredgecolor=sns_palette[color_ind[i]], markerfacecolor=sns_palette[color_ind[i]])
    plt.xscale("log")
    plt.xlabel('n', fontsize=20)
    plt.ylabel('Number of Errors', fontsize=20)
    plt.legend(['True Error']+['0.7 Oracle True Positives']+[algorithms[-1]+' True Positives'] +[algorithms[-1]+' False Positives'], fontsize=15)
    if error_method == 'iid':
        plt.title('IID Noise (P='+str(noise_level)+') and '+estimate_method +' with '+str(correct_method), fontsize=15)
    elif error_method == 'BT':
        plt.title('Bradley Terry Noise (P=' + "%.1f" % BT_alpha_Perror[noise_level] + ') and ' + estimate_method + ' with ' + str(correct_method), fontsize=15)

    for i in range(len(num_error_list)):
        plt.fill_between(n_accurate_list[:num_points], np.mean(num_error_list[i][:num_points,:],axis=1)-np.std(num_error_list[i][:num_points,:],axis=1), np.mean(num_error_list[i][:num_points,:],axis=1)+np.std(num_error_list[i][:num_points,:],axis=1),
                 color=sns_palette[color_ind[i]], alpha=0.3)
    fig_num_error.savefig("../fig/num_error_" + fig_name_base + ".pdf", bbox_inches='tight')

    print "done"
