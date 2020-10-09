import pickle
from networkx.classes.multidigraph import MultiDiGraph
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from minimumFeedbackArcSet import binary_accuracy,greedyFAS,ILSR,kwik_sort_fas,ConvexPLModel, average, MLE,MAP, greedy_break_cycles,\
    construct_graph, greedy_random_cycles, greedy_break_cycles_update_beta, greedy_break_cycles_uncertain, greedy_break_cycles_flip_remove,\
    sorted_greedy_cycle, sorted_greedy
from sklearn.metrics import roc_auc_score
import sys
from scipy.stats import mode
import random
from sklearn.model_selection import KFold

def check_all_comparisons_in_the_same_set(class_label_array, comparison_label_array,cmp_tuples):
    # img_name_list = partition_file['order_name'].values()
    class_label_list = list(class_label_array)
    for cmp_ind in list(comparison_label_array):
        img_i_name_ind, img_j_name_ind = cmp_tuples[cmp_ind][0], cmp_tuples[cmp_ind][1]
        if img_i_name_ind in class_label_list and  img_j_name_ind in class_label_list:
            continue
        else:
            return False
    return True

def filter_cmp_from_imgs(cmp_pair_array, indices_imgs_list):
    # Return filtered array
    cmp_filtered_list = []
    for i in range(cmp_pair_array.shape[0]):
        if cmp_pair_array[i,0] in indices_imgs_list and cmp_pair_array[i,1] in indices_imgs_list:
            cmp_filtered_list.append(i)
    return np.array(cmp_filtered_list)


def ROP_subsample_partition(num_sub_imgs, rop_folder_path='../data/ROP/',name_base = '../data/ROP/comparison_complexity_subsample'):
    """
        This function save the partition file for subsampling comparison complexity work.
        :param rop_folder_path: This folder contains the data used in previous paper. Logistic regression's partition[0], u_net feature
        :param rop_comparison_complexity_partition_file: This is the output pickle file
        :return: the string of the output file
    """
    rop_comparison_complexity_partition_file = name_base + "_"+str(num_sub_imgs)+".p"
    repeat_index = 0
    num_of_expert = 5
    feature_file_path = '100Features_ordered.p'  # Unet Segmentation
    feat_file = pickle.load(open(rop_folder_path + feature_file_path, 'rb'))
    # cmp_feat = feat_file['cmpFeat'][:,:-1]
    class_feat = feat_file['labelFeat'][:-1, :-1]
    partition_pickle_file_path = 'Partitions.p'
    partition_mat_file_path = 'iROP_6DD_1st100_Partition.mat'
    partition_pickle_file = pickle.load(open(rop_folder_path + partition_pickle_file_path, 'rb'))
    partition_mat_file = loadmat(rop_folder_path + partition_mat_file_path)
    cmp_data = partition_pickle_file['cmpData']
    cmp_tuples = cmp_data[0]
    order_name = partition_pickle_file['orderName']
    img_name_list = order_name.values()
    cmp_pair_single_expert = [(img_name_list.index(img_i_name), img_name_list.index(img_j_name)) for
                              (img_i_name, img_j_name, cmp_label) in cmp_tuples]
    cmp_pair_array_single_expert = np.array(cmp_pair_single_expert,
                                            dtype=np.int)  # 5941 by 2 array. Each row (img_i_ind, img_j_ind) are indices in img_name_list
    cmp_feat = class_feat[cmp_pair_array_single_expert[:, 0], :] - class_feat[cmp_pair_array_single_expert[:, 1], :]
    cmp_label_multi_expert = []
    for ind_expert in range(num_of_expert):
        cmp_label_single_expert = [cmp_label for (img_i_name, img_j_name, cmp_label) in cmp_data[ind_expert]]
        cmp_label_multi_expert.append(cmp_label_single_expert[:])
    cmp_label_array_multi_expert = np.array(cmp_label_multi_expert, dtype=np.int).T  #
    # 5941 by 5 array. Each row is a comparison label corresponds to cmp_pair_array_single_expert. Each column is an experts's comparison label.
    all_80_RSD_train_plus_partition = partition_mat_file['RSDTrainPlusPartition'][repeat_index, :]
    RSD_test_plus_partition = partition_mat_file['RSDTestPlusPartition'][repeat_index, :]
        # RSD_val_plus_list = []
    RSD_train_plus_list = []
    cmp_train_plus_partition_list = []
    np.random.seed(0)
    for fold_index in range(5):
        print "Processing "+str(num_sub_imgs)+" in fold "+str(fold_index)
        # remaining_folds = range(5)
        # remaining_folds.remove(fold_index)
        # val_index = fold_index + 1
        # if val_index == 5:
        #     val_index = 0
        # RSD_val_plus_list.append(1*RSD_test_plus_partition[val_index,:])
        # remaining_folds.remove(val_index)
        # num_folds_to_train = int(num_sub_imgs / 20)
        # random.seed(fold_index)
        # random.shuffle(remaining_folds)
        # train_fold_indices = remaining_folds[:num_folds_to_train]
        # temp_train_indices_list = [1*RSD_test_plus_partition[i,:] for i in train_fold_indices]
        all_80_train_indices = all_80_RSD_train_plus_partition[fold_index,:]
        RSD_train_plus_list.append(np.sort(np.random.choice(all_80_train_indices,num_sub_imgs,replace=False)))
        temp_cmp_train_indices = filter_cmp_from_imgs(cmp_pair_array_single_expert, RSD_train_plus_list[-1].tolist())
        cmp_train_plus_partition_list.append(1*temp_cmp_train_indices)
    RSD_train_plus_partition = np.array(RSD_train_plus_list)

    # RSD_train_plus_partition = partition_mat_file['RSDTrainPlusPartition'][repeat_index, :]
    RSD_test_plus_partition = partition_mat_file['RSDTestPlusPartition'][repeat_index, :]
    # cmp_train_plus_partition_object = partition_mat_file['cmpTrainPlusPartition'][repeat_index, :]
    # cmp_train_plus_partition_list = [cmp_train_plus_partition_object[i][0, :] for i in
    #                                  range(cmp_train_plus_partition_object.shape[0])]
    cmp_test_plus_partition_object = partition_mat_file['cmpTestPlusPartition'][repeat_index, :]
    cmp_test_plus_partition_list = [cmp_test_plus_partition_object[i][0, :] for i in
                                    range(cmp_test_plus_partition_object.shape[0])]
    # check_all_comparisons_in_the_same_set(RSD_train_plus_partition[4,:], cmp_train_plus_partition[4][0,:],cmp_tuples,img_name_list)
    # 1 Plus 2 Preplus, 3 Normal
    output_dict = {'class_feat': class_feat, 'RSD_labels': partition_pickle_file['RSDLabels'],
                   'class_labels': partition_pickle_file['label13'],
                   'cmp_feat': cmp_feat, 'cmp_label_array_multi_expert': cmp_label_array_multi_expert,
                   'cmp_pair_array_single_expert': cmp_pair_array_single_expert,
                   'order_name': order_name,
                   'RSD_train_plus_partition': RSD_train_plus_partition,
                   'RSD_test_plus_partition': RSD_test_plus_partition,
                   'cmp_train_plus_partition_list': cmp_train_plus_partition_list,
                   'cmp_test_plus_partition_list': cmp_test_plus_partition_list}
    pickle.dump(output_dict, open(rop_comparison_complexity_partition_file, 'wb'))
    return rop_comparison_complexity_partition_file

def ROP_subsample_standard_partition(num_sub_cmps, rop_folder_path='../data/ROP/',name_base = '../data/ROP/comparison_complexity_st_subsample'):
    """
        This uses the standard cross validation for comparisons.
        This function save the partition file for subsampling comparison complexity work.
        :param rop_folder_path: This folder contains the data used in previous paper. Logistic regression's partition[0], u_net feature
        :param rop_comparison_complexity_partition_file: This is the output pickle file
        :return: the string of the output file
    """
    rop_comparison_complexity_partition_file = name_base + "_"+str(num_sub_cmps)+".p"
    repeat_index = 0
    num_of_expert = 5
    feature_file_path = '100Features_ordered.p'  # Unet Segmentation
    feat_file = pickle.load(open(rop_folder_path + feature_file_path, 'rb'))
    # cmp_feat = feat_file['cmpFeat'][:,:-1]
    class_feat = feat_file['labelFeat'][:-1, :-1]
    partition_pickle_file_path = 'Partitions.p'
    partition_mat_file_path = 'iROP_6DD_1st100_Partition.mat'
    partition_pickle_file = pickle.load(open(rop_folder_path + partition_pickle_file_path, 'rb'))
    partition_mat_file = loadmat(rop_folder_path + partition_mat_file_path)
    cmp_data = partition_pickle_file['cmpData']
    cmp_tuples = cmp_data[0]
    order_name = partition_pickle_file['orderName']
    img_name_list = order_name.values()
    cmp_pair_single_expert = [(img_name_list.index(img_i_name), img_name_list.index(img_j_name)) for
                              (img_i_name, img_j_name, cmp_label) in cmp_tuples]
    cmp_pair_array_single_expert = np.array(cmp_pair_single_expert,
                                            dtype=np.int)  # 5941 by 2 array. Each row (img_i_ind, img_j_ind) are indices in img_name_list
    cmp_feat = class_feat[cmp_pair_array_single_expert[:, 0], :] - class_feat[cmp_pair_array_single_expert[:, 1], :]
    cmp_label_multi_expert = []
    for ind_expert in range(num_of_expert):
        cmp_label_single_expert = [cmp_label for (img_i_name, img_j_name, cmp_label) in cmp_data[ind_expert]]
        cmp_label_multi_expert.append(cmp_label_single_expert[:])
    cmp_label_array_multi_expert = np.array(cmp_label_multi_expert, dtype=np.int).T  #
    # 5941 by 5 array. Each row is a comparison label corresponds to cmp_pair_array_single_expert. Each column is an experts's comparison label.
    all_80_RSD_train_plus_partition = partition_mat_file['RSDTrainPlusPartition'][repeat_index, :]
    RSD_test_plus_partition = partition_mat_file['RSDTestPlusPartition'][repeat_index, :]
        # RSD_val_plus_list = []
    RSD_train_plus_list, RSD_test_plus_list = [],[]
    cmp_train_plus_partition_list, cmp_test_plus_partition_list = [], []
    np.random.seed(0)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index_all, test_index in kf.split(cmp_feat):
        if num_sub_cmps == 0:
            train_index = 1* train_index_all
        else:
            train_index = np.sort(np.random.choice(train_index_all,num_sub_cmps,replace=False))
        cmp_train_plus_partition_list.append(1*train_index)
        RSD_train_plus_list.append(1*np.sort(np.unique(cmp_pair_array_single_expert[train_index])))
        cmp_test_plus_partition_list.append(1*test_index)
        RSD_test_plus_list.append(1*np.sort(np.unique(cmp_pair_array_single_expert[test_index])))
    RSD_train_plus_partition = np.array(RSD_train_plus_list)
    RSD_test_plus_partition = np.array(RSD_test_plus_list)

    # check_all_comparisons_in_the_same_set(RSD_train_plus_partition[4,:], cmp_train_plus_partition[4][0,:],cmp_tuples,img_name_list)
    # 1 Plus 2 Preplus, 3 Normal
    output_dict = {'class_feat': class_feat, 'RSD_labels': partition_pickle_file['RSDLabels'],
                   'class_labels': partition_pickle_file['label13'],
                   'cmp_feat': cmp_feat, 'cmp_label_array_multi_expert': cmp_label_array_multi_expert,
                   'cmp_pair_array_single_expert': cmp_pair_array_single_expert,
                   'order_name': order_name,
                   'RSD_train_plus_partition': RSD_train_plus_partition,
                   'RSD_test_plus_partition': RSD_test_plus_partition,
                   'cmp_train_plus_partition_list': cmp_train_plus_partition_list,
                   'cmp_test_plus_partition_list': cmp_test_plus_partition_list}
    pickle.dump(output_dict, open(rop_comparison_complexity_partition_file, 'wb'))
    return rop_comparison_complexity_partition_file

def ROP_generate_partition(rop_folder_path='../data/ROP/', rop_comparison_complexity_partition_file = '../data/ROP/comparison_complexity.p'):
    """
    This function save the partition file for comparison complexity work.
    :param rop_folder_path: This folder contains the data used in previous paper. Logistic regression's partition[0], u_net feature
    :param rop_comparison_complexity_partition_file: This is the output pickle file
    :return: the string of the output file
    """
    repeat_index = 0
    num_of_expert = 5
    feature_file_path = '100Features_ordered.p'  # Unet Segmentation
    feat_file = pickle.load(open(rop_folder_path + feature_file_path, 'rb'))
    # cmp_feat = feat_file['cmpFeat'][:,:-1]
    class_feat = feat_file['labelFeat'][:-1,:-1]
    partition_pickle_file_path = 'Partitions.p'
    partition_mat_file_path = 'iROP_6DD_1st100_Partition.mat'
    partition_pickle_file = pickle.load(open(rop_folder_path + partition_pickle_file_path, 'rb'))
    partition_mat_file = loadmat(rop_folder_path+partition_mat_file_path)
    cmp_data = partition_pickle_file['cmpData']
    cmp_tuples = cmp_data[0]
    order_name = partition_pickle_file['orderName']
    img_name_list = order_name.values()
    cmp_pair_single_expert = [(img_name_list.index(img_i_name),img_name_list.index(img_j_name) )for (img_i_name, img_j_name, cmp_label) in cmp_tuples]
    cmp_pair_array_single_expert = np.array(cmp_pair_single_expert,dtype=np.int) # 5941 by 2 array. Each row (img_i_ind, img_j_ind) are indices in img_name_list
    cmp_feat = class_feat[cmp_pair_array_single_expert[:,0],:] - class_feat[cmp_pair_array_single_expert[:,1],:]
    cmp_label_multi_expert = []
    for ind_expert in range(num_of_expert):
        cmp_label_single_expert = [cmp_label for (img_i_name, img_j_name, cmp_label) in cmp_data[ind_expert]]
        cmp_label_multi_expert.append(cmp_label_single_expert[:])
    cmp_label_array_multi_expert = np.array(cmp_label_multi_expert,dtype=np.int).T #
    # 5941 by 5 array. Each row is a comparison label corresponds to cmp_pair_array_single_expert. Each column is an experts's comparison label.
    RSD_train_plus_partition = partition_mat_file['RSDTrainPlusPartition'][repeat_index,:]
    RSD_test_plus_partition = partition_mat_file['RSDTestPlusPartition'][repeat_index,:]
    cmp_train_plus_partition_object = partition_mat_file['cmpTrainPlusPartition'][repeat_index,:]
    cmp_train_plus_partition_list = [cmp_train_plus_partition_object[i][0,:] for i in range(cmp_train_plus_partition_object.shape[0])]
    cmp_test_plus_partition_object = partition_mat_file['cmpTestPlusPartition'][repeat_index,:]
    cmp_test_plus_partition_list = [cmp_test_plus_partition_object[i][0,:] for i in range(cmp_test_plus_partition_object.shape[0])]
    # check_all_comparisons_in_the_same_set(RSD_train_plus_partition[4,:], cmp_train_plus_partition[4][0,:],cmp_tuples,img_name_list)
    # 1 Plus 2 Preplus, 3 Normal
    output_dict = {'class_feat':class_feat,'RSD_labels':partition_pickle_file['RSDLabels'], 'class_labels':partition_pickle_file['label13'],
                   'cmp_feat':cmp_feat,'cmp_label_array_multi_expert':cmp_label_array_multi_expert,'cmp_pair_array_single_expert':cmp_pair_array_single_expert,
                   'order_name':order_name,
                   'RSD_train_plus_partition':RSD_train_plus_partition,'RSD_test_plus_partition':RSD_test_plus_partition,
                   'cmp_train_plus_partition_list':cmp_train_plus_partition_list,'cmp_test_plus_partition_list':cmp_test_plus_partition_list}
    pickle.dump(output_dict,open(rop_comparison_complexity_partition_file,'wb'))
    return rop_comparison_complexity_partition_file

class ROP(object):
    def __init__(self,name_base='../data/ROP/comparison_complexity_st_subsample', lam=0, nums=80):
        partition_file = pickle.load(open(name_base+"_"+str(nums)+".p",'rb'))
        self.class_feat = partition_file['class_feat']
        self.RSD_labels = partition_file['RSD_labels']
        self.cmp_feat = partition_file['cmp_feat']
        # self.cmp_feat = self.normalize_data_matrix(self.cmp_feat)
        self.cmp_label_array_multi_expert = partition_file['cmp_label_array_multi_expert']
        self.cmp_pair_array_single_expert = partition_file['cmp_pair_array_single_expert']
        self.RSD_train_plus_partition = partition_file['RSD_train_plus_partition']
        self.RSD_test_plus_partition = partition_file['RSD_test_plus_partition']
        self.cmp_train_plus_partition_list = partition_file['cmp_train_plus_partition_list']
        self.cmp_test_plus_partition_list = partition_file['cmp_test_plus_partition_list']
        self.lam = lam#list(np.logspace(-3, 3, 15))
        self.num_of_experts = 5
        # for k in range(5):
        #     print k
        #     print check_all_comparisons_in_the_same_set(
        #         self.RSD_train_plus_partition[k], self.cmp_train_plus_partition_list[k],self.cmp_pair_array_single_expert.tolist())
        #     print check_all_comparisons_in_the_same_set(self.RSD_test_plus_partition[k],self.cmp_test_plus_partition_list[k],self.cmp_pair_array_single_expert.tolist())

        self.greedy_algorithms = ['greedyCycle', 'greedyCycleBeta', 'greedyCycleUncertain', 'greedyCycleFlipRemove',
                                  'greedyMulitCycleBeta', 'greedyMultiCycle',
                                  'sortedGreedyCycle', 'sortedGreedyCycleBeta', 'sortedGreedy']

    def normalize_data_matrix(self, data):

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        centralized_data = data - mean
        normalized_data = centralized_data / std

        return normalized_data

    def estimate_beta_by_averaging(self, feat, label):
        # This function output an estimated beta by inpued with feature and label.
        # Input:
        #       - feat (N,d) nd array. N is the number of data samples, and d is the dimensionality
        #       - label (N,1) nd array. Each element is either (1 or -1). We assign label with 0 as 1.
        # Output:
        #       - beta_est (d,1) nd array. The average of Sigma_est^{-1}*y_i*x_i.
        if len(label.shape) == 1:
            label = 1 * label[:,np.newaxis]
        else:
            label = 1 * label
        y_x_est = np.multiply(feat, label).T
        beta_est = np.mean(y_x_est, axis=1)[:, np.newaxis]
        return beta_est

    def estimate_beta_by_MLE(self, feat, labels):
        #  This function estimate beta by using logistic regression in scikit learning without penalty (setting C very large).
        # Input:
        #       - feat (N,d) nd array. N is the number of data samples, and d is the dimensionality
        #       - label (N,1) nd array. Each element is either (1 or -1). We assign label with 0 as 1.
        # Output:
        #       - beta_est (d,1) nd array.
        logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(feat, labels[:,0])
        beta_est = logistic.coef_.T
        return beta_est

    def estimate_beta_by_MAP(self, feat, labels):
        #  This function estimate beta by using logistic regression in scikit learning with penalty.
        # Input:
        #       - feat (N,d) nd array. N is the number of data samples, and d is the dimensionality
        #       - label (N,1) nd array. Each element is either (1 or -1). We assign label with 0 as 1.
        # Output:
        #       - beta_est (d,1) nd array.
        beta_lam_list = []
        if self.lam == 0:
            C = np.inf
        else:
            C = 1/self.lam
        C = 1
        logistic = LogisticRegression(C=C, fit_intercept=False, solver="newton-cg").fit(feat, labels)
        beta_est = 1*logistic.coef_.T
        # const_est = 1*logistic.intercept_
        return beta_est

    def flip_error_labels(self, noisy_label, ind_error_label):
        # This function is to flip some noisy_label with i.i.d. probability prob_correct.
        # Input:
        #       - noisy_label, (N,1) nd array. Each element is in {-1,+1}. There exists some error labels.
        #       - ind_error_label (N_error,1) nd array. Each element is the index in noisy_label and indicates the error label.
        #       - prob_correct, float in [0,1]. The probability to correct the label.
        # Output:
        #       - corrected_noisy_label, (N,1) nd array. some of error are corrected.
        if len(noisy_label.shape) == 1:
            noisy_label = 1 * noisy_label
        else:
            noisy_label = 1 * noisy_label[:,[0]]
        error_label = 1*noisy_label[ind_error_label]
        corrected_error_label = -1*error_label
        corrected_noisy_label = 1 * noisy_label
        corrected_noisy_label[ind_error_label] = 1 * corrected_error_label
        return corrected_noisy_label

    def remove_error_labels(self, feats, noisy_label, ind_error_label):
        # This function is to remove the error labels, instead of flipping them.
        # Input:
        #       - noisy_label, (N,1) nd array. Each element is in {-1,+1}. There exists some error labels.
        #       - ind_error_label (N_error,1) nd array. Each element is the index in noisy_label and indicates the error label.
        #       - prob_correct, float in [0,1]. The probability to correct the label.
        # Output:
        #       - feats_after_removed: (N_after,) nd array.
        #       - noisy_label_after_removed, (N_after,1) nd array. some of error are corrected.
        # feats = 1.*self.pair_feat
        feats_after_removed = np.delete(feats, ind_error_label, axis=0)
        noisy_label_after_removed = np.delete(noisy_label, ind_error_label, axis=0)
        return feats_after_removed, noisy_label_after_removed

    def FAS_algorithm(self, cmp_feat_train ,cmp_label_train,cmp_pair_train,algorithm, correct_method='flip',test_set=None):
        if algorithm in ['greedyFAS','kwikSort']:
            G = construct_graph(cmp_pair_train, cmp_label_train)
            ind_error_labels = self.find_error_pair_indices_by_graph(G, algorithm,cmp_label_train)
        if algorithm=='average':
            ind_error_labels = average(cmp_feat_train, cmp_label_train, noiseless_label=None)
        elif algorithm=='Repeated MLE':
            ind_error_labels = MLE(cmp_feat_train, cmp_label_train)
        elif algorithm == 'greedyCycle':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_break_cycles(cmp_label_train,cmp_pair_train,cmp_feat_train,correct_method=correct_method)
        elif algorithm == 'greedyCycleBeta':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_break_cycles_update_beta(cmp_label_train, cmp_pair_train,
                                                                                    cmp_feat_train,
                                                                                    correct_method=correct_method)
        elif algorithm == 'sortedGreedyCycle':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = sorted_greedy_cycle(
                cmp_label_train, cmp_pair_train, cmp_feat_train,
                correct_method=correct_method,
                test_set=test_set)
        elif algorithm == 'sortedGreedy':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = sorted_greedy(cmp_label_train,
                cmp_pair_train, cmp_feat_train,correct_method=correct_method,test_set=test_set)
        elif algorithm == 'greedyCycleUncertain':
            ind_error_labels, cycles_size, removed_likelihood = greedy_break_cycles_uncertain(cmp_label_train,cmp_pair_train,cmp_feat_train,correct_method=correct_method)
        elif algorithm == 'greedyCycleFlipRemove':
            ind_error_labels, cycles_size, removed_likelihood = greedy_break_cycles_flip_remove(cmp_label_train,cmp_pair_train,cmp_feat_train,correct_method=correct_method)
        elif algorithm =='greedyRandomCycle':
            ind_error_labels = greedy_random_cycles(cmp_label_train,cmp_pair_train)
        else:
            sys.exit("The algorithm is one of {'greedyFAS','ILSR'}. Receive " + str(algorithm) + " now.")

        # Check the ind_error_labels
        feat = 1*cmp_feat_train
        if correct_method == 'flip':
            label = self.flip_error_labels(cmp_label_train, ind_error_labels)
        elif correct_method == 'remove':
            feat, label = self.remove_error_labels(cmp_feat_train,cmp_label_train, ind_error_labels)
        elif correct_method == 'no':
            label = cmp_label_train
        else:
            sys.exit("The correct method is one of {'flip', 'remove', 'No'}. Receive " + str(correct_method) + " now.")
        if algorithm in self.greedy_algorithms:
            return feat, label, ind_error_labels, cycles_size, removed_likelihood, beta_err_est
        else:
            return feat, label, ind_error_labels

    def train(self, feat, label, estimate_method):
        # This function is to estimate beta with different methods.
        # Input:
        #       - feat (N,d) ndarray.
        #       - label (N,1) ndarray, each element is +1 or -1.
        #       - estimate_method, choose from {'average', 'MLE'}. 'Average' is to average y_i*xi. 'MLE' is to use
        # maximum likelihood to estimate beta.
        if estimate_method == "average":
            beta_est = self.estimate_beta_by_averaging(feat, label)
        elif estimate_method == "MLE":
            beta_est = self.estimate_beta_by_MAP(feat, label)
        else:
            sys.exit("The estimate_method is one of {'average', 'MLE'}." +"Now received "+estimate_method)
        return beta_est


    def test(self,k, beta_est):
        class_test_indices = self.RSD_test_plus_partition[k]
        class_feat_test = self.class_feat[class_test_indices,:]
        RSD_labels_test = 1*self.RSD_labels[class_test_indices]
        RSD_plus_labels_test = RSD_labels_test==1
        RSD_plus_labels_test.astype(np.int)
        RSD_normal_labels_test = RSD_labels_test!=3
        RSD_normal_labels_test.astype(np.int)
        score_class = class_feat_test.dot(beta_est)
        auc_class_plus = roc_auc_score(RSD_plus_labels_test,score_class)
        acc_class_plus = binary_accuracy(score_class,RSD_plus_labels_test)
        auc_class_normal = roc_auc_score(RSD_normal_labels_test, score_class)
        acc_class_normal = binary_accuracy(score_class,RSD_normal_labels_test)
        cmp_test_indices = self.cmp_test_plus_partition_list[k]
        cmp_feat_test, cmp_label_test, _ = self.obtain_cmp_feat_label(cmp_test_indices)
        score_cmp = cmp_feat_test.dot(beta_est)
        auc_cmp = roc_auc_score(cmp_label_test,score_cmp)
        cmp_feat_sin_test, cmp_maj_label = self.obtain_majority_comparison(cmp_test_indices)
        cmp_score_sin_test = cmp_feat_sin_test.dot(beta_est)
        auc_maj_cmp = roc_auc_score(cmp_maj_label,cmp_score_sin_test)
        acc_cmp = binary_accuracy(score_cmp,cmp_label_test)
        acc_maj_cmp = binary_accuracy(cmp_score_sin_test,cmp_maj_label)
        return auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp

    def obtain_majority_comparison(self,indices,):
        cmp_feat_single = self.cmp_feat[indices, :]
        cmp_label_array_multi = self.cmp_label_array_multi_expert[indices, :]
        cmp_label_majority,_ = mode(cmp_label_array_multi,axis=1)
        return cmp_feat_single, cmp_label_majority


    def obtain_cmp_feat_label(self,indices):
        cmp_feat_single = self.cmp_feat[indices,:]
        cmp_label_array_multi = self.cmp_label_array_multi_expert[indices,:]
        cmp_pair_single = self.cmp_pair_array_single_expert[indices,:]
        cmp_feat_ready_to_use = np.tile(cmp_feat_single,[self.num_of_experts,1])
        cmp_label_ready_to_use = cmp_label_array_multi.flatten('F')
        cmp_pair_ready_to_use = np.tile(cmp_pair_single,[self.num_of_experts,1])
        return cmp_feat_ready_to_use, cmp_label_ready_to_use, cmp_pair_ready_to_use


    def train_test(self,k, algorithm, estimate_method, correct_method='remove'):
        cmp_train_indices = self.cmp_train_plus_partition_list[k]
        cmp_feat_train, cmp_label_train, cmp_pair_train = self.obtain_cmp_feat_label(cmp_train_indices)
        cmp_test_indices = self.cmp_test_plus_partition_list[k]
        cmp_feat_test, cmp_label_test, cmp_pair_test = self.obtain_cmp_feat_label(cmp_test_indices)
        test_set = [cmp_feat_test, cmp_label_test]
        # # cmp_train_indices = self.cmp_train_plus_partition_list[k]
        # #####################
        # cmp_train_indices = np.arange(1000,5941)
        # cmp_feat_train_temp, cmp_label_train_temp, cmp_pair_train_temp = self.obtain_cmp_feat_label(cmp_train_indices)
        # # np.random.seed(10)
        # # # random_indices = np.random.choice(np.arange(cmp_feat_train_temp.shape[0]).astype(np.int),2,replace=False)
        # random_indices = np.arange(1000)
        # cmp_feat_train = cmp_feat_train_temp[random_indices,:]
        # cmp_label_train = cmp_label_train_temp[random_indices]
        #
        # logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(cmp_feat_train, cmp_label_train)
        # beta_est = 1 * logistic.coef_.T
        #
        # #####################
        #
        # # cmp_test_indices = self.cmp_test_plus_partition_list[k]
        # cmp_test_indices = np.arange(0,1000)
        # cmp_feat_test, cmp_label_test, cmp_pair_test = self.obtain_cmp_feat_label(cmp_test_indices)
        # test_set = [cmp_feat_test, cmp_label_test]
        #
        # train_items = np.unique(cmp_pair_train_temp)
        # test_items = np.unique(cmp_pair_test)
        # score_train_cmp = cmp_feat_train.dot(beta_est)
        # auc_train_cmp = roc_auc_score(cmp_label_train, score_train_cmp)
        # score_cmp = cmp_feat_test.dot(beta_est)
        # auc_cmp = roc_auc_score(cmp_label_test, score_cmp)
        # print auc_train_cmp, auc_cmp

        if algorithm == 'No Correction':
            # Run no correction
            beta_est  = self.train(cmp_feat_train, cmp_label_train, estimate_method)
            cmp_test_indices = self.cmp_test_plus_partition_list[k]
            #######################
            score_train_cmp = cmp_feat_train.dot(beta_est)
            auc_train_cmp = roc_auc_score(cmp_label_train,score_train_cmp)
            score_cmp = cmp_feat_test.dot(beta_est)
            auc_cmp = roc_auc_score(cmp_label_test, score_cmp)
            ##########################
            auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp = self.test(k, beta_est)
            return auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp
        elif algorithm in ["greedyFAS", "kwikSort"]:
            feat, label_alg, ind_error_labels = self.FAS_algorithm(cmp_feat_train, cmp_label_train, cmp_pair_train, algorithm,
                                                                   correct_method=correct_method)
            beta_est  = self.train(feat, label_alg, estimate_method)
            auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp = self.test(k, beta_est)
            return auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp, ind_error_labels
        elif algorithm in self.greedy_algorithms:
            feat, label_alg, ind_error_labels, cycles_size, removed_likelihood, beta_est_iter = self.FAS_algorithm(cmp_feat_train, cmp_label_train, cmp_pair_train,
                                                                                                    algorithm,
                                                                                                    correct_method=correct_method,
                                                                                                    test_set=test_set)
            beta_est = self.train(feat, label_alg, estimate_method)
            auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp = self.test(k, beta_est)
            return auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp, ind_error_labels, cycles_size, removed_likelihood, beta_est_iter
        # elif algorithm == 'greedyCycleBeta':
        #     feat, label_alg, ind_error_labels, cycles_size, removed_likelihood = self.FAS_algorithm(noisy_label, algorithm, correct_method=correct_method)
        #     num_true_error, num_false_error = true_false_errors(ind_error_labels,ground_truth_ind_error_labels)
        #     beta_est_alg = self.estimate_beta(feat, label_alg, estimate_method)
        #     error_alg = self.beta_estimate_error(beta_est_alg,  self.beta)
        #     beta_est_list.append(beta_est_alg * 1.)
        #     error_list.append(error_alg * 1.)
        #     return beta_est_list,error_list, num_true_error,num_false_error, cycles_size, removed_likelihood
        else:
            feat, label_alg, ind_error_labels = self.FAS_algorithm(cmp_feat_train, cmp_label_train, cmp_pair_train,
                                                                   algorithm,
                                                                   correct_method=correct_method)
            beta_est = self.train(feat, label_alg, estimate_method)
            auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp = self.test(k, beta_est)
            return auc_class_plus, acc_class_plus, auc_class_normal, acc_class_normal, auc_cmp,acc_cmp,auc_maj_cmp, acc_maj_cmp, ind_error_labels,
            # sys.exit("The algorithm is not recognized. Receive " + str(algorithm) + " now.")
            # algorithm seeds will never be used.


if __name__=='__main__':
    # rop_data = ROP_data()
    # comparison_complexity_partition_file = ROP_subsample_partition(20)
    # comparison_complexity_partition_file = ROP_subsample_partition(40)
    # comparison_complexity_partition_file = ROP_subsample_partition(60)
    # comparison_complexity_partition_file = ROP_subsample_partition(80)
    a = ROP_subsample_standard_partition(40)
    a = ROP_subsample_standard_partition(80)
    a = ROP_subsample_standard_partition(120)
    a = ROP_subsample_standard_partition(160)
    a = ROP_subsample_standard_partition(200)
    a = ROP_subsample_standard_partition(300)
    a = ROP_subsample_standard_partition(500)
    print("done")
