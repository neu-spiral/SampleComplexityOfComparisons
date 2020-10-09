
from pandas import read_csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from minimumFeedbackArcSet import greedyFAS,ILSR,kwik_sort_fas,ConvexPLModel, average, MLE, greedy_break_cycles,\
    construct_graph, greedy_random_cycles, greedy_break_cycles_update_beta, greedy_break_cycles_uncertain, greedy_break_cycles_flip_remove,\
    greedy_multi_cycles, greedy_multi_break_cycles_update_beta, sorted_greedy_cycle,sorted_greedy_cycle_update_beta, sorted_greedy
from networkx.generators.random_graphs import random_regular_graph
import sys

def true_false_errors(ind_error_algorithm, ind_error_ground_truth):
    num_error_detected = ind_error_algorithm.shape[0]
    intersect = np.intersect1d(ind_error_algorithm,ind_error_ground_truth)
    num_true_error = intersect.shape[0]
    num_false_error = num_error_detected-num_true_error
    return num_true_error, num_false_error


def admission_user_partition(num_sub_item,k,file="../data/Admission/Admission_Predict_Ver1.1.csv"):
    name_base = "../data/Admission/Admission_itemCV_k_"+str(k)+"_sub_"+str(num_sub_item)+".p"
    data_raw =  np.array(read_csv(file))
    feat_cont = data_raw[:,1:-2]
    prob_admission = data_raw[:,-1]
    feat_cate = data_raw[:,[-2]]
    scaler = StandardScaler()
    feat_cont_normalized = scaler.fit_transform(feat_cont)
    feat_abs = np.concatenate((feat_cont_normalized,feat_cate),axis=1)
    num_samples = feat_abs.shape[0]
    cmp_pairs = []
    cmp_label = []
    cmp_feat = []
    for i in range(num_samples):
        for j in range(i+1,num_samples):
            if prob_admission[i] != prob_admission[j]:
                cmp_pairs.append((i, j))
                feat_diff = feat_abs[[i], :] - feat_abs[[j], :]
                cmp_feat.append(1 * feat_diff)
                if prob_admission[i] > prob_admission[j]:
                    cmp_label.append(1)
                else:
                    cmp_label.append(-1)
    cmp_feat = np.concatenate(cmp_feat,axis=0)
    cmp_scaler = StandardScaler()
    cmp_feat_normalized = cmp_scaler.fit_transform(cmp_feat)
    cmp_label = np.array(cmp_label)

    np.random.seed(0)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cmp_train_ind =[]
    cmp_test_ind = []
    num_train_cmp_fold = []
    num_test_cmp_fold = []
    if k!= 0:
        if num_sub_item==0:
            num_sub_item = int(0.8*num_samples)
        n_train = int((num_sub_item) * ( num_sub_item- 1) / 2)
        if not str(k).isdigit():
            m_train = int((2. * n_train ** 2 + (4 * n_train ** 4 - 1. / 27) ** 0.5) ** (1. / 3) + (
                        2. * n_train ** 2 - (4 * n_train ** 4 - 1. / 27) ** 0.5) ** (1. / 3))
            k = int(np.sqrt(m_train-1))
            if k % 2 != 0:
                k = k - 1
        else:
            m_train = int(n_train / k * 2)
        while (k >= m_train):
            m_train += 1
        print("degree at "+str(k), "m="+str(m_train), "n="+str(n_train))
        graph = random_regular_graph(k, m_train, 0)
        comps = graph.edges
        for abs_train_full, abs_test in kf.split(feat_abs):
            if num_sub_item == 0:
                abs_train = abs_train_full
            else:
                abs_train = np.sort(np.random.choice(abs_train_full, m_train, replace=False))
            cmps_train_in_graph = [(abs_train[i],abs_train[j])for i,j in comps]
            train_fold = []
            test_fold = []
            pair_id = 0
            for i,j in cmp_pairs:
                if i in abs_train and j in abs_train:
                    if (i, j) in cmps_train_in_graph:
                        train_fold.append(pair_id)
                elif i in abs_test and j in abs_test:
                    test_fold.append(pair_id)
                pair_id += 1
            cmp_train_ind.append(train_fold[:])
            num_train_cmp_fold.append(len(train_fold))
            cmp_test_ind.append(test_fold[:])
            num_test_cmp_fold.append(len(test_fold))

    else:
        # Tournament

        for abs_train_full, abs_test in kf.split(feat_abs):
            if num_sub_item == 0:
                abs_train = abs_train_full
            else:
                abs_train = np.sort(np.random.choice(abs_train_full, num_sub_item, replace=False))
            train_fold = []
            test_fold = []
            pair_id = 0
            for i,j in cmp_pairs:
                if i in abs_train and j in abs_train:
                    train_fold.append(pair_id)
                elif i in abs_test and j in abs_test:
                    test_fold.append(pair_id)
                pair_id += 1
            cmp_train_ind.append(train_fold[:])
            num_train_cmp_fold.append(len(train_fold))
            cmp_test_ind.append(test_fold[:])
            num_test_cmp_fold.append(len(test_fold))
    print("sub="+str(num_sub_item),"k="+str(k),"n="+str(len(cmp_train_ind[0])))

    out_dict = {'cmp_pairs':cmp_pairs,'cmp_feat':cmp_feat,'cmp_feat_normalized':cmp_feat_normalized,'cmp_label':cmp_label,'cmp_train_ind':cmp_train_ind,'cmp_test_ind':cmp_test_ind}
    pickle.dump(out_dict,open(name_base,'wb'))
    return



def admission_standard_partition(num_sub_item,k,file="../data/Admission/Admission_Predict_Ver1.1.csv"):
    name_base = "../data/Admission/Admission_itemCV_k_" + str(k) + "_sub_" + str(num_sub_item) + ".p"
    data_raw =  np.array(read_csv(file))
    feat_cont = data_raw[:,1:-2]
    prob_admission = data_raw[:,-1]
    feat_cate = data_raw[:,[-2]]
    scaler = StandardScaler()
    feat_cont_normalized = scaler.fit_transform(feat_cont)
    feat_abs = np.concatenate((feat_cont_normalized,feat_cate),axis=1)
    num_samples = feat_abs.shape[0]
    cmp_pairs = []
    cmp_label = []
    cmp_feat = []
    for i in range(num_samples):
        for j in range(i+1,num_samples):
            if prob_admission[i] != prob_admission[j]:
                cmp_pairs.append((i,j))
                feat_diff = feat_abs[[i],:]-feat_abs[[j],:]
                cmp_feat.append(1*feat_diff)
                if prob_admission[i]>prob_admission[j]:
                    cmp_label.append(1)
                else:
                    cmp_label.append(-1)
    cmp_feat = np.concatenate(cmp_feat,axis=0)
    cmp_scaler = StandardScaler()
    cmp_feat_normalized = cmp_scaler.fit_transform(cmp_feat)
    cmp_label = np.array(cmp_label)

    np.random.seed(0)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    cmp_train_ind =[]
    cmp_test_ind = []
    num_train_cmp_fold = []
    num_test_cmp_fold = []
    for train_fold_full, test_fold in kf.split(cmp_feat):
        train_fold = np.sort(np.random.choice(train_fold_full, num_sub_item, replace=False))
        cmp_train_ind.append(train_fold[:])
        num_train_cmp_fold.append(len(train_fold))
        cmp_test_ind.append(np.array(test_fold[:]))
        num_test_cmp_fold.append(len(test_fold))

    out_dict = {'cmp_pairs':cmp_pairs,'cmp_feat':cmp_feat,'cmp_feat_normalized':cmp_feat_normalized,'cmp_label':cmp_label,'cmp_train_ind':cmp_train_ind,'cmp_test_ind':cmp_test_ind}
    pickle.dump(out_dict,open(name_base,'wb'))
    return

class Admission():
    def __init__(self,num_sub_item,k,seed=0):
        # To init a class, this function is to read features, labels and partitions.
        # Input:
        #       - d: dimensionalty of each data point.
        name_base = "../data/Admission/Admission_itemCV_k_" + str(k) + "_sub_" + str(num_sub_item) + ".p"
        data = pickle.load(open(name_base,'rb'))
        self.seed = seed
        self.cmp_feat, self.pair_array, self.noiseless_labels = data['cmp_feat'], np.array(data['cmp_pairs']), data['cmp_label']
        self.cmp_train_parition, self.cmp_test_parition = data['cmp_train_ind'], data['cmp_test_ind'] 
        self.greedy_algorithms = ['greedyCycle', 'greedyCycleBeta', 'greedyCycleUncertain', 'greedyCycleFlipRemove',
                                  'greedyMulitCycleBeta', 'greedyMultiCycle',
                                  'sortedGreedyCycle', 'sortedGreedyCycleBeta', 'sortedGreedy']

    def generate_iid_error_label(self, noiseless_label, prob_error):
        # This function is to flip the noiseless label with i.i.d. probability, prob_error.
        # (Not necessarily use entire dataset's noiseless label. It can be only added noiseless data)
        # Input:
        #       - noiseless_label (N,1) nd array. Each element is in {-1,+1}.
        #       - prob_error, float number between [0,1]. The probability to flip label.
        # Output:
        #       - noisy_label (N,1) nd array. some label are flipped.
        #       - ind_error_labels, np array, each element is the index in noisy label, indicating which is flipped.
        N = noiseless_label.shape[0]
        num_of_errors = int(prob_error*N)
        np.random.seed(self.seed)
        ind_error_label = np.random.choice(np.arange(N).astype(np.int),num_of_errors)
        noisy_label = 1*noiseless_label
        noisy_label[ind_error_label] = -1*noisy_label[ind_error_label]
        return noisy_label, ind_error_label
    
    def find_error_pair_indices_by_graph(self, G, algorithm,noisy_label):
        """
        Given a graph. Run an algorithm to detect error comparisons.
        :param G:  Graph G
        :return: indices in pair_arrary indicate which rows are wrong. (This may not be the ground truth, but it is consistent (no cycle)).
        """
        if algorithm=='greedyFAS':
            ordered_items = greedyFAS(G, self.seed)  # The first is rank 1 (win most comparisons).
        elif algorithm=='kwikSort':
            ordered_items = range(0,self.m)
            np.random.seed(self.seed)
            kwik_sort_fas(G,ordered_items,0,self.m-1)
            ordered_items = ordered_items[::-1]
        else:
            sys.exit("The error method must be one of {'greedyFAS','kwikSort'}. Receive " + str(algorithm) + " Now.")
        ordered_items_array = np.array(ordered_items)[:, np.newaxis]
        items_rank_number = np.arange(1,len(ordered_items)+1)[:,np.newaxis]
        item_rank = np.concatenate((ordered_items_array,items_rank_number),axis=1)
        ind_error_label = self.rank_to_error_pair_indices(item_rank,noisy_label)
        return ind_error_label

    def find_error_pair_indices_by_spectral(self, algorithm,noisy_label):
        # Change the order accordingly, because spectral algorithms requires that the first one is the winner.
        ordered_pair_array = np.copy(self.pair_array)
        ind_swap = np.where(noisy_label == -1)
        i_swap, j_swap = ordered_pair_array[ind_swap, 0], ordered_pair_array[ind_swap, 1]
        ordered_pair_array[ind_swap, 0], ordered_pair_array[ind_swap, 1] = j_swap, i_swap
        if algorithm == 'ILSR':
            weights = ILSR(self.m,ordered_pair_array) # Higher weights means higher rank (highest has rank 1).
        elif algorithm=='newton':
            newton = ConvexPLModel(ordered_pair_array[:,0],ordered_pair_array,self.m)
            weigths ,_ = newton.fit()
        else:
            sys.exit("algorithm in spectral should be either 'ILSR' or 'newton'. Receive " + str(algorithm) + " Now.")
        ordered_items = np.argsort(-weights) # argsort list indices from minimum to highest. We want indices from maximum to minimum
        ordered_items_array = np.array(ordered_items)[:,np.newaxis]
        items_rank_number = np.arange(1, len(ordered_items) + 1)[:, np.newaxis]
        item_rank = np.concatenate((ordered_items_array, items_rank_number), axis=1)
        ind_error_label = self.rank_to_error_pair_indices(item_rank, noisy_label)
        return ind_error_label

    def flip_error_labels(self, noisy_label, ind_error_label):
        # This function is to flip some noisy_label with i.i.d. probability prob_correct.
        # Input:
        #       - noisy_label, (N,1) nd array. Each element is in {-1,+1}. There exists some error labels.
        #       - ind_error_label (N_error,1) nd array. Each element is the index in noisy_label and indicates the error label.
        #       - prob_correct, float in [0,1]. The probability to correct the label.
        # Output:
        #       - corrected_noisy_label, (N,1) nd array. some of error are corrected.
        if ind_error_label.size == 0:
            return noisy_label
        error_label = noisy_label[ind_error_label]
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
        # feats = 1.*self.cmp_feat
        feats_after_removed = np.delete(feats, ind_error_label, axis=0)
        noisy_label_after_removed = np.delete(noisy_label, ind_error_label, axis=0)
        return feats_after_removed, noisy_label_after_removed
    
    def estimate_beta_by_averaging(self, feat, label):
        # This function output an estimated beta by inpued with feature and label.
        # Input:
        #       - feat (N,d) nd array. N is the number of data samples, and d is the dimensionality
        #       - label (N,1) nd array. Each element is either (1 or -1). We assign label with 0 as 1.
        # Output:
        #       - beta_est (d,1) nd array. The average of Sigma_est^{-1}*y_i*x_i.
        y_x_est = np.multiply(feat, label).T
        beta_est = np.mean(y_x_est, axis=1)[:, np.newaxis]
        return beta_est

    def estimate_beta_by_MaximumLikelihood(self, feat, labels):
        #  This function estimate beta by using logistic regression in scikit learning without penalty (setting C very large).
        # Input:
        #       - feat (N,d) nd array. N is the number of data samples, and d is the dimensionality
        #       - label (N,1) nd array. Each element is either (1 or -1). We assign label with 0 as 1.
        # Output:
        #       - beta_est (d,1) nd array.
        logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(feat, labels)
        beta_est = logistic.coef_.T
        return beta_est
    
    def FAS_algorithm(self,noisy_label, algorithm, correct_method='flip', train_test_split=None):
        if train_test_split is None:
            train_index = np.arange(noisy_label.shape[0])
            test_index = np.arange(noisy_label.shape[0])
        else:
            train_index, test_index = train_test_split[0], train_test_split[1]
        test_set = [self.cmp_feat[test_index, :], noisy_label[test_index], self.noiseless_labels[test_index]]
        if algorithm in ['greedyFAS','kwikSort']:
            G = construct_graph(self.pair_array[train_index],noisy_label[train_index])
            ind_error_labels = self.find_error_pair_indices_by_graph(G, algorithm,noisy_label)
        elif algorithm=='ILSR':
            ind_error_labels = self.find_error_pair_indices_by_spectral(algorithm,noisy_label[train_index])
        elif algorithm=='average':
            ind_error_labels = average(self.cmp_feat[train_index], noisy_label[train_index], noiseless_label=self.noiseless_labels)
        elif algorithm=='Repeated MLE':
            ind_error_labels = MLE(self.cmp_feat[train_index], noisy_label[train_index])
        elif algorithm == 'greedyCycle':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_break_cycles(noisy_label,self.pair_array,self.cmp_feat,\
                                                                                                  correct_method=correct_method)
        elif algorithm == 'greedyMultiCycle':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_multi_cycles(noisy_label[train_index],self.pair_array[train_index,:],self.cmp_feat[train_index,:],
                                                                                                  correct_method=correct_method,test_set=test_set)
        elif algorithm == 'greedyCycleBeta':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_break_cycles_update_beta(noisy_label, self.pair_array,
                                                                                    self.cmp_feat,
                                                                                    correct_method=correct_method)
        elif algorithm =='greedyMultiCycleBeta':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_multi_break_cycles_update_beta(noisy_label,self.pair_array,self.cmp_feat,correct_method=correct_method)
        elif algorithm == 'greedyCycleUncertain':
            ind_error_labels, cycles_size, removed_likelihood = greedy_break_cycles_uncertain(noisy_label,self.pair_array,self.cmp_feat,correct_method=correct_method)
        elif algorithm=='sortedGreedyCycle':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = sorted_greedy_cycle(noisy_label[train_index],self.pair_array[train_index,:],self.cmp_feat[train_index,:],
                                                                                                  correct_method=correct_method,
                                                                                                  test_set=test_set)
        elif algorithm=='sortedGreedy':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = sorted_greedy(noisy_label[train_index],self.pair_array[train_index,:],self.cmp_feat[train_index,:],
                                                                                                  correct_method=correct_method,
                                                                                                  test_set=test_set)
        elif algorithm == 'sortedGreedyCycleBeta':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = sorted_greedy_cycle_update_beta(noisy_label,
                                                                                                  self.pair_array,
                                                                                                  self.cmp_feat,
                                                                                                  correct_method=correct_method,)
        elif algorithm == 'greedyCycleFlipRemove':
            ind_error_labels, cycles_size, removed_likelihood = greedy_break_cycles_flip_remove(noisy_label,self.pair_array,self.cmp_feat,correct_method=correct_method)
        elif algorithm =='greedyRandomCycle':
            ind_error_labels = greedy_random_cycles(noisy_label,self.pair_array,self.num_totoal_errors)
        else:
            sys.exit("The algorithm is one of {'greedyFAS','ILSR'}. Receive " + str(algorithm) + " now.")

        # (label != self.noiseless_labels).sum()
        # (noisy_label != self.noiseless_labels).sum()
        # beta_noisy = self.estimate_beta_by_MaximumLikelihood(feat, noisy_label)
        # beta_noiseless = self.estimate_beta_by_MaximumLikelihood(feat, self.noiseless_labels)
        # beta_alg = self.estimate_beta_by_MaximumLikelihood(feat, label)
        # (label != noisy_label).sum()
        # self.beta_estimate_error(beta_noisy, beta_alg)
        # self.beta_estimate_error(beta_noisy, beta_noiseless)
        # self.beta_estimate_error(beta_alg, beta_noiseless)

        # Check the ind_error_labels
        feat = 1*self.cmp_feat[train_index,:]
        if correct_method == 'flip':
            label = self.flip_error_labels(noisy_label[train_index], ind_error_labels)
        elif correct_method == 'remove':
            feat, label = self.remove_error_labels(feat, noisy_label[train_index], ind_error_labels)
        else:
            sys.exit("The correct method is one of {'flip', 'remove'}. Receive " + str(correct_method) + " now.")
        if algorithm in self.greedy_algorithms:
            return feat, label, ind_error_labels, cycles_size, removed_likelihood,beta_err_est
        else:
            return feat, label, ind_error_labels
        
        
    def estimate_beta(self, feat, label, estimate_method):
        # This function is to estimate beta with different methods.
        # Input:
        #       - feat (N,d) ndarray.
        #       - label (N,1) ndarray, each element is +1 or -1.
        #       - estimate_method, choose from {'average', 'MLE'}. 'Average' is to average y_i*xi. 'MLE' is to use
        # maximum likelihood to estimate beta.
        if estimate_method == "average":
            beta_est = self.estimate_beta_by_averaging(feat, label)
        elif estimate_method == "MLE":
            beta_est = self.estimate_beta_by_MaximumLikelihood(feat, label)
        # else:
        #     sys.exit("The estimate_method is one of {'average', 'MLE'}")
        return beta_est
    def test_auc(self, beta_est,feat_test,label_test):
        score_test = feat_test.dot(beta_est)
        auc = roc_auc_score(label_test,score_test)
        return auc
    
    def estimate(self, algorithm, estimate_method, noise_level, correct_method, cv_fold_index = None):
        # This function is to consecutively generate each data samples and estimate beta with each number in N_list
        # The data samples come from the Gaussian distribution with zero mean and Sigma as covariance. Each sample has
        # a probability prob_error to contain error comparison labels.
        # Input:
        #       - algorithm to detect error comparisons: {'greedyFAS'}
        #       - error_method: The method to generate error comparison labels. Choose from {'iid', 'BT', 'no'},
        # corresponding to generate iid error labels, which requires prob_error. BT uses Bradlety-Terry model to generate
        # error labels. None uses noiseless label.
        #       - estimate_method, choose from {'average', 'MLE'}. 'Average' is to average y_i*xi. 'MLE' is to use
        # maximum likelihood to estimate beta.
        #       - noise_level float in [0,1]. If error_method == 'iid', it is the probability to generate error labels.
        #                                    If error_method == 'BT', it is the alpha in logistic function. alpha close
        #                                    to zero means high noise.
        #       - correct_method, choose from { 'flip', 'remove', 'no'}. After identifying error labels, we flip or remove them. Or not do anything.
        # Output:
        #       - error_alg,error_no_correction,error_oracel
        counter = 0
        noisy_label, ground_truth_ind_error_labels = self.generate_iid_error_label(self.noiseless_labels, noise_level)
        train_index = self.cmp_train_parition[cv_fold_index]
        test_index = self.cmp_test_parition[cv_fold_index]

        self.num_total_errors = ground_truth_ind_error_labels.shape[0]
        # Run algorithm, no correction, and oracle.
        # Run algorithm
        beta_est_list = []
        error_list = []
        num_true_error, num_false_error = None, None

        if algorithm == 'Oracle' :
            # Run Oracel (noiseless)
            beta_oracle = self.estimate_beta(self.cmp_feat[train_index,:], self.noiseless_labels[train_index], estimate_method)
            test_auc = self.test_auc(beta_oracle,self.cmp_feat[test_index,:],noisy_label[test_index])
            test_noiseless_auc = self.test_auc(beta_oracle,self.cmp_feat[test_index,:],self.noiseless_labels[test_index])
            beta_est_list.append(beta_oracle * 1)
            num_true_error, num_false_error = 0, 0
            return test_auc, test_noiseless_auc

        elif algorithm == 'No Correction':
            # Run no correction
            beta_est_no_correction = self.estimate_beta(self.cmp_feat[train_index,:], noisy_label[train_index], estimate_method)
            test_auc = self.test_auc(beta_est_no_correction, self.cmp_feat[test_index, :], noisy_label[test_index])
            test_noiseless_auc = self.test_auc(beta_est_no_correction, self.cmp_feat[test_index,:],
                                               self.noiseless_labels[test_index])
            beta_est_list.append(beta_est_no_correction*1)
            num_true_error, num_false_error = true_false_errors(ground_truth_ind_error_labels,ground_truth_ind_error_labels)
            return test_auc, test_noiseless_auc, num_true_error, num_false_error

        elif algorithm in ["greedyFAS", "kwikSort"]:
            feat,label_alg, ind_error_labels = self.FAS_algorithm(noisy_label[train_index,:], algorithm, correct_method=correct_method)
            num_true_error, num_false_error = true_false_errors(ind_error_labels, ground_truth_ind_error_labels[train_index])
            beta_est_alg = self.estimate_beta(feat, label_alg, estimate_method)
            test_auc = self.test_auc(beta_est_alg, self.cmp_feat[test_index, :], noisy_label[test_index, :])
            test_noiseless_auc =  self.test_auc(beta_est_alg, self.cmp_feat[test_index, :],
                                               self.noiseless_labels[test_index, :])
            beta_est_list.append(beta_est_alg*1.)
            return test_auc, test_noiseless_auc, beta_est_list

            
        elif algorithm in self.greedy_algorithms:
            feat, label_alg, ind_error_labels, cycles_size, removed_likelihood, beta_est_iter = self.FAS_algorithm(noisy_label, algorithm,
                                                                                                                   correct_method=correct_method,train_test_split=(train_index,test_index))
            num_true_error, num_false_error = true_false_errors(ind_error_labels,ground_truth_ind_error_labels)
            return num_true_error,num_false_error, cycles_size, removed_likelihood, beta_est_iter
        else:
            # algorithm seeds will never be used.
            feat, label_alg,ind_error_labels = self.FAS_algorithm(noisy_label, algorithm, correct_method=correct_method)
            num_true_error, num_false_error = true_false_errors(ind_error_labels,ground_truth_ind_error_labels)
            beta_est_alg = self.estimate_beta(feat, label_alg, estimate_method)
            beta_est_list.append(beta_est_alg * 1.)
        return beta_est_list,error_list, num_true_error,num_false_error


if __name__=='__main__':

    count = 0
    k_list = [4,'sqrt'] #0,4, 'sqrt'
    sub_list = [5,8,10,12,15,20]
    for k in k_list:
        for sub in sub_list:
            count += 1
            print "Processing "+str(count)+'/'+str(len(k_list)*len(sub_list))
            admission_user_partition(sub,k)



    # admission_standard_partition(10)
    # admission_standard_partition(50)
    # admission_standard_partition(100)
    # admission_standard_partition(500)
    # admission_standard_partition(0)

    # cmp_feat_train = cmp_feat_normalized[train_fold,:]
    # cmp_label_train = cmp_label[train_fold]
    #
    # cmp_feat_test = cmp_feat_normalized[test_fold,:]
    # cmp_label_test = cmp_label[test_fold]
    # logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(cmp_feat_train,
    #                                                                                      cmp_label_train)
    # beta_est = 1 * logistic.coef_.T
    # score_train_cmp = cmp_feat_train.dot(beta_est)
    # auc_train_cmp = roc_auc_score(cmp_label_train, score_train_cmp)
    # score_cmp = cmp_feat_test.dot(beta_est)
    # auc_cmp = roc_auc_score(cmp_label_test, score_cmp)
    # print auc_train_cmp, auc_cmp

