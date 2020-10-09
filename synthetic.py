import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from random import sample
import random
from collections import deque
# from joblib import Parallel, delayed
from minimumFeedbackArcSet import greedyFAS,ILSR,kwik_sort_fas,ConvexPLModel, average, MLE, greedy_break_cycles,\
    construct_graph, greedy_random_cycles, greedy_break_cycles_update_beta, greedy_break_cycles_uncertain, greedy_break_cycles_flip_remove,\
    greedy_multi_cycles, greedy_multi_break_cycles_update_beta, sorted_greedy_cycle,sorted_greedy_cycle_update_beta, sorted_greedy, auc_avoid_unique_label
from networkx.classes.multidigraph import MultiDiGraph
from itertools import combinations
from networkx.generators.random_graphs import random_regular_graph
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score



def generate_comparison_ij(m, k, n_cpu=-1):
    """
    :param m:
    :param k:
    :param n_cpu: -1 for all cpus and 1 for 1 cpu
    :return:
    """
    comps = Parallel(n_jobs=n_cpu, backend='multiprocessing')(map(delayed(create_comparison_pairs), [m] * k))
    # Below goes from [[[1, 3], [2, 4]], [[1, 4], [2, 3]]] to [[1, 3], [2, 4], [1, 4], [2, 3]]
    return [ij_pair for k_list in comps for ij_pair in k_list]

def generate_tournament(m):
    comps = list(combinations(range(m),2))
    return comps

def generate_comparison_fixed_degree(m, degree, seed):
    """
    This function generates comparisons by a graph with fix degree each node.
    :param m:  number of items
    :return: comps: list of tuples. Each tuple contains an (i,j) pair, where, i j are the indices of items.
    """
    # Degree times m must be even
    graph = random_regular_graph(degree,m,seed)
    comps = graph.edges
    return tuple(comps)

def create_comparison_pairs(m):
    """
    :param m: number of data samples
    :return: mapping i, j
    """

    possible_j = set(range(m))

    comparisons = deque()

    for i in range(m-1):
        # Sample from all j's besides i's
        sampled_j = sample(possible_j - {i}, 1)[0]
        # Add sampled j as a pair to i
        comparisons.append([i, sampled_j])
        # Remove the sampled j from the set of possible j.
        possible_j -= {sampled_j}

    # Check the last pair are same items or not.
    if (m-1) in possible_j:
        comparisons[m-2][1] = m-1
        comparisons.append([m-1,sampled_j])
    else:

        comparisons.append([m-1, sample(possible_j,1)[0]])

    return list(comparisons)

def true_false_errors(ind_error_algorithm, ind_error_ground_truth):
    num_error_detected = ind_error_algorithm.shape[0]
    intersect = np.intersect1d(ind_error_algorithm,ind_error_ground_truth)
    num_true_error = intersect.shape[0]
    num_false_error = num_error_detected-num_true_error
    return num_true_error, num_false_error


def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True

class synthetic_independent_comparisons(object):
    # This class is to generate synthetic data (noise or noiseless), then correct some wrong labels. Finally it can estimate the beta by ("Maximum Likelihood")
    def __init__(self,n_train,n_test,k,d,train_pairs,test_pairs, seeds=(0,100)):
        # To init a class, this function is to generate a beta and a Sigma.
        # Input:
        #       - d: dimensionalty of each data point.
        self.d = d
        self.train_seed = seeds[0]
        self.test_seed = seeds[1]
        self.k = k
        # Generate a random beta and Covariance (random or identity).
        np.random.seed(1)# Fixed beta
        self.beta = np.random.randn(d, 1)
        self.Sigma = np.eye(d)
        self.n_train = n_train
        self.n_test = n_test
        self.k_test = n_test-1
        # self.numOfComparisons = m*k/2
        self.train_pair_feat, self.train_pair_array, self.train_noiseless_labels = self.generate_comparison_noiseless_label(self.n_train,train_pairs,self.train_seed)
        self.test_pair_feat, self.test_pair_array, self.test_noiseless_labels = self.generate_comparison_noiseless_label(self.n_test,test_pairs,self.test_seed)
        self.greedy_algorithms = ['greedyCycle','greedyCycleBeta','greedyCycleUncertain','greedyCycleFlipRemove','greedyMulitCycleBeta','greedyMultiCycle',
                                  'sortedGreedyCycle','sortedGreedyCycleBeta','sortedGreedy']

    def generate_independent_samples(self,n, seed):
        # This function is to generate N data samples with d dimension. The data samples come from the Gaussian distribution
        # (N is not necessarily entire dataset. It can be only added noiseless data)
        # with zero mean and Sigma as covariance. Note beta, d and Sigma from self in this class.
        # Input:
        #       - N, int, the number of data point generated
        #       - seed. Random seed.
        # Output:
        #       - noiseless_feat (N,d) nd array. N is the number of data samples, and d is the dimensionality
        #       - noiseless_label (N,1) nd array. Each element is either (1 or -1). We assign label with 0 as 1.
        mu = np.zeros((self.d, 1))
        # Generate noiseless synthetic data
        np.random.seed(seed)
        feat_list = []
        for i in range(n):
            x_i = self.Sigma.dot(np.random.randn(self.d, 1)) + mu
            feat_list.append(x_i.T)
        feat = np.concatenate(feat_list, axis=0)
        return feat

    def generate_comparison_noiseless_label(self,n,pairs_origin,seed):
        feat = self.generate_independent_samples(n,seed)
        pairs = list(pairs_origin)
        random.seed(seed)
        random.shuffle(pairs)
        pair_array = np.array(pairs)
        item_is,item_js = pair_array[:,0],pair_array[:,1]
        pair_feat = feat[item_is,:]-feat[item_js,:]
        noiseless_labels = np.sign(pair_feat.dot(self.beta))
        noiseless_labels[np.where(noiseless_labels == 0)[0], :] = 1
        return pair_feat, pair_array, noiseless_labels

    def generate_iid_error_label(self, noiseless_label, prob_error,seed):
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
        np.random.seed(seed)
        ind_error_label = np.random.choice(np.arange(N).astype(np.int),num_of_errors,replace=False)
        noisy_label = 1*noiseless_label
        noisy_label[ind_error_label] = -1*noisy_label[ind_error_label]
        # error_label = np.random.choice([-1, 1], size=(N, 1), p=[prob_error, 1 - prob_error])
        # ind_error_label = np.where(error_label == -1)[0]
        # noisy_label = np.multiply(noiseless_label, error_label)
        return noisy_label, ind_error_label

    # def BradleyTerryProb(self, X, alpha=1):
    #     # Given x_i, compute  P(y_i=+1) = 1/(1+exp(-alpha*beta*x_i))
    #     # X (N,d) shape.
    #     # alpha: float number between [0,1]. If 1, the function equals to Bradley-Terry model.
    #     return 1. / (1. + np.exp(alpha * (-X.dot(self.beta))))
    #
    # def generate_BradleyTerry_error(self, pair_feat, noiseless_label, noise_level,seed):
    #     # Generate noisy label by using BradleyTerry model.
    #     #   noise_level: float number between [0,1]
    #     N = noiseless_label.shape[0]
    #     prob_BT = self.BradleyTerryProb(pair_feat, alpha=noise_level)
    #     np.random.seed(seed)
    #     uniform = np.random.uniform(0.0, 1.0, size=(N, 1))
    #     noisy_pos_label = np.where(prob_BT - uniform >= 0)[0]
    #     noisy_neg_label = np.where(prob_BT - uniform < 0)[0]
    #     noisy_label = np.zeros((N, 1))
    #     noisy_label[noisy_pos_label, :] = 1
    #     noisy_label[noisy_neg_label, :] = -1
    #     ind_error_label = np.where(noisy_label - noiseless_label != 0)[0]
    #     return noisy_label, ind_error_label


    def rank_to_error_pair_indices(self, item_rank, pair_array, noisy_label):
        """
        Given the item and its rank, we generate the indices telling which comparison is wrong.
        :param item_rank:  m by 2 matrix. m is the number of items.  First column is the item number in [1,m] and
        the second column is its rank. (1 wins most comparisons).
        noisy_label: n by 1 matrix. n is the number of comparisons. +1 -1, +1 means i beats j.
        :return:  indices in pair_arrary indicate which rows are wrong. (This may not be the ground truth, but it is consistent (no cycle)).
        """
        item_rank = item_rank[np.argsort(item_rank[:, 0]), :]
        is_rank, js_rank = item_rank[pair_array[:, 0], 1], item_rank[pair_array[:, 1], 1]
        subtract_ij_rank = is_rank - js_rank
        label_consistent = -1 * np.sign(subtract_ij_rank)
        ind_error_label = np.where((label_consistent[:, np.newaxis] - noisy_label) != 0)[0]
        return ind_error_label

    def find_error_pair_indices_by_graph(self, G, algorithm,noisy_label,seed):
        """
        Given a graph. Run an algorithm to detect error comparisons.
        :param G:  Graph G
        :return: indices in pair_arrary indicate which rows are wrong. (This may not be the ground truth, but it is consistent (no cycle)).
        """
        if algorithm=='greedyFAS':
            ordered_items = greedyFAS(G, seed)  # The first is rank 1 (win most comparisons).
        elif algorithm=='kwikSort':
            ordered_items = range(0,self.n_train)
            np.random.seed(self.train_seed)
            kwik_sort_fas(G,ordered_items,0,self.n_train-1)
            ordered_items = ordered_items[::-1]
        else:
            sys.exit("The error method must be one of {'greedyFAS','kwikSort'}. Receive " + str(algorithm) + " Now.")
        ordered_items_array = np.array(ordered_items)[:, np.newaxis]
        items_rank_number = np.arange(1,len(ordered_items)+1)[:,np.newaxis]
        item_rank = np.concatenate((ordered_items_array,items_rank_number),axis=1)
        ind_error_label = self.rank_to_error_pair_indices(item_rank,self.train_pair_array,noisy_label)
        return ind_error_label

    def find_error_pair_indices_by_spectral(self, algorithm,noisy_label):
        # Change the order accordingly, because spectral algorithms requires that the first one is the winner.
        ordered_pair_array = np.copy(self.train_pair_array)
        ind_swap = np.where(noisy_label == -1)
        i_swap, j_swap = ordered_pair_array[ind_swap, 0], ordered_pair_array[ind_swap, 1]
        ordered_pair_array[ind_swap, 0], ordered_pair_array[ind_swap, 1] = j_swap, i_swap
        if algorithm == 'ILSR':
            weights = ILSR(self.n_train,ordered_pair_array) # Higher weights means higher rank (highest has rank 1).
        elif algorithm=='newton':
            newton = ConvexPLModel(ordered_pair_array[:,0],ordered_pair_array,self.n_train)
            weigths ,_ = newton.fit()
        else:
            sys.exit("algorithm in spectral should be either 'ILSR' or 'newton'. Receive " + str(algorithm) + " Now.")
        ordered_items = np.argsort(-weights) # argsort list indices from minimum to highest. We want indices from maximum to minimum
        ordered_items_array = np.array(ordered_items)[:,np.newaxis]
        items_rank_number = np.arange(1, len(ordered_items) + 1)[:, np.newaxis]
        item_rank = np.concatenate((ordered_items_array, items_rank_number), axis=1)
        ind_error_label = self.rank_to_error_pair_indices(item_rank, self.train_pair_array,noisy_label)
        return ind_error_label

    def flip_error_labels(self, noisy_label, ind_error_label):
        # This function is to flip some noisy_label with i.i.d. probability prob_correct.
        # Input:
        #       - noisy_label, (N,1) nd array. Each element is in {-1,+1}. There exists some error labels.
        #       - ind_error_label (N_error,1) nd array. Each element is the index in noisy_label and indicates the error label.
        #       - prob_correct, float in [0,1]. The probability to correct the label.
        # Output:
        #       - corrected_noisy_label, (N,1) nd array. some of error are corrected.
        error_label = noisy_label[ind_error_label, [0]]
        corrected_error_label = -1*error_label
        corrected_noisy_label = 1 * noisy_label
        corrected_noisy_label[ind_error_label,0] = 1 * corrected_error_label
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

    def normalize_to_unit_length(self, vector):
        # This function is to normalize the vector such that the output vector has the unit length in L2 norm.
        norm_l2 = np.linalg.norm(vector)
        vector_normalized = 1. * vector / norm_l2
        return vector_normalized

    def beta_estimate_error(self, beta_est, beta):
        # This funtion compute the l2 norm of two vectors in (d,1) shape. This function will normalize both vector to
        # unit length
        beta_est_unit_length = self.normalize_to_unit_length(beta_est)
        beta_unit_length = self.normalize_to_unit_length(beta)
        l2_error = np.linalg.norm(beta_est_unit_length - beta_unit_length)
        return l2_error

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
        logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(feat, labels[:,0])
        beta_est = logistic.coef_.T
        return beta_est

    def generate_label(self, error_method, noise_level, noiseless_labels, seed):
        # This function generate feature, labels (noise or noiseless for N points.
        # - noise_level float in [0,1]. If error_method == 'iid', it is the probability to generate error labels.
        #                               If error_method == 'BT', it is the alpha in logistic function. alpha close
        #                               to zero means high noise.
        if error_method == 'iid':
            noisy_label, ind_error_labels = self.generate_iid_error_label(noiseless_labels, noise_level,seed)
        # elif error_method == 'BT':
        #     noisy_label, ind_error_labels = self.generate_BradleyTerry_error(self.pair_feat, self.noiseless_labels, noise_level)
        # elif error_method == 'no':
        #     # Noiseless setting
        #     noisy_label, ind_error_labels = self.noiseless_labels, np.array([])
        else:
            sys.exit("The error method must be one of {'iid'}. Receive " + str(error_method) + " Now.")

        return noisy_label,ind_error_labels

    def FAS_algorithm(self,train_noisy_label,test_noisy_label, algorithm, correct_method='flip'):
        # if train_test_split is None:
        #     train_index = np.arange(noisy_label.shape[0])
        #     test_index = np.arange(noisy_label.shape[0])
        # else:
        #     train_index, test_index = train_test_split[0], train_test_split[1]
        test_set = [self.test_pair_feat, test_noisy_label, self.test_noiseless_labels]
        if algorithm in ['greedyFAS','kwikSort']:
            G = construct_graph(self.train_pair_array,train_noisy_label)
            ind_error_labels = self.find_error_pair_indices_by_graph(G, algorithm,train_noisy_label,self.train_seed)
        elif algorithm=='ILSR':
            ind_error_labels = self.find_error_pair_indices_by_spectral(algorithm,train_noisy_label)
        elif algorithm=='average':
            ind_error_labels = average(self.train_pair_feat, train_noisy_label, noiseless_label=self.train_noiseless_labels)
        elif algorithm=='Repeated MLE':
            ind_error_labels = MLE(self.train_pair_feat, train_noisy_label)
        elif algorithm == 'greedyCycle':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_break_cycles(train_noisy_label,self.train_pair_array,self.train_pair_feat,true_beta=1*self.beta,correct_method=correct_method)
        elif algorithm == 'greedyMultiCycle':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_multi_cycles(train_noisy_label,self.train_pair_array,self.train_pair_feat,correct_method=correct_method,true_beta=self.beta,test_set=test_set)
        elif algorithm == 'greedyCycleBeta':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_break_cycles_update_beta(train_noisy_label, self.train_pair_array,self.train_pair_feat,true_beta=1*self.beta,correct_method=correct_method)
        elif algorithm =='greedyMultiCycleBeta':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = greedy_multi_break_cycles_update_beta(train_noisy_label,self.train_pair_array,self.train_pair_feat,true_beta=self.beta*1,correct_method=correct_method)
        elif algorithm == 'greedyCycleUncertain':
            ind_error_labels, cycles_size, removed_likelihood = greedy_break_cycles_uncertain(train_noisy_label,self.train_pair_array,self.train_pair_feat,correct_method=correct_method)
        elif algorithm=='sortedGreedyCycle':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = sorted_greedy_cycle(train_noisy_label,self.train_pair_array,self.train_pair_feat,correct_method=correct_method,true_beta=self.beta,test_set=test_set)
        elif algorithm=='sortedGreedy':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = sorted_greedy(train_noisy_label,self.train_pair_array,self.train_pair_feat,correct_method=correct_method,true_beta=self.beta,test_set=test_set)
        elif algorithm == 'sortedGreedyCycleBeta':
            ind_error_labels, cycles_size, removed_likelihood, beta_err_est = sorted_greedy_cycle_update_beta(train_noisy_label,self.train_pair_array,self.train_pair_feat,correct_method=correct_method,true_beta=self.beta)
        elif algorithm == 'greedyCycleFlipRemove':
            ind_error_labels, cycles_size, removed_likelihood = greedy_break_cycles_flip_remove(train_noisy_label,self.train_pair_array,self.train_pair_feat,correct_method=correct_method)
        elif algorithm =='greedyRandomCycle':
            ind_error_labels = greedy_random_cycles(train_noisy_label,self.train_pair_array,self.train_num_total_errors)
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
        feat = 1*self.train_pair_feat
        if correct_method == 'flip':
            if ind_error_labels.shape[0]==0:
                label = train_noisy_label
            else:
                label = self.flip_error_labels(train_noisy_label, ind_error_labels)
        elif correct_method == 'remove':
            feat, label = self.remove_error_labels(self.train_pair_feat, train_noisy_label, ind_error_labels)
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
        else:
            sys.exit("The estimate_method is one of {'average', 'MLE'}")
        return beta_est
    def test_auc(self, beta_est,feat_test,label_test):
        auc = auc_avoid_unique_label(feat_test,label_test,beta_est)
        return auc

    def estimate(self, error_method, algorithm, estimate_method, noise_level, correct_method, cv_fold_index = None):
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
        train_noisy_label, train_ground_truth_ind_error_labels = self.generate_label(error_method, noise_level,self.train_noiseless_labels,self.train_seed)

        test_noisy_label, test_ground_truth_ind_error_labels = self.generate_label(error_method, noise_level,
                                                                                     self.test_noiseless_labels,
                                                                                     self.test_seed)
        # Generate k_fold_cross_validation:
        # if cv_fold_index >= 0: # negative value will train and test the entire dataset.
        #     kf = KFold(n_splits=5,random_state=0, shuffle=False)
        #     # train_index, test_index = list(kf.split(np.arange(self.n)))[cv_fold_index]
        #     #### The following 3 lines are using sample split.
        #     class_train_index, class_test_index = list(kf.split(np.arange(self.n_total)))[cv_fold_index]
        #     np.random.seed(self.seed)
        #     if self.n<class_train_index.shape[0]:
        #         class_train_index_subsampled = np.random.choice(class_train_index,self.n, replace=False)
        #     else:
        #         class_train_index_subsampled = class_train_index
        #     # class_train_index_subsampled = class_train_index
        #     train_index = np.where(np.all(np.isin(self.pair_array, class_train_index_subsampled),axis=1))[0]
        #
        #     if class_test_index.shape[0]>100:
        #         class_test_index_subsampled = np.random.choice(class_test_index,100, replace=False)
        #     else:
        #         class_test_index_subsampled = class_test_index
        #     test_index = np.where(np.all(np.isin(self.pair_array, class_test_index_subsampled),axis=1))[0]
        # else:
        #     train_index = np.arange(self.pair_array.shape[0])
        #     test_index = np.arange(self.pair_array.shape[0])

        self.train_num_total_errors = train_ground_truth_ind_error_labels.shape[0]
        # Run algorithm, no correction, and oracle.
        # Run algorithm
        beta_est_list = []
        error_list = []
        num_true_error, num_false_error = None, None

        if 'Oracle' in algorithm:
            oracle_name_list = algorithm.split()
            if len(oracle_name_list)==1:
                # Run Oracel (noiseless)
                beta_oracle = self.estimate_beta(self.train_pair_feat, self.train_noiseless_labels, estimate_method)
                error_oracle = self.beta_estimate_error(beta_oracle, self.beta)
                test_auc = self.test_auc(beta_oracle,self.test_pair_feat,test_noisy_label)
                test_noiseless_auc = self.test_auc(beta_oracle,self.test_pair_feat,self.test_noiseless_labels)
                beta_est_list.append(beta_oracle * 1)
                error_list.append((error_oracle * 1,test_auc*1, test_noiseless_auc*1))
                num_true_error, num_false_error = 0, 0
            else:
                percentage_correct = float(oracle_name_list[0])
                number_corrected = int(percentage_correct * train_ground_truth_ind_error_labels.shape[0])
                np.random.seed(self.train_seed)
                ind_error_labels = np.random.choice(train_ground_truth_ind_error_labels, (number_corrected,), replace=False)
                num_true_error, num_false_error = true_false_errors(ind_error_labels, train_ground_truth_ind_error_labels)
                feat = 1 * self.train_pair_feat
                if correct_method == 'flip':
                    label_part_oracle = self.flip_error_labels(train_noisy_label, ind_error_labels)
                elif correct_method == 'remove':
                    feat, label_part_oracle = self.remove_error_labels(self.train_pair_feat,train_noisy_label, ind_error_labels)
                else:
                    sys.exit('correct method is either flip or remove, now receive '+str(correct_method))
                beta_est_part_oracle = self.estimate_beta(feat, label_part_oracle, estimate_method)
                error_part_oracle = self.beta_estimate_error(beta_est_part_oracle, self.beta)
                beta_est_list.append(beta_est_part_oracle * 1.)
                error_list.append(error_part_oracle * 1.)

        elif algorithm == 'No Correction':
            # Run no correction
            beta_est_no_correction = self.estimate_beta(self.train_pair_feat, train_noisy_label, estimate_method)
            error_no_correction = self.beta_estimate_error(beta_est_no_correction, self.beta)
            test_auc = self.test_auc(beta_est_no_correction, self.test_pair_feat, test_noisy_label)
            test_noiseless_auc = self.test_auc(beta_est_no_correction, self.test_pair_feat,
                                               self.test_noiseless_labels)
            beta_est_list.append(beta_est_no_correction*1)
            error_list.append((error_no_correction*1,test_auc*1, test_noiseless_auc*1))
            num_true_error, num_false_error = true_false_errors(train_ground_truth_ind_error_labels,train_ground_truth_ind_error_labels)

        elif algorithm in ["greedyFAS", "kwikSort"]:
            feat,label_alg, ind_error_labels = self.FAS_algorithm(train_noisy_label, algorithm, correct_method=correct_method)
            num_true_error, num_false_error = true_false_errors(ind_error_labels, train_ground_truth_ind_error_labels)
            beta_est_alg = self.estimate_beta(feat, label_alg, estimate_method)
            test_auc = self.test_auc(beta_est_alg, self.test_pair_feat, test_noisy_label)
            error_alg = self.beta_estimate_error(beta_est_alg, self.beta)
            beta_est_list.append(beta_est_alg*1.)
            error_list.append((error_alg*1.,test_auc*1))
        elif algorithm in self.greedy_algorithms:
            feat, label_alg, ind_error_labels, cycles_size, removed_likelihood, beta_est_iter = self.FAS_algorithm(train_noisy_label,test_noisy_label, algorithm)
            num_true_error, num_false_error = true_false_errors(ind_error_labels,train_ground_truth_ind_error_labels)
            beta_est_alg = self.estimate_beta(feat, label_alg, estimate_method)
            error_alg = self.beta_estimate_error(beta_est_alg,  self.beta)
            beta_est_list.append(beta_est_alg * 1.)
            error_list.append(error_alg * 1.)
            return beta_est_list,error_list, num_true_error,num_false_error, cycles_size, removed_likelihood, beta_est_iter
        elif algorithm == 'greedyEdge':
            from greedy_SC import greedy_edge
            import networkx as nx
            G = nx.DiGraph()

            for i in range(len(train_noisy_label)):
                pair = self.train_pair_array[i]
                if train_noisy_label[i] == 1:
                    G.add_edge(pair[0], pair[1])
                else:
                    G.add_edge(pair[1], pair[0])

            new_G = greedy_edge(G)
            edges = np.array(new_G.edges())
            indices = np.zeros(len(edges), dtype=np.int)

            for i, edge in enumerate(edges):
                for j, pair in enumerate(self.train_pair_array):
                    if np.array_equal(edge, pair) or \
                            np.array_equal(np.array((edge[1], edge[0])), pair):
                        indices[i] = j

            new_X = self.train_pair_feat[indices]
            new_Y = train_noisy_label[indices]

            beta_est_alg = self.estimate_beta(new_X, new_Y, estimate_method)

            error_alg = self.beta_estimate_error(beta_est_alg, self.beta)

            beta_est_list.append(beta_est_alg * 1.)
            error_list.append(error_alg * 1.)

            return beta_est_list, error_list, 0, 0

        else:
            # algorithm seeds will never be used.
            feat, label_alg,ind_error_labels = self.FAS_algorithm(train_noisy_label, algorithm, correct_method=correct_method)
            num_true_error, num_false_error = true_false_errors(ind_error_labels,train_ground_truth_ind_error_labels)
            beta_est_alg = self.estimate_beta(feat, label_alg, estimate_method)
            error_alg = self.beta_estimate_error(beta_est_alg,  self.beta)
            beta_est_list.append(beta_est_alg * 1.)
            error_list.append(error_alg * 1.)
        return beta_est_list,error_list, num_true_error,num_false_error

def k_by_n(n):
    k = int((n-1)/2)
    return k

def m_by_n_k(n):
    k = k_by_n(n)
    m = n*k
    return m

# g_after = self.construct_graph(label_alg)
# import networkx as nx
# c = nx.find_cycle(g_after)


