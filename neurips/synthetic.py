import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_spd_matrix
from random import sample
import random
from collections import deque
# from joblib import Parallel, delayed
from networkx.classes.multidigraph import MultiDiGraph
from itertools import combinations
from networkx.generators.random_graphs import random_regular_graph
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


def k_by_n(n):
    k = int((n-1)/2)
    return k

def m_by_n_k(n):
    k = k_by_n(n)
    m = n*k
    return m

def normalize_to_unit_length(vector):
    # This function is to normalize the vector such that the output vector has the unit length in L2 norm.
    norm_l2 = np.linalg.norm(vector)
    vector_normalized = 1. * vector / norm_l2
    return vector_normalized


def beta_estimate_error(beta_est, beta):
    # This funtion compute the l2 norm of two vectors in (d,1) shape. This function will normalize both vector to
    # unit length
    beta_est_unit_length = normalize_to_unit_length(beta_est)
    beta_unit_length = normalize_to_unit_length(beta)
    l2_error = np.linalg.norm(beta_est_unit_length - beta_unit_length)
    return l2_error

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


def auc_avoid_unique_label(test_feat,test_label,beta_est_test):
    """
    If the test labels only contain 1 or -1.
    :param test_feat:
    :param test_label:
    :param beta:
    :return:
    """
    if  np.unique(test_label).shape[0] == 1:
        test_feat_temp = 1 * test_feat
        test_label_temp = 1 * test_label
        half_num_label = test_label_temp.shape[0]/2
        test_feat_temp[:half_num_label, :] = -1 * test_feat_temp[:half_num_label, :]
        test_label_temp[:half_num_label, :] = -1 * test_label_temp[:half_num_label, :]
        test_auc = roc_auc_score(test_label_temp, test_feat_temp.dot(beta_est_test))
    else:
        test_auc = roc_auc_score(test_label, test_feat.dot(beta_est_test))
    return test_auc

class synthetic_independent_comparisons(object):
    # This class is to generate synthetic data (noise or noiseless), then correct some wrong labels. Finally it can estimate the beta by ("Maximum Likelihood")
    def __init__(self,n_train,k,d,train_pairs, seed=0):
        # To init a class, this function is to generate a beta and a Sigma.
        # Input:
        #       - d: dimensionalty of each data point.
        self.d = d
        self.train_seed = seed
        self.k = k
        # Generate a random beta and Covariance (random or identity).
        np.random.seed(1)# Fixed beta, mu and covariance
        self.beta = np.random.randn(d, 1)
        self.Sigma = np.eye(d)
        self.n_train = n_train
        # self.numOfComparisons = m*k/2
        self.train_pair_feat, self.train_pair_array, self.train_noiseless_labels = self.generate_comparison_noiseless_label(self.n_train,train_pairs,self.train_seed)

    def generate_independent_samples(self,n, seed):
        # This function is to generate N data samples with d dimension. The data samples come from the Gaussian distribution
        # (N is not necessarily entire dataset. It can be only added noiseless data)
        # with mean mu and Sigma as covariance. Note beta, d and Sigma from self in this class.
        # Input:
        #       - N, int, the number of data point generated
        #       - seed. Random seed.
        # Output:
        #       - noiseless_feat (N,d) nd array. N is the number of data samples, and d is the dimensionality
        #       - noiseless_label (N,1) nd array. Each element is either (1 or -1). We assign label with 0 as 1.
        L = np.linalg.cholesky(self.Sigma)
        # Generate noiseless synthetic data
        np.random.seed(seed)
        feat_list = []
        for i in range(n):
            x_i = L.dot(np.random.randn(self.d, 1))
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
        noisy_label[ind_error_label,0] = -1*noisy_label[ind_error_label,0]
        return noisy_label, ind_error_label

    def BradleyTerryProb(self, X, alpha=1):
        # Given x_i, compute  P(y_i=+1) = 1/(1+exp(-alpha*beta*x_i))
        # X (N,d) shape.
        # alpha: float number between [0,1]. If 1, the function equals to Bradley-Terry model.
        return 1. / (1. + np.exp(alpha * (-X.dot(self.beta))))
    #
    def generate_BradleyTerry_error(self, pair_feat, noiseless_label, noise_level,seed):
        # Generate noisy label by using BradleyTerry model.
        #   noise_level: float number between [0,1]
        N = noiseless_label.shape[0]
        prob_BT = self.BradleyTerryProb(pair_feat, alpha=noise_level)
        np.random.seed(seed)
        uniform = np.random.uniform(0.0, 1.0, size=(N, 1))
        noisy_pos_label = np.where(prob_BT - uniform >= 0)[0]
        noisy_neg_label = np.where(prob_BT - uniform < 0)[0]
        noisy_label = np.zeros((N, 1))
        noisy_label[noisy_pos_label, :] = 1
        noisy_label[noisy_neg_label, :] = -1
        ind_error_label = np.where(noisy_label - noiseless_label != 0)[0]
        return noisy_label, ind_error_label


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

    def generate_label(self, noise_level, noiseless_labels, seed):
        # This function generate feature, labels (noise or noiseless for N points.
        # - noise_level float in [0,1]. If error_method == 'iid', it is the probability to generate error labels.
        #                               If error_method == 'BT', it is the alpha in logistic function. alpha close
        #                               to zero means high noise.
        
        noisy_label, ind_error_labels = self.generate_iid_error_label(noiseless_labels, noise_level,seed)
        c = 2.*(1-2*noise_level)/np.sqrt(np.pi)/np.linalg.norm(self.beta)
        self.cbeta = c*self.beta
        

        return noisy_label,ind_error_labels

    def estimate_beta(self, feat, label):
        # This function is to estimate beta with by averaging.
        # Input:
        #       - feat (N,d) ndarray.
        #       - label (N,1) ndarray, each element is +1 or -1.
        #      
        # maximum likelihood to estimate beta.
        beta_est = self.estimate_beta_by_averaging(feat, label)
        
        return beta_est
    def test_auc(self, beta_est,feat_test,label_test):
        auc = auc_avoid_unique_label(feat_test,label_test,beta_est)
        return auc

    def estimate(self, noise_level, cv_fold_index = None):
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

        train_noisy_label, train_ground_truth_ind_error_labels = self.generate_label(noise_level,self.train_noiseless_labels,self.train_seed+1000)

       

        self.train_num_total_errors = train_ground_truth_ind_error_labels.shape[0]
        # Run algorithm
        beta_est_list = []
        error_list = []
        num_true_error, num_false_error = None, None
        
        # Run no correction
        beta_est_no_correction = self.estimate_beta(self.train_pair_feat, train_noisy_label)
        error_no_correction = np.linalg.norm(beta_est_no_correction-self.cbeta)
        error_normalized_no_correction = self.beta_estimate_error(beta_est_no_correction,self.beta)
        beta_est_list.append(beta_est_no_correction*1)
        error_list.append(((error_no_correction*1,error_normalized_no_correction*1)))
        num_true_error, num_false_error = true_false_errors(train_ground_truth_ind_error_labels,train_ground_truth_ind_error_labels)

        
        return beta_est_list,error_list, num_true_error,num_false_error






