"""
Helper codes
"""
from itertools import combinations
from collections import deque
from random import sample
from joblib import Parallel, delayed
import numpy as np
from networkx.generators.random_graphs import \
    random_regular_graph, erdos_renyi_graph
# from networkx.algorithms.dag import is_directed_acyclic_graph
# from networkx import find_cycle, simple_cycles
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def get_f_name(args):
    pass


def get_c1(beta, f_cov):
    """
    Estimate c1 = 4E[sigmoid'(beta^T(X-Y))]
    """
    # Resulting variance from the inner product
    sigma2 = 2*beta @ f_cov @ beta
    # Points to sample derivative from
    x = np.linspace(-2*sigma2, 2*sigma2, 10**6)
    # pdf(x)
    pdf = np.exp(-x**2/(2*sigma2))/(2*np.pi*sigma2)**.5
    if np.trapz(pdf, x) < 0.999:
        print('c1 accuracy may be low.')
    # sigmoid(x)
    sig_x = (1 + np.exp(-x))**-1
    # sigmoid'(x)pdf(x)
    y = sig_x*(1-sig_x)*pdf
    e_c1 = 4*np.trapz(y, x)

    return e_c1


def get_f_stats(d, ld):
    """
    Samples a mean and a psd covariance
    where largest eigen value is 1
    and smallest eigen value is 0.
    """
    # Feature mean
    f_mean = np.random.rand(d)*10 - 5

    basis = np.random.randn(d, d)
    # Orthonormal basis as columns
    basis, _ = np.linalg.qr(basis)
    eigen_values = np.linspace(ld, 1, d)
    # Feature covariance with eigen value composition
    f_cov = basis*eigen_values @ basis.T

    return f_mean, f_cov


def get_NM(k, N1, N2):
    """
    Given input k, N1, N2, return arrays N, M
    """
    N = np.ceil(np.logspace(N1, N2, 10)).astype(np.int32)

    if k == 1:
        M = N
    elif k == 2:
        M = np.ceil(N*np.log(np.log(N))).astype(np.int32)
    elif k == 3:
        M = np.ceil(N*np.log(N)).astype(np.int32)
    elif k == 4:
        M = np.ceil(N*N**.5).astype(np.int32)

    return N, M


def auc_avoid_unique_label(test_feat, test_label, beta_est_test):
    """
    If the test labels only contain 1 or -1.
    :param test_feat:
    :param test_label:
    :param beta:
    :return:
    """
    if np.unique(test_label).shape[0] == 1:
        test_feat_temp = 1 * test_feat
        test_label_temp = 1 * test_label
        half_num_label = test_label_temp.shape[0]/2
        test_feat_temp[:half_num_label, :] = \
            -1 * test_feat_temp[:half_num_label, :]
        test_label_temp[:half_num_label, :] = \
            -1 * test_label_temp[:half_num_label, :]
        test_auc = roc_auc_score(test_label_temp,
                                 test_feat_temp.dot(beta_est_test))
    else:
        test_auc = roc_auc_score(test_label, test_feat.dot(beta_est_test))
    return test_auc


def gen_tour(N):
    """
    Genereta a pairs of a tournament
    :param N: number of elements in the tournament
    :return: all pairs [(i,j)]
    """
    comps = combinations(range(N), 2)
    return list(comps)


def gen_comp_fixed(N, k):
    """
    This function generates pairs as a graph with fixed degree at each node.
    :param N: number of items
    :param k: degree of a node
    :return: comps: list of tuples. Each tuple contains an (i,j) pair,
        where i j are the indices of items.
    """
    graph = random_regular_graph(k, N)
    comps = graph.edges
    return list(comps)


def gen_comp_p(N, p):
    """
    Generate erdos renyi graph with N nodes and edge prob p
    """
    graph = erdos_renyi_graph(N, p, directed=True)
    comps = graph.edges
    return list(comps)


def gen_data_comp(N, d, k=-1, p=0):
    """
    Generates standard normal data for comparisons
    :param N: Number of items
    :param d: dimensionality
    :param k: degree of a node, if 0 generates tournament
    """
    X = np.random.randn(N, d)  # Features

    if k == 0:
        E = np.array(gen_tour(N))
    elif k > 0:
        E = np.array(gen_comp_fixed(N, k))
    elif p:
        E = np.array(gen_comp_p(N, p))
    else:
        raise Exception('Degree k is negative, needs to be non negative.')

    X_comp = X[E[:, 0]] - X[E[:, 1]]

    V = np.unique(E)

    return X, X_comp, V, E


def normalize_data_matrix(data):
    """
    Normalize so that each feature is zero mean and std 1
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    centralized_data = data - mean
    normalized_data = centralized_data / std

    return normalized_data


def estimate_beta_by_MLE(feat, labels):
    """
    This function estimate beta by using logistic regression
    in scikit learning without penalty (setting C very large).
    Input:
          - feat (N,d) nd array. N is the number of data samples
          - label (N,1) nd array. Each element is either (1 or -1).
    Output:
           - beta_est (d,1) nd array.
    """
    logistic = LogisticRegression(C=np.inf, fit_intercept=False,
                                  solver="newton-cg").fit(feat, labels[:, 0])
    beta_est = logistic.coef_.T
    return beta_est


def k_by_n(n):
    """
    Max node degree of a graph with n vertices
    """
    k = int((n-1)/2)
    return k


def m_by_n_k(n):
    """
    Total number of edges when each vertice has maximum degree
    """
    k = k_by_n(n)
    m = n*k
    return m


def generate_comparison_ij(m, k, n_cpu=-1):
    """
    :param m:
    :param k:
    :param n_cpu: -1 for all cpus and 1 for 1 cpu
    :return:
    """
    comps = Parallel(n_jobs=n_cpu, backend='multiprocessing')
    comps = comps(map(delayed(create_comparison_pairs), [m] * k))
    # Below goes from [[[1, 3], [2, 4]], [[1, 4], [2, 3]]]
    # to [[1, 3], [2, 4], [1, 4], [2, 3]]
    return [ij_pair for k_list in comps for ij_pair in k_list]


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
    if m-1 in possible_j:
        comparisons[m-2][1] = m-1
        comparisons.append([m-1, sampled_j])
    else:
        comparisons.append([m-1, sample(possible_j, 1)[0]])

    return list(comparisons)


def normalize_to_unit_length(vector):
    """
    This function is to normalize the vector such that the output
    vector has the unit length in L2 norm.
    """
    norm_l2 = np.linalg.norm(vector)
    vector_normalized = 1. * vector / norm_l2
    return vector_normalized


def beta_estimate_error(beta_est, beta):
    """
    This funtion compute the l2 norm of two vectors in (d,1) shape.
    This function will normalize both vector to unit length
    """
    beta_est_unit_length = normalize_to_unit_length(beta_est)
    beta_unit_length = normalize_to_unit_length(beta)
    l2_error = np.linalg.norm(beta_est_unit_length - beta_unit_length)
    return l2_error


if __name__ == '__main__':
    X, X_comp, V, E = gen_data_comp(N=5, d=3, k=2)

    print(X)
    print(V)
    print(E)
