import numpy as np
from networkx.classes.multidigraph import MultiDiGraph
from networkx.algorithms.dag import is_directed_acyclic_graph
from networkx import find_cycle, simple_cycles
import random
import scipy.sparse.linalg as spsl
from time import time
import numpy as np
import statsmodels.base.model
from scipy.misc import logsumexp
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score
from networkx.algorithms.shortest_paths.generic import shortest_path
from sklearn.metrics import roc_auc_score
import sys

def binary_accuracy(scores,label):
    predict_label = np.sign(scores)
    acc = accuracy_score(label,predict_label)
    return acc

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

def greedyFAS(G,seed):
    """"
    greedyFAS returns a list, where the first item beats other items.
    """
    random.seed(seed)
    s_1 = []
    s_2 = []
    # Initiall remove the node with maximum delta(node)=outdegree(node)-indegredd(node)
    while(G.number_of_nodes()!=0):
        out_degree_nodes = dict(G.out_degree(G.nodes())) # Dict
        has_sink_nodes = 0 in out_degree_nodes.values()
        while (has_sink_nodes):
            sink_nodes = [node for node, out_degreee in out_degree_nodes.iteritems() if out_degreee == 0]
            sink_node_chosen = random.choice(sink_nodes)
            s_2 = [1*sink_node_chosen]+s_2
            G.remove_node(sink_node_chosen)
            out_degree_nodes =dict(G.out_degree(G.nodes()))  # Dict
            has_sink_nodes = 0 in out_degree_nodes.values()
        in_degree_nodes = dict(G.in_degree(G.nodes()))
        has_source_nodes = 0 in in_degree_nodes.values() # Check exist source nodes
        while (has_source_nodes):
            source_nodes = [node for node, in_degreee in in_degree_nodes.iteritems() if in_degreee == 0]
            source_node_chosen = random.choice(source_nodes)
            s_1.append(1*source_node_chosen)
            G.remove_node(source_node_chosen)
            in_degree_nodes = dict(G.in_degree(G.nodes()))
            has_source_nodes = 0 in in_degree_nodes.values()
        if G.number_of_nodes()!=0:
            nodes = G.nodes()
            in_degree_nodes = dict(G.in_degree(nodes))
            out_degree_nodes = dict(G.out_degree(nodes))
            delta = {node: out_degree_nodes[node] - in_degree_nodes[node] for node in in_degree_nodes.keys()}
            max_val_delta = max(delta.values())
            max_delta_nodes = [node for node, delta_value in delta.iteritems() if delta_value == max_val_delta]
            max_delta_node_chosen = random.choice(max_delta_nodes)
            s_1.append(1*max_delta_node_chosen)
            G.remove_node(max_delta_node_chosen)
    s = s_1+s_2   # Here the first element loses most comparions. The last element should rank 1.
    rank_best_worst = s[::-1]
    return rank_best_worst


def statdist(generator, method="power", v_init=None, n_iter= 500, rtol=1e-4):
    """Compute the stationary distribution of a Markov chain, described by its infinitesimal generator matrix.
    Computing the stationary distribution can be done with one of the following methods:
    - `kernel`: directly computes the left null space (co-kernel) the generator
      matrix using its LU-decomposition. Alternatively: ns = spl.null_space(generator.T)
    - `eigenval`: finds the leading left eigenvector of an equivalent
      discrete-time MC using `scipy.sparse.linalg.eigs`.
    - `power`: finds the leading left eigenvector of an equivalent
      discrete-time MC using power iterations. v_init is the initial eigenvector.
    """
    n = generator.shape[0]
    if method == "eigenval":
        '''
        Arnoldi iteration has cubic convergence rate, but does not guarantee positive eigenvector
        '''
        if v_init is None:
            v_init = np.random.rand(n,)
        # mat = generator+eye is row stochastic, i.e. rows add up to 1.
        # Multiply by eps to make sure each entry is at least -1. Sum by 1 to make sure that each row sum is exactly 1.
        eps = 1.0 / np.max(np.abs(generator))
        mat = np.eye(n) + eps*generator
        A = mat.T
        # Find the leading left eigenvector, corresponding to eigenvalue 1
        _, vecs = spsl.eigs(A, k=1, v0=v_init)
        res = np.real(vecs[:,0])
        return (1.0 / res.sum()) * res
    if method == "power":
        '''
        Power iteration has linear convergence rate and slow for lambda2~lambda1. 
        But guarantees positive eigenvector, if started accordingly.
        '''
        if v_init is None:
            v = np.random.rand(n,)
        else:
            v = v_init
        # mat = generator+eye is row stochastic, i.e. rows add up to 1.
        # Multiply by eps to make sure each entry is at least -1. Sum by 1 to make sure that each row sum is exactly 1.
        eps = 1.0 / np.max(np.abs(generator))
        mat = np.eye(n) + eps * generator
        A = mat.T
        # Find the leading left eigenvector, corresponding to eigenvalue 1
        normAest = np.sqrt(np.linalg.norm(A, ord=1) * np.linalg.norm(A, ord=np.inf))
        v = 1.*v/np.linalg.norm(v)
        Av = np.dot(A,v)
        for ind_iter in range(n_iter):
            v = 1.*Av/np.linalg.norm(Av)
            Av = np.dot(A,v)
            lamda = np.dot(v.T, Av)
            r = Av-v*lamda
            normr = np.linalg.norm(r)
            if normr < rtol*normAest:
                #print('Power iteration converged in ' + str(ind_iter) + ' iterations.')
                break
        res = np.real(v)
        return (1.0 / res.sum()) * res
    else:
        raise RuntimeError("not (yet?) implemented")



def ILSR(n, rankings, weights=None, rtol=1e-4):
    """Iterative Luce spectral ranking algorithm for partial ranking data. Remove the inner loop for top-1 ranking.
    For each ranking, there are (k-1) independent observations with winner=i and losers = {i+1,...,k}
    n:: int , number of items.
    rankings: m by c matrix. m is the number of comparisons and c is the number of items in each comparison event.
    """
    epsilon = np.finfo(float).eps
    if weights is None:
        weights = 1.0 * np.ones(n, dtype=float) / n
    ilsr_conv = False
    while not ilsr_conv:
        chain = np.zeros((n, n), dtype=float)
        for ranking in rankings:
            sum_weights = sum(weights[x] for x in ranking) + epsilon
            for i, winner in enumerate(ranking):
                val = 1.0 / sum_weights
                for loser in ranking[i + 1:]:
                    chain[loser, winner] += val
                sum_weights -= weights[winner]
        # each row sums up to 0
        chain -= np.diag(chain.sum(axis=1))
        weights_prev = weights
        weights = statdist(chain, v_init=weights)
        # Check convergence
        ilsr_conv = np.linalg.norm(weights_prev - weights) < rtol * np.linalg.norm(weights)
    return weights


def kwik_sort_fas(G, A, lo, hi):

   """
    Need to reverse the order of A. Then the first one would be rank 1.
   :param G: the graph of interest
   :param A: list, list of all vertices in the graph. Vertices should be integers.
   :param lo: integer, lower index in A to be sorted
   :param hi: integer, higher index in A to be sorted
   """
   if lo < hi:
       swapped = False
       lt, i, gt = lo, lo, hi
       p = np.random.randint(lo, hi + 1)  # + 1 since p is in [lo, hi]
       while i <= gt:
           if (i, p) in G.edges():
               temp = A[i]
               A[i] = A[lt]
               A[lt] = temp
               lt += 1
               i += 1
               swapped = True

           elif (p, i) in G.edges():
               temp = A[i]
               A[i] = A[gt]
               A[gt] = temp
               gt -= 1
               swapped = True

           else:
               i += 1

       kwik_sort_fas(G, A, lo, lt - 1)

       if swapped:
           kwik_sort_fas(G, A, lt, gt)

       kwik_sort_fas(G, A, gt + 1, hi)




def softmax(a):
    tmp = np.exp(a - np.max(a))
    return 1.*tmp / np.sum(tmp)

class ConvexPLModel(statsmodels.base.model.LikelihoodModel):
    '''
    reparametrize by pi=e^theta
    maximize loglikelihood(theta) wrt theta
    Unconstrained
    '''
    def __init__(self, endog, exog, n, **kwargs):
        '''
        params: theta, n*1
        n: number of items
        :param endog: (dependent variable): {winner i}, M*1
        :param exog: (independent variable): {A}, M*k
        '''
        super(ConvexPLModel, self).__init__(endog, exog, **kwargs)
        self._n = n

    def loglike(self, params):
        params = np.append(params, 0.0)
        ll = 0.0
        for ids, x in zip(self.exog, self.endog):
            ll += params[x] - logsumexp(params[ids])
        return ll

    def score(self, params):
        params = np.append(params, 0.0)
        grad = np.zeros(self._n, dtype=float)
        for ids, x in zip(self.exog, self.endog):
            grad[ids] -= softmax(params[ids])
            grad[x] += 1
        return grad[:-1]

    def hessian(self, params):
        epsilon = np.finfo(float).eps
        params = np.append(params, 0.0)
        hess = np.eye((self._n), dtype=float) * epsilon
        for ids in self.exog:
            vals = softmax(params[ids])
            hess[np.ix_(ids, ids)] += np.outer(vals, vals) - np.diag(vals)
        return hess[:-1,:-1]

    def fit(self, start_params=None, maxiter=20000, **kwargs):
        '''
        :param start_params: initial theta
        :return: final theta
        '''
        if start_params is None:
            # Reasonable starting values
            start_params = np.zeros(self._n - 1, dtype=float)
        start = time()
        res = super(ConvexPLModel, self).fit(start_params=start_params, maxiter=maxiter, method='newton', **kwargs)
        end = time()
        # Add the last parameter back, and zero-mean it for good measure.
        res.params = np.append(res.params, 0)
        res.params -= res.params.mean()
        return res.params, (end-start)

def average(feat, noisy_label, max_iter=50, noiseless_label=None):
    if len(noisy_label.shape) == 1:
        noisy_label = 1*noisy_label[:,np.newaxis]
    else:
        noisy_label = 1*noisy_label
    y_x_est = np.multiply(feat, noisy_label).T
    beta_est = np.mean(y_x_est, axis=1)[:, np.newaxis]
    label_consistent = np.sign(feat.dot(beta_est))
    from synthetic import synthetic_independent_comparisons
    s = synthetic_independent_comparisons(2,1,1,[(0,1)])
    beta_noiseless = s.estimate_beta(feat,noiseless_label,'average')
    beta_error_list = []
    for iter in range(max_iter):
        num_diff_oracle = (label_consistent != noiseless_label).sum()
        y_x_est_new = np.multiply(feat, label_consistent).T
        beta_est_new = np.mean(y_x_est_new, axis=1)[:, np.newaxis]
        error = s.beta_estimate_error(beta_noiseless,beta_est_new)
        print error, num_diff_oracle, noiseless_label.shape
        beta_error_list.append(error)
        label_consistent_new = np.sign(feat.dot(beta_est_new))
        label_consistent = 1* label_consistent_new

    ind_error_label = np.where((label_consistent - noisy_label) != 0)[0]
    return ind_error_label

def MLE(feat,noisy_label):
    #       - beta_est (d,1) nd array.
    if len(noisy_label.shape) == 1:
        label = 1*noisy_label[:,np.newaxis]
    else:
        label = 1*noisy_label
    noisy_label_MLE = 1*label[:,0]
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(feat,noisy_label_MLE[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    noisy_label_MLE = np.sign(feat.dot(beta_est))
    ind_error_label = np.where((noisy_label_MLE - label) != 0)[0]
    return ind_error_label

def MAP(feat,noisy_label,C):
    #       - beta_est (d,1) nd array.
    noisy_label_MLE = 1*noisy_label[:,0]
    logistic = LogisticRegression(C=C, fit_intercept=False, solver="newton-cg").fit(feat,noisy_label_MLE)
    beta_est = logistic.coef_.T
    # const = logistic.intercept_
    noisy_label_MLE = np.sign(feat.dot(beta_est))
    ind_error_label = np.where((noisy_label_MLE - noisy_label) != 0)[0]
    return ind_error_label


def get_order_pair_array(pair_array, label):
    ordered_pair_array = np.copy(pair_array)
    # In a graph, an edge always points to the winner. Swap all (i j) when y_ij = +1
    ind_swap = np.where(label[:,0] == 1)
    i_swap, j_swap = ordered_pair_array[ind_swap, 0], ordered_pair_array[ind_swap, 1]
    ordered_pair_array[ind_swap, 0], ordered_pair_array[ind_swap, 1] = j_swap, i_swap
    return ordered_pair_array

def construct_graph(pair_array, noisy_label):
    """

    :param noisy_label: +1 means in (i,j) pair, i beats j. i should have a higher rank.
    :return: graph G
    """
    ordered_pair_array = get_order_pair_array(pair_array,noisy_label)
    G = MultiDiGraph()
    G.add_edges_from(ordered_pair_array)
    return G


def find_index_all(l, key):
    return [i for i, d in enumerate(l) if np.all(d==key)]

def logistic_likelihood(feat,label,beta):
    return 1./(1.+np.exp(-np.multiply(feat.dot(beta),label)))

def greedy_break_cycles(noisy_label,pair_array,pair_feat,correct_method = 'remove',true_beta = None):
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:,np.newaxis]
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(pair_feat, noisy_label[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:,np.newaxis]
        label = 1*noisy_label
    else:
        label = 1*noisy_label
    G = construct_graph(pair_array, label)
    ind_kept_edges_in_remove = np.ones((len(ordered_pair_list),),dtype=bool)
    count = 0
    count_cycles = 0
    cycles_size = []
    removed_likelihood = []
    beta_est_iter = []
    while ((not is_directed_acyclic_graph(G))):
        t0 = time()
        count_cycles += 1
        print "cycles: "+str(count_cycles)+"/"+str(noisy_label.shape[0])
        cycles = find_cycle(G,orientation='original')
        edges = [edge[0:2] for edge in cycles]
        cycles_size.append(len(cycles))
        ind_pair_list = [find_index_all(ordered_pair_list,pair) for pair in edges]
        t1 = time()
        ind_pair = np.array([element for sub_list in ind_pair_list for element in sub_list])
        likelihood = logistic_likelihood(pair_feat[ind_pair,:],label[ind_pair],beta_est)
        ind_likelihood_min = np.argmin(likelihood)
        removed_likelihood.append(1*likelihood[ind_likelihood_min])
        #  feed_back_pair may show up multiple times.
        feed_back_edge = ordered_pair_list[ind_pair[ind_likelihood_min]]
        ind_all_feed_back_edges = np.array(find_index_all(ordered_pair_list,feed_back_edge))
        t2 = time()
        if correct_method == 'flip':
            label[ind_all_feed_back_edges] = -1*label[ind_all_feed_back_edges]
            ordered_pair_list = list(get_order_pair_array(pair_array, label))
            G = construct_graph(pair_array, label)
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(
                pair_feat, label[:,0])
            beta_est_test = logistic.coef_.T
        elif correct_method == 'remove':
            ind_kept_edges_in_remove[ind_all_feed_back_edges] = False
            # G = construct_graph(pair_array[ind_kept_edges_in_remove,:], label[ind_kept_edges_in_remove,[0]])
            G.remove_edges_from([ordered_pair_list[ind] for ind in  list(ind_all_feed_back_edges)])
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(
                pair_feat[ind_kept_edges_in_remove, :], noisy_label[ind_kept_edges_in_remove,0])
            # const_est = logistic.intercept_
            beta_est_test = logistic.coef_.T

        count += 1 # To avoid infinite loops.
        if true_beta is not None:
            beta_est_iter.append(beta_estimate_error(beta_est_test, true_beta))
        else:
            beta_est_iter.append(beta_est_test * 1)
        t_end = time()
        print "done"
    if correct_method == 'flip':
        ind_error_label = np.where((label - noisy_label) != 0)[0]
    elif correct_method == 'remove':
        ind_error_label = np.where(ind_kept_edges_in_remove==False)[0]
    return ind_error_label, cycles_size, removed_likelihood, beta_est_iter

def find_multi_cycles(G,num_cycles_find):
    """
    Given a graph, find multiple different cycles
    :param G:
    :param num_cycles_find:
    :return:
    """
    nodes = list(G)
    np.random.seed(1)
    source_nodes = list(np.random.choice(nodes,num_cycles_find))
    max_len_cycle = 0
    all_cycles = []
    largest_cycle = None
    for i in range(num_cycles_find):
        np.random.seed(i)
        np.random.shuffle(source_nodes)
        try:
            cycle = find_cycle(G,source_nodes, orientation='original')
        except:
            continue
        all_cycles.append(cycle[:])
        len_cycle = len(cycle)
        if len_cycle>max_len_cycle:
            max_len_cycle=len_cycle
            largest_cycle = cycle[:]
            # [len(c) for c in all_cycles]
    if largest_cycle is None:
        largest_cycle = find_cycle(G,orientation='original')
    return largest_cycle

def greedy_multi_cycles(noisy_label,pair_array,pair_feat,correct_method = 'remove',true_beta=None, num_cycles_find = 10,test_set=None):
    """
    Find multiple cycles and choose the longest cycle for correction.
    :param noisy_label:
    :param pair_array:
    :param pair_feat:
    :param correct_method:
    :return:
    """
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(pair_feat, noisy_label[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:,np.newaxis]
        label = 1*noisy_label
    else:
        label = 1*noisy_label
    G = construct_graph(pair_array, label)
    ind_kept_edges_in_remove = np.ones((len(ordered_pair_list),),dtype=bool)
    count = 0
    count_cycles = 0
    cycles_size = []
    removed_likelihood = []
    beta_est_iter = []
    while ((not is_directed_acyclic_graph(G))):
        t0 = time()
        count_cycles += 1
        print "cycles: "+str(count_cycles)+"/"+str(noisy_label.shape[0])
        # Find multiple cycles
        cycles = find_multi_cycles(G,num_cycles_find)
        # cycles = find_cycle(G,orientation='original')
        edges = [edge[0:2] for edge in cycles]
        cycles_size.append(len(cycles))
        ind_pair_list = [find_index_all(ordered_pair_list,pair) for pair in edges]
        t1 = time()
        ind_pair = np.array([element for sub_list in ind_pair_list for element in sub_list])
        likelihood = logistic_likelihood(pair_feat[ind_pair,:],label[ind_pair],beta_est)
        ind_likelihood_min = np.argmin(likelihood)
        removed_likelihood.append(1*likelihood[ind_likelihood_min])
        #  feed_back_pair may show up multiple times.
        feed_back_edge = ordered_pair_list[ind_pair[ind_likelihood_min]]
        ind_all_feed_back_edges = np.array(find_index_all(ordered_pair_list,feed_back_edge))
        t2 = time()
        if correct_method == 'flip':
            label[ind_all_feed_back_edges] = -1*label[ind_all_feed_back_edges]
            ordered_pair_list = list(get_order_pair_array(pair_array, label))
            G = construct_graph(pair_array, label)
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(
                pair_feat, label[:,0])
            beta_est_test = logistic.coef_.T
        elif correct_method == 'remove':
            ind_kept_edges_in_remove[ind_all_feed_back_edges] = False
            G.remove_edges_from([ordered_pair_list[ind] for ind in  list(ind_all_feed_back_edges)])
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(
                pair_feat[ind_kept_edges_in_remove, :], noisy_label[ind_kept_edges_in_remove,0])
            beta_est_test = logistic.coef_.T
        if true_beta is not None:
            if test_set is not None:
                test_feat, test_label = test_set[0],test_set[1]
                test_auc = auc_avoid_unique_label(test_feat, test_label,beta_est_test)
                beta_est_iter.append((beta_estimate_error(beta_est_test, true_beta),test_auc))
            else:
                beta_est_iter.append(beta_estimate_error(beta_est_test,true_beta))
        else:
            beta_est_iter.append(beta_est_test*1)
        count += 1
        t_end = time()
        print "done"
    if correct_method == 'flip':
        ind_error_label = np.where((label - noisy_label) != 0)[0]
    elif correct_method == 'remove':
        ind_error_label = np.where(ind_kept_edges_in_remove==False)[0]
    return ind_error_label, cycles_size, removed_likelihood, beta_est_iter


def change_single_label(test_feat,test_label):
     if np.unique(test_label).shape[0]!=2:
         n_test = test_feat.shape[0]
         np.random.seed(1)
         select_num = int(n_test/2)
         if not select_num>0:
             select_num = 1
         ran_ind = np.random.choice(n_test,select_num,replace=False)
         test_feat[ran_ind,:] = test_feat[ran_ind,:]*-1
         test_label[ran_ind] = test_label[ran_ind]*-1
     return test_feat, test_label

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

def save_one_iter(beta_est_test, beta_est_iter,true_beta,test_set):
    if true_beta is not None:
        if test_set is not None:
            test_feat, test_label = test_set[0], test_set[1]
            test_feat, test_label = change_single_label(test_feat, test_label)
            test_auc = auc_avoid_unique_label(test_feat, test_label,beta_est_test)
            if len(test_set) == 3:
                test_noiseless_label = test_set[2]
                test_noiseless_auc = auc_avoid_unique_label(test_feat,test_noiseless_label,beta_est_test)
                beta_est_iter.append((beta_estimate_error(beta_est_test, true_beta), test_auc, test_noiseless_auc))
            else:
                beta_est_iter.append((beta_estimate_error(beta_est_test, true_beta), test_auc))
            return beta_est_iter, test_auc
        else:
            beta_est_iter.append(beta_estimate_error(beta_est_test, true_beta))
            return beta_est_iter
    elif test_set is not None:
        test_feat, test_label = test_set[0], test_set[1]
        test_feat, test_label = change_single_label(test_feat, test_label)
        test_auc = auc_avoid_unique_label(test_feat, test_label, beta_est_test)
        if len(test_set) == 3:
            test_noiseless_label = test_set[2]
            test_noiseless_auc = auc_avoid_unique_label(test_feat, test_noiseless_label, beta_est_test)
            beta_est_iter.append((test_auc, test_noiseless_auc))
        else:
            beta_est_iter.append(test_auc)
        return beta_est_iter, test_auc
    else:
        beta_est_iter.append(beta_est_test * 1)
        return beta_est_iter

def sorted_greedy_cycle(noisy_label,pair_array,pair_feat,correct_method = 'remove',true_beta=None, num_cycles_find = 10, test_set= None,save_all_iters=False,solver='saga'):
    """
    Find multiple cycles and choose the longest cycle for correction.
    :param noisy_label:
    :param pair_array:
    :param pair_feat:
    :param correct_method:
    :return:
    """
    if len(noisy_label.shape)==1:
        noisy_label = noisy_label[:,np.newaxis]
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(pair_feat, noisy_label[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:,np.newaxis]
        label = 1*noisy_label
    else:
        label = 1*noisy_label
    G = construct_graph(pair_array, label)
    ind_kept_edges_in_remove = np.ones((len(ordered_pair_list),),dtype=bool)
    likelihoods = logistic_likelihood(pair_feat,label,beta_est)
    pair_ind_arg_min = np.argsort(likelihoods,axis=0)
    count = 0
    count_cycles = 0
    cycles_size = []
    beta_est_iter = []
    beta_est_iter, best_test_auc = save_one_iter(beta_est, beta_est_iter,true_beta,test_set)
    removed_likelihood = [np.array(0)]
    ind_error_label = np.array([])
    for i in list(pair_ind_arg_min[:,0]):
        if likelihoods[i] >=0.5 or is_directed_acyclic_graph(G):
            break
        pair_unlikely = pair_array[i,:].tolist()
        if label[i] == 1:
            pair_unlikely = [pair_unlikely[1],pair_unlikely[0]] # Change the direction to source sink.(There is an edge between pair_unlikely[0] to pair_unlikely[1])
        # Find if this edge is in a cycle.
        try:
            path = shortest_path(G,source=pair_unlikely[1],target=pair_unlikely[0])
            cycles_size.append(len(path))
            removed_likelihood.append(likelihoods[i]*1)
            cycle_found = True
        except:
            cycle_found = False
        if cycle_found:
            t0 = time()
            count_cycles += 1
            print "cycles: "+str(count_cycles)+"/"+str(noisy_label.shape[0])
            #  feed_back_pair may show up multiple times.
            feed_back_edge = ordered_pair_list[i]
            t2 = time()
            if correct_method == 'flip':
                ind_all_feed_back_edges = np.array(find_index_all(ordered_pair_list, feed_back_edge))
                label[ind_all_feed_back_edges] = -1*label[ind_all_feed_back_edges]
                ordered_pair_list = list(get_order_pair_array(pair_array, label))
                G = construct_graph(pair_array, label)
                if save_all_iters:
                    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(
                    pair_feat, label[:,0])
                    beta_est_test = logistic.coef_.T
            elif correct_method == 'remove':
                ind_kept_edges_in_remove[i] = False
                G.remove_edges_from([feed_back_edge])
                # G.remove_edges_from([ordered_pair_list[ind] for ind in  list(ind_all_feed_back_edges)])
                if save_all_iters:
                    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(
                    pair_feat[ind_kept_edges_in_remove], noisy_label[ind_kept_edges_in_remove,0])
                    beta_est_test = logistic.coef_.T
            if save_all_iters:
                beta_est_iter, test_auc = save_one_iter(beta_est_test, beta_est_iter, true_beta, test_set)
                if test_auc > best_test_auc:
                    best_test_auc = 1 * test_auc
                    if correct_method == 'flip':
                        ind_error_label = np.where((label - noisy_label) != 0)[0]
                    elif correct_method == 'remove':
                        ind_error_label = np.where(ind_kept_edges_in_remove == False)[0]
                    else:
                        sys.exit("correct_method should be in {flip, remove}. Now received " + correct_method)
            t_end = time()
    if not save_all_iters:
        if correct_method == 'flip':
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(
                pair_feat, label[:, 0])
            beta_est_test = logistic.coef_.T
        elif correct_method == 'remove':
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(
                pair_feat[ind_kept_edges_in_remove], noisy_label[ind_kept_edges_in_remove, 0])
            beta_est_test = logistic.coef_.T
        beta_est_iter, test_auc = save_one_iter(beta_est_test, beta_est_iter, true_beta, test_set)
    return ind_error_label, cycles_size, removed_likelihood, beta_est_iter



def sorted_greedy_cycle_update_beta(noisy_label,pair_array,pair_feat,correct_method = 'remove',true_beta=None, num_cycles_find = 10,solver='saga'):
    """
    Find multiple cycles and choose the longest cycle for correction.
    :param noisy_label:
    :param pair_array:
    :param pair_feat:
    :param correct_method:
    :return:
    """
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(pair_feat, noisy_label[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:,np.newaxis]
        label = 1*noisy_label
    else:
        label = 1*noisy_label
    G = construct_graph(pair_array, label)
    ind_kept_edges_in_remove = np.ones((len(ordered_pair_list),),dtype=bool)
    likelihoods = logistic_likelihood(pair_feat,label,beta_est)
    pair_ind_arg_min = np.argsort(likelihoods,axis=0)
    count = 0
    count_cycles = 0
    cycles_size = []
    removed_likelihood = []
    beta_est_iter = []
    i = 0
    while(likelihoods[pair_ind_arg_min[i]]<0.5 and count_cycles <= 0.5*noisy_label.shape[0] and not is_directed_acyclic_graph(G)):
        # if likelihoods[i] >=0.5:
        #     break
        pair_unlikely = pair_array[ind_kept_edges_in_remove,:][i,:].tolist()
        if label[i] == 1:
            pair_unlikely = [pair_unlikely[1],pair_unlikely[0]] # Change the direction to source sink.(There is an edge between pair_unlikely[0] to pair_unlikely[1])
        # Find if this edge is in a cycle.
        try:
            path = shortest_path(G,source=pair_unlikely[1],target=pair_unlikely[0])
            cycles_size.append(len(path))
            removed_likelihood = likelihoods[i]*1
            cycle_found = True
        except:
            cycle_found = False
        if cycle_found:
            t0 = time()
            count_cycles += 1
            print "cycles: "+str(count_cycles)+"/"+str(noisy_label.shape[0])
            #  feed_back_pair may show up multiple times.
            feed_back_edge = ordered_pair_list[i]
            ind_all_feed_back_edges = np.array(find_index_all(ordered_pair_list,feed_back_edge))
            t2 = time()
            if correct_method == 'flip':
                label[ind_all_feed_back_edges] = -1*label[ind_all_feed_back_edges]
                ordered_pair_list = list(get_order_pair_array(pair_array, label))
                G = construct_graph(pair_array, label)
                logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(
                    pair_feat, label[:,0])
                beta_est = logistic.coef_.T
                likelihoods = logistic_likelihood(pair_feat, label, beta_est)

            elif correct_method == 'remove':
                ind_kept_edges_in_remove[ind_all_feed_back_edges] = False
                G.remove_edges_from([ordered_pair_list[ind] for ind in  list(ind_all_feed_back_edges)])
                logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(
                    pair_feat[ind_kept_edges_in_remove, :], noisy_label[ind_kept_edges_in_remove,0])
                beta_est = logistic.coef_.T
                likelihoods = logistic_likelihood(pair_feat[ind_kept_edges_in_remove, :], label[ind_kept_edges_in_remove,:], beta_est)
                pair_ind_arg_min = np.argsort(likelihoods, axis=0)
                i = -1
            if true_beta is not None:
                beta_est_iter.append(beta_estimate_error(beta_est,true_beta))
            else:
                beta_est_iter.append(beta_est*1)
            count += 1
            i+= 1
            t_end = time()
    if correct_method == 'flip':
        ind_error_label = np.where((label - noisy_label) != 0)[0]
    elif correct_method == 'remove':
        ind_error_label = np.where(ind_kept_edges_in_remove==False)[0]
    return ind_error_label, cycles_size, removed_likelihood, beta_est_iter


def sorted_greedy(noisy_label,pair_array,pair_feat,correct_method = 'remove',true_beta=None, num_cycles_find = 10, test_set= None,save_all_iters=False,solver='saga'):
    """
    Find multiple cycles and choose the longest cycle for correction.
    :param noisy_label:
    :param pair_array:
    :param pair_feat:
    :param correct_method:
    :return:
    """
    ind_error_label = np.array([])
    if len(noisy_label.shape)==1:
        noisy_label = noisy_label[:,np.newaxis]
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(pair_feat, noisy_label[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:,np.newaxis]
        label = 1*noisy_label
    else:
        label = 1*noisy_label
    ind_kept_edges_in_remove = np.ones((len(ordered_pair_list),),dtype=bool)
    likelihoods = logistic_likelihood(pair_feat,label,beta_est)
    pair_ind_arg_min = np.argsort(likelihoods,axis=0)
    count = 0
    count_edges = 0
    best_test_auc = 0
    cycles_size = []
    removed_likelihood = [np.array(0)]
    beta_est_iter = []
    beta_est_iter, best_test_auc = save_one_iter(beta_est, beta_est_iter, true_beta, test_set)
    for i in list(pair_ind_arg_min[:,0]):
        if likelihoods[i] >=0.5:
            break
        removed_likelihood.append(likelihoods[i])
        pair_unlikely = pair_array[i,:].tolist()
        if label[i] == 1:
            pair_unlikely = [pair_unlikely[1],pair_unlikely[0]] # Change the direction to source sink.(There is an edge between pair_unlikely[0] to pair_unlikely[1])
        # Find if this edge is in a cycle.
        t0 = time()
        count_edges += 1
        print "edges: "+str(count_edges)+"/"+str(noisy_label.shape[0])
        #  feed_back_pair may show up multiple times.
        feed_back_edge = ordered_pair_list[i]
        t2 = time()
        if correct_method == 'flip':
            ind_all_feed_back_edges = np.array(find_index_all(ordered_pair_list, feed_back_edge))
            label[ind_all_feed_back_edges] = -1*label[ind_all_feed_back_edges]
            ordered_pair_list = list(get_order_pair_array(pair_array, label))
            if save_all_iters:
                logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(
                pair_feat, label[:,0])
                beta_est_test = logistic.coef_.T
        elif correct_method == 'remove':
            ind_kept_edges_in_remove[i] = False
            if save_all_iters:
                logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(
                pair_feat[ind_kept_edges_in_remove], noisy_label[ind_kept_edges_in_remove,0])
                beta_est_test = logistic.coef_.T
        if save_all_iters:
            beta_est_iter, test_auc = save_one_iter(beta_est_test, beta_est_iter, true_beta, test_set)
            if test_auc > best_test_auc:
                best_test_auc = 1*test_auc
                if correct_method == 'flip':
                    ind_error_label = np.where((label - noisy_label) != 0)[0]
                elif correct_method == 'remove':
                    ind_error_label = np.where(ind_kept_edges_in_remove == False)[0]
                else:
                    sys.exit("correct_method should be in {flip, remove}. Now received " + correct_method)
        count += 1
    if not save_all_iters:
        if correct_method == 'flip':
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(
                pair_feat, label[:, 0])
            beta_est_test = logistic.coef_.T
        elif correct_method == 'remove':
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver=solver,max_iter=1000).fit(
                pair_feat[ind_kept_edges_in_remove], noisy_label[ind_kept_edges_in_remove, 0])
            beta_est_test = logistic.coef_.T
        beta_est_iter, test_auc = save_one_iter(beta_est_test, beta_est_iter, true_beta, test_set)
    return ind_error_label, cycles_size, removed_likelihood, beta_est_iter


def greedy_break_cycles_uncertain(noisy_label,pair_array,pair_feat,correct_method = 'remove'):
    """
    Remove the edge closest to 0.5.
    :param noisy_label:
    :param pair_array:
    :param pair_feat:
    :param num_total_errors:
    :param correct_method:
    :return:
    """
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(pair_feat, noisy_label[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:, np.newaxis]
        label = 1 * noisy_label
    else:
        label = 1*noisy_label
    G = construct_graph(pair_array, label)
    ind_kept_edges_in_remove = np.ones((len(ordered_pair_list),),dtype=bool)
    count = 0
    count_cycles = 0
    cycles_size = []
    removed_likelihood = []
    while ((not is_directed_acyclic_graph(G))):
        t0 = time()
        count_cycles += 1
        print "cycles: "+str(count_cycles)+"/"+str(noisy_label.shape[0])
        cycles = find_cycle(G,orientation='original')
        edges = [edge[0:2] for edge in cycles]
        cycles_size.append(len(cycles))
        ind_pair_list = [find_index_all(ordered_pair_list,pair) for pair in edges]
        t1 = time()
        ind_pair = np.array([element for sub_list in ind_pair_list for element in sub_list])
        likelihood = logistic_likelihood(pair_feat[ind_pair,:],label[ind_pair],beta_est)
        ind_likelihood_min = np.argmin(np.abs(likelihood-0.5))
        removed_likelihood.append(1*likelihood[ind_likelihood_min])
        #  feed_back_pair may show up multiple times.
        feed_back_edge = ordered_pair_list[ind_pair[ind_likelihood_min]]
        ind_all_feed_back_edges = np.array(find_index_all(ordered_pair_list,feed_back_edge))
        t2 = time()
        if correct_method == 'flip':
            label[ind_all_feed_back_edges] = -1*label[ind_all_feed_back_edges]
            ordered_pair_list = list(get_order_pair_array(pair_array, label))
            G = construct_graph(pair_array, label)
        elif correct_method == 'remove':
            ind_kept_edges_in_remove[ind_all_feed_back_edges] = False
            # G = construct_graph(pair_array[ind_kept_edges_in_remove,:], label[ind_kept_edges_in_remove,[0]])
            G.remove_edges_from([ordered_pair_list[ind] for ind in  list(ind_all_feed_back_edges)])
        count += 1 # To avoid infinite loops.
        t_end = time()
        print "done"
    if correct_method == 'flip':
        ind_error_label = np.where((label - noisy_label) != 0)[0]
    elif correct_method == 'remove':
        ind_error_label = np.where(ind_kept_edges_in_remove==False)[0]
    return ind_error_label, cycles_size, removed_likelihood


def greedy_break_cycles_flip_remove(noisy_label,pair_array,pair_feat,correct_method = 'remove'):
    """
    Flip the least likely one (smallest likelihood) and remove the most uncertain one (closest to 0.5).
    :param noisy_label:
    :param pair_array:
    :param pair_feat:
    :param num_total_errors:
    :param correct_method:
    :return:
    """
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(pair_feat, noisy_label[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:, np.newaxis]
        label = 1 * noisy_label
    else:
        label = 1*noisy_label
    G = construct_graph(pair_array, label)
    ind_kept_edges_in_remove = np.ones((len(ordered_pair_list),),dtype=bool)
    count = 0
    count_cycles = 0
    cycles_size = []
    removed_likelihood = []
    flipped_likelihood = []
    while ((not is_directed_acyclic_graph(G))):
        t0 = time()
        count_cycles += 1
        print "cycles: "+str(count_cycles)+"/"+str(noisy_label.shape[0])
        cycles = find_cycle(G,orientation='original')
        edges = [edge[0:2] for edge in cycles]
        cycles_size.append(len(cycles))
        ind_pair_list = [find_index_all(ordered_pair_list,pair) for pair in edges]
        t1 = time()
        ind_pair = np.array([element for sub_list in ind_pair_list for element in sub_list])
        likelihood = logistic_likelihood(pair_feat[ind_pair,:],label[ind_pair],beta_est)
        ind_removed_likelihood_min = np.argmin(np.abs(likelihood-0.5))
        removed_likelihood.append(1*likelihood[ind_removed_likelihood_min])
        ind_flip_likelihood = np.argmin(likelihood)
        flipped_likelihood.append(1*likelihood[ind_flip_likelihood])
        #  feed_back_pair may show up multiple times.
        feed_back_edge_remove = ordered_pair_list[ind_pair[ind_removed_likelihood_min]]
        ind_all_feed_back_edges_remove = np.array(find_index_all(ordered_pair_list,feed_back_edge_remove))
        feedback_edge_flip = ordered_pair_list[ind_pair[ind_flip_likelihood]]
        ind_all_feed_back_edges_flip = np.array(find_index_all(ordered_pair_list,feedback_edge_flip))
        t2 = time()
        label[ind_all_feed_back_edges_flip] = -1*label[ind_all_feed_back_edges_flip]
        ordered_pair_list = list(get_order_pair_array(pair_array, label))
        ind_kept_edges_in_remove[ind_all_feed_back_edges_remove] = False
        G = construct_graph(pair_array[ind_kept_edges_in_remove,:], label[ind_kept_edges_in_remove,[0]])
        count += 1 # To avoid infinite loops.
        t_end = time()
        print "done"
    # if correct_method == 'flip':
    #     ind_error_label = np.where((label - noisy_label) != 0)[0]
    # el]if correct_method == 'remove':
    ind_error_label = np.concatenate((np.where(ind_kept_edges_in_remove==False)[0],np.where((label - noisy_label) != 0)[0]))
    # ind_error_label = np.where((ind_kept_edges_in_remove==False) | (label - noisy_label) != 0 )[0]
    return ind_error_label, cycles_size, [removed_likelihood,flipped_likelihood]


def greedy_break_cycles_update_beta(noisy_label,pair_array,pair_feat,true_beta=None,correct_method = 'remove'):
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(pair_feat, noisy_label[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:, np.newaxis]
        label = 1 * noisy_label
    else:
        label = 1*noisy_label
    G = construct_graph(pair_array, label)
    ind_kept_edges_in_remove = np.ones((len(ordered_pair_list),),dtype=bool)
    count = 0
    count_cycles = 0
    cycles_size = []
    removed_likelihood = []
    beta_est_iter = []
    while ((not is_directed_acyclic_graph(G))):
        t0 = time()
        count_cycles += 1
        print "cycles: "+str(count_cycles)+"/"+str(noisy_label.shape[0])
        cycles = find_cycle(G,orientation='original')
        edges = [edge[0:2] for edge in cycles]
        cycles_size.append(len(cycles))
        ind_pair_list = [find_index_all(ordered_pair_list,pair) for pair in edges]
        t1 = time()
        ind_pair = np.array([element for sub_list in ind_pair_list for element in sub_list])
        likelihood = logistic_likelihood(pair_feat[ind_pair,:],label[ind_pair],beta_est)
        ind_likelihood_min = np.argmin(likelihood)
        removed_likelihood.append(1*likelihood[ind_likelihood_min])
        #  feed_back_pair may show up multiple times.
        feed_back_edge = ordered_pair_list[ind_pair[ind_likelihood_min]]
        ind_all_feed_back_edges = np.array(find_index_all(ordered_pair_list,feed_back_edge))
        t2 = time()
        if correct_method == 'flip':
            label[ind_all_feed_back_edges] = -1*label[ind_all_feed_back_edges]
            ordered_pair_list = list(get_order_pair_array(pair_array, label))
            G = construct_graph(pair_array, label)
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(
                pair_feat, label[:,0])
            beta_est = logistic.coef_.T
            # const_est = logistic.intercept_
        elif correct_method == 'remove':
            ind_kept_edges_in_remove[ind_all_feed_back_edges] = False
            # G = construct_graph(pair_array[ind_kept_edges_in_remove,:], label[ind_kept_edges_in_remove,[0]])
            G.remove_edges_from([ordered_pair_list[ind] for ind in  list(ind_all_feed_back_edges)])
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(pair_feat[ind_kept_edges_in_remove,:], noisy_label[ind_kept_edges_in_remove,0])
            # const_est = logistic.intercept_
            beta_est = logistic.coef_.T
        if true_beta is not None:
            beta_est_iter.append(beta_estimate_error(beta_est,true_beta))
        else:
            beta_est_iter.append(beta_est*1)
        count += 1 # To avoid infinite loops.
        t_end = time()
        print "done"
    if correct_method == 'flip':
        ind_error_label = np.where((label - noisy_label) != 0)[0]
    elif correct_method == 'remove':
        ind_error_label = np.where(ind_kept_edges_in_remove==False)[0]
    # pickle.dump({'beta_error_iter': beta_error_iter}, open('error_iter_ind_6_remove_greedyCycleBeta.p', 'wb'))
    return ind_error_label, cycles_size, removed_likelihood, beta_est_iter


def greedy_multi_break_cycles_update_beta(noisy_label,pair_array,pair_feat,true_beta=None,correct_method = 'remove', num_cycles_find=5):
    logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(pair_feat, noisy_label[:,0])
    beta_est = logistic.coef_.T
    # const_est = logistic.intercept_
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    if len(noisy_label.shape) == 1:
        noisy_label = noisy_label[:, np.newaxis]
        label = 1 * noisy_label
    else:
        label = 1*noisy_label
    G = construct_graph(pair_array, label)
    ind_kept_edges_in_remove = np.ones((len(ordered_pair_list),),dtype=bool)
    count = 0
    count_cycles = 0
    cycles_size = []
    removed_likelihood = []
    beta_est_iter = []
    while ((not is_directed_acyclic_graph(G))):
        t0 = time()
        count_cycles += 1
        print "cycles: "+str(count_cycles)+"/"+str(noisy_label.shape[0])
        cycles = find_multi_cycles(G,num_cycles_find)
        # cycles = find_cycle(G,orientation='original')
        edges = [edge[0:2] for edge in cycles]
        cycles_size.append(len(cycles))
        ind_pair_list = [find_index_all(ordered_pair_list,pair) for pair in edges]
        t1 = time()
        ind_pair = np.array([element for sub_list in ind_pair_list for element in sub_list])
        likelihood = logistic_likelihood(pair_feat[ind_pair,:],label[ind_pair],beta_est)
        ind_likelihood_min = np.argmin(likelihood)
        removed_likelihood.append(1*likelihood[ind_likelihood_min])
        #  feed_back_pair may show up multiple times.
        feed_back_edge = ordered_pair_list[ind_pair[ind_likelihood_min]]
        ind_all_feed_back_edges = np.array(find_index_all(ordered_pair_list,feed_back_edge))
        t2 = time()
        if correct_method == 'flip':
            label[ind_all_feed_back_edges] = -1*label[ind_all_feed_back_edges]
            ordered_pair_list = list(get_order_pair_array(pair_array, label))
            G = construct_graph(pair_array, label)
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(
                pair_feat, label[:,0])
            beta_est = logistic.coef_.T
            # const_est = logistic.intercept_
        elif correct_method == 'remove':
            ind_kept_edges_in_remove[ind_all_feed_back_edges] = False
            # G = construct_graph(pair_array[ind_kept_edges_in_remove,:], label[ind_kept_edges_in_remove,[0]])
            G.remove_edges_from([ordered_pair_list[ind] for ind in  list(ind_all_feed_back_edges)])
            logistic = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cg").fit(pair_feat[ind_kept_edges_in_remove,:], noisy_label[ind_kept_edges_in_remove,0])
            # const_est = logistic.intercept_
            beta_est = logistic.coef_.T
        if true_beta is not None:
            beta_est_iter.append(beta_estimate_error(beta_est,true_beta))
        else:
            beta_est_iter.append(beta_est*1)
        count += 1 # To avoid infinite loops.
        t_end = time()
        print "done"
    if correct_method == 'flip':
        ind_error_label = np.where((label - noisy_label) != 0)[0]
    elif correct_method == 'remove':
        ind_error_label = np.where(ind_kept_edges_in_remove==False)[0]
    # pickle.dump({'beta_error_iter': beta_error_iter}, open('error_iter_ind_6_remove_greedyCycleBeta.p', 'wb'))
    return ind_error_label, cycles_size, removed_likelihood, beta_est_iter

def greedy_random_cycles(noisy_label,pair_array,num_total_errors):
    ordered_pair_list = list(get_order_pair_array(pair_array, noisy_label))
    label = 1*noisy_label
    G = construct_graph(pair_array, label)
    count = 0
    while ((not is_directed_acyclic_graph(G)) and count <= num_total_errors):
        cycles = find_cycle(G,orientation='original')
        edges = [edge[0:2] for edge in cycles]
        ind_pair_list = [find_index_all(ordered_pair_list,pair) for pair in edges]
        ind_pair = np.array([element for sub_list in ind_pair_list for element in sub_list])
        # Find a random edge to remove
        ind_random_edge = np.random.randint(0,ind_pair.shape[0])
        #  feed_back_pair may show up multiple times.
        feed_back_edge = ordered_pair_list[ind_pair[ind_random_edge]]
        ind_all_feed_back_edges = np.array(find_index_all(ordered_pair_list,feed_back_edge))
        label[ind_all_feed_back_edges] = -1*label[ind_all_feed_back_edges]
        ordered_pair_list = list(get_order_pair_array(pair_array, label))
        G = construct_graph(pair_array, label)
        count += 1 # To avoid infinite loops.
    ind_error_label = np.where((label - noisy_label) != 0)[0]
    return ind_error_label







# if __name__ == '__main__':
#    G = nx.MultiDiGraph()
#    edges = [[4, 6], [6, 0], [0, 1], [7, 1], [5, 7], [5, 4], [3, 6], [3, 5], [3, 4], [0, 2], [1, 2], [7, 2], [2, 3]]
#
#    G.add_edges_from(edges)
#
#    m = 8
#    linear_arrangement = range(m)
#
#    t = time()
#    kwik_sort_fas(G, linear_arrangement, 0, m - 1)
#    print time() - t, linear_arrangement


if __name__=='__main__':
    G = MultiDiGraph()
    # directed_comparison_pairs = [(0,1),(1,2),(1,3),(2,3),(7,1),(5,7),(6,5),(6,8),(8,2),(8,3),(3,4),(4,7),(4,5),(4,6)]
    directed_comparison_pairs = [(0,1), (1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)]
    G.add_edges_from(directed_comparison_pairs)
    rank_greedyFAS = greedyFAS(G,1)
    print rank_greedyFAS
    np.random.seed(1)
    item = range(1,9)
    rank_kwik = kwik_sort_fas(G,item,1,8)
    print item[::-1]

    m = 9
    winners = [pairs[0] for pairs in directed_comparison_pairs]
    newton = ConvexPLModel(winners, np.array(directed_comparison_pairs),m)
    newton_params,_ = newton.fit()
    # scores_items = softmax(newton_params)
    print newton_params


    print "done"
    # while()
