"""
Data generation and reading related
"""
from random import choices
from pathlib import Path
import pickle
import numpy as np
from helpers import cv_edges, get_unq_nodes, get_max_acc


def get_data(N, M, beta, f_mean, f_cov, alpha):
    """
    Synthetic data
    Generates d dimensional gaussian vectors.
    Uniformly at random pairwise comparisons are chosen.
    Labels are generated with Bradley Terry model.
    """
    # Data for covariance estimation
    X = np.random.multivariate_normal(f_mean, f_cov, N)

    # Data for comparisons
    X1 = np.random.multivariate_normal(f_mean, f_cov, N)

    # Uniformly at random edges
    # Choices with repetition
    vertices = list(range(N))
    u = choices(vertices, k=M)
    v = choices(vertices, k=M)

    # An edge from u to v implies v beats u
    # Comparison features
    XC = X1[v] - X1[u]

    # beta^T(X-Y)
    scores = XC @ beta
    # BTL Probability
    p = (1+np.exp(-alpha*scores))**-1
    # Uniform random variables
    urv = np.random.rand(M)
    # BTL Labels
    yn = np.sign(p - urv)
    # True labels
    y = np.sign(scores)

    if y.size > 5:
        print('Error ratio is %.3f' % (np.sum(yn != y)/y.size))

    return X, XC, yn, y


def read_sushi_data(path):
    """
    path: str, path to sushi3-2016 dir
    """
    # Features of sushis
    feats = np.zeros((100, 18))
    with open(path+'sushi3.idata') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            line = np.array([float(x) for x in line[2:]])
            feats[i][:2] = line[:2]  # Style and major group
            feats[i][2+int(line[2])] = 1  # Minor group
            feats[i][-4:] = line[-4:]  # oil, freq, price, shop freq
    # Profiles of users
    profs = np.zeros((5000, 102))
    # gender, age, time to fill, pref unt 15, pref now
    # prefact 47 foreign countries never exists, removed
    # so each pref gives 45 categorical features
    useful_is = [1, 2, 3, 4, 7]
    with open(path+'sushi3.udata') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            line = np.array([int(line[ui]) for ui in useful_is])
            profs[i][0] = line[0]
            profs[i][1+line[1]] = 1
            profs[i][7] = line[2]
            profs[i][8+line[3]] = 1
            profs[i][55+line[4]] = 1
    # Edges from a
    edges = []
    with open(path+'sushi3a.5000.10.order') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip first line
            line = line.strip().split()[2:]
            line = [int(x) for x in line]
            for j in range(9):
                for v in line[j+1:]:
                    edges.append([line[j], v])
    # Edges from b
    edges_b = []
    with open(path+'sushi3b.5000.10.order') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip first line
            line = line.strip().split()[2:]
            line = [int(x) for x in line]
            for j in range(9):
                for v in line[j+1:]:
                    edges_b.append([line[j], v])
    # Mean scores for nodes
    scores_b = [[] for _ in range(100)]
    with open(path+'sushi3b.5000.10.score') as f:
        for line in f:
            line = line.strip().split()
            for i, score in enumerate(line):
                score = int(score)
                if score != -1:
                    scores_b[i].append(score)

    scores_b = np.array([np.mean(x) for x in scores_b])

    return feats, profs, edges, edges_b, scores_b


def split_sushi_data(K):
    """
    Needs to be run once on raw sushi data
    before starting sushi experiments

    Splits edges in set b of sushi data with cross validation
    Makes sure no node is shared in train and test sets
    Saves splits and scores
    """
    print('Reading sushi data...')
    home_path = str(Path.home())
    features, _, _, edges, scores = \
        read_sushi_data(home_path + '/Data/sushi3-2016/')
    print('Splitting edges per fold...')
    splits = cv_edges(edges, K)

    for i, split in enumerate(splits):
        print('For split %i, get stats_u, train_u...' % i)
        train_e, test_e = split
        print('Train edge count before stats/train split: %i' % len(train_e))
        train_u = get_unq_nodes(train_e)
        N = len(train_u)//2
        stats_u = train_u[N:]
        train_u = train_u[:N]
        for edge in train_e:
            u, v = edge
            if u in train_u and v in train_u:
                continue
            else:
                train_e.remove(edge)
        test_u = get_unq_nodes(test_e)
        print('Train edge count after split: %i' % len(train_e))
        with open(home_path + '/Data/sushi3-2016/split%i' % i, 'wb+') as f:
            pickle.dump([stats_u, train_u, train_e, test_u, test_e], f)

    with open(home_path + '/Data/sushi3-2016/features', 'wb+') as f:
        pickle.dump(features, f)

    with open(home_path + '/Data/sushi3-2016/scores', 'wb+') as f:
        pickle.dump(scores, f)


def get_sushi_fs():
    """
    Read sushi data features and scores
    for every item in set b
    """
    home_path = str(Path.home())

    with open(home_path + '/Data/sushi3-2016/features', 'rb') as f:
        a_feats = pickle.load(f)

    with open(home_path + '/Data/sushi3-2016/scores', 'rb') as f:
        a_scrs = pickle.load(f)

    return a_feats, a_scrs


def get_sushi_data(cvk, N):
    """
    Get sushi data from splits
    """
    features, scores = get_sushi_fs()

    home_path = str(Path.home())
    with open(home_path + '/Data/sushi3-2016/split%i' % cvk, 'rb') as f:
        stats_u, train_u, train_e, test_u, test_e = pickle.load(f)

    stats_u = stats_u[:N]
    train_u = train_u[:N]

    for edge in train_e:
        u, v = edge
        if u in train_u and v in train_u:
            continue
        else:
            train_e.remove(edge)

    # Get features, comparisons, labels, scores
    # For training
    M = len(train_e)
    X = features[stats_u]
    u, v = [list(x) for x in zip(*train_e)]
    XC = features[v] - features[u]
    XC[M//2:] *= -1
    yn = np.ones(M)
    yn[M//2:] *= -1
    # For testing
    M = len(test_e)
    test_X = features[test_u]
    u, v = [list(x) for x in zip(*test_e)]
    test_XC = features[v] - features[u]
    test_XC[M//2:] *= -1
    test_yn = np.ones(M)
    test_yn[M//2:] *= -1
    test_scores = scores[test_u]

    return X, XC, test_X, test_XC, yn, test_yn, test_scores


def get_sushi_max_acc(K):
    """
    Computes the maximum accuracy any classifier
    could achieve on sushi dataset edges.
    """
    print('Reading sushi data...')
    home_path = str(Path.home())
    _, _, _, edges, _ = read_sushi_data(home_path + '/Data/sushi3-2016/')
    edges = tuple(edges)  # ([u, v], ...)

    # Compute on all data
    print('Computing max accuracy on all data...')
    max_all_acc = get_max_acc(edges)
    print('Max accuracy on all data is: %.3f.' % max_all_acc)

    # Compute the average on cross validation test sets
    print('Computing average max accuracy on test sets...')
    print('Splitting edges per fold...')
    splits = cv_edges(edges, K)

    max_accs = np.zeros(K)

    for i, split in enumerate(splits):
        _, test_e = split
        test_e = tuple(test_e)
        max_test_acc = get_max_acc(test_e)
        max_accs[i] = max_test_acc

    print('Average max accuracy on test sets is: %.3f.' % max_accs.mean())


if __name__ == '__main__':
    np.random.seed(1)
    split_sushi_data(K=5)
    get_sushi_max_acc(K=5)
