"""
Data generation and reading related
"""
from path import Path
from random import choices
import numpy as np
from helper import cv_edges


def get_data(N, M, beta, f_mean, f_cov):
    """
    Generates d dimensional gaussian vectors.
    Uniformly at random pairwise comparisons are chosen.
    Labels are generated with Bradley Terry model.
    """
    # Data for covariance estimation
    X = np.random.multivariate_normal(f_mean, f_cov, N)

    # Data for comparisons
    X1 = np.random.multivariate_normal(f_mean, f_cov, N)

    # Uniformly at random edges
    vertices = list(range(N))
    u = choices(vertices, k=M)
    v = choices(vertices, k=M)

    # An edge from u to v implies v beats u
    # Comparison features
    XC = X1[v] - X1[u]

    # beta^T(X-Y)
    scores = XC @ beta
    # BTL Probability
    p = (1+np.exp(-scores))**-1
    # Uniform random variables
    urv = np.random.rand(M)
    # BTL Labels
    yn = np.sign(p - urv)
    # True labels
    y = np.sign(scores)

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
                    edges.append((line[j], v))
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
                    edges_b.append((line[j], v))
    # Mean scores for nodes
    scores_b = [[] for _ in range(100)]
    with open(path+'sushi3b.5000.10.score') as f:
        for line in f:
            line = line.strip().split()
            for i, score in enumerate(line):
                score = int(score)
                if score != -1:
                    scores_b[i].append(score)

    m_scores = np.array([np.mean(x) for x in scores_b])
    std_scores = np.array([np.std(x) for x in scores_b])
    print(m_scores)
    print(std_scores)
    scores_b = m_scores - 2*std_scores

    return feats, profs, edges, edges_b, scores_b


def get_sushi_data():
    """
    """
    home_path = Path.home()
    X, _, _, edges, scores = read_sushi_data(home_path + '/Data/sushi3-2016/')

    return X


if __name__ == '__main__':
    feats, profs, edges, edges_b, scores_b = get_sushi_data()
    print(scores_b)
