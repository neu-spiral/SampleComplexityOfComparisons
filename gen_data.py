import numpy as np
from itertools import combinations
from networkx.generators.random_graphs import random_regular_graph, \
        erdos_renyi_graph


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


if __name__ == '__main__':
    X, X_comp, V, E = gen_data_comp(N=5, d=3, k=2)

    print(X)
    print(V)
    print(E)
