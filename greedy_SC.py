import operator
import numpy as np
import networkx as nx
from copy import deepcopy

from gen_data import gen_data_comp


def has_cycle(G):
    """
    Check if G has a cycle.
    :param G: networkx directed graph
    """

    try:
        nx.algorithms.cycles.find_cycle(G)
        return True
    except Exception:
        return False


def greedy_node(source_G, length=0):
    """
    Runs the natural greedy algorithm on G and returns G' with cycles removed
    """

    G = deepcopy(source_G)
    copy_G = deepcopy(source_G)

    S = []  # Set of removed nodes
    if length == 0:
        length = len(G)
    counts = np.zeros(length)  # Counter for each node

    # Find cycles
    cycles = []

    while has_cycle(G):
        edges = nx.algorithms.cycles.find_cycle(G)  # Find any cycle
        cycles.append(edges)  # Save it
        for edge in edges:  # Remove each edge in current cycle
            G.remove_edge(edge[0], edge[1])

    print('Found # of cycles:', len(cycles))

    # Below line goes from [[(0,1), (1, 0)]] to [[0, 1]]
    # Edges to vertices
    cycles = list(map(lambda x: [y[0] for y in x], cycles))

    # Counts of each node appearence in cycles
    for cycle in cycles:
        for node in cycle:
            counts[node] += 1

    while cycles:
        current_node = np.argmax(counts)
        S.append(current_node)

        for cycle in cycles:
            if current_node in cycle:
                cycles.remove(cycle)
                for node in cycle:
                    counts[node] -= 1

    copy_G.remove_nodes_from(S)

    if has_cycle(copy_G):
        copy_G = greedy_node(copy_G, length)

    return copy_G


def greedy_edge(source_G):
    """
    Runs the natural greedy algorithm on G and returns G' with cycles removed
    """

    G = deepcopy(source_G)
    copy_G = deepcopy(source_G)

    S = []  # Set of removed edges
    counts = {}  # Counter for each edge

    # Find cycles
    cycles = []

    while has_cycle(G):
        edges = nx.algorithms.cycles.find_cycle(G)  # Find any cycle
        cycles.append(edges)  # Save it
        edge_t_r = int(len(edges)*np.random.rand())  # Remove a random edge
        G.remove_edge(edges[edge_t_r][0], edges[edge_t_r][1])

    print('Found # of cycles:', len(cycles))

    # Counts of each node appearence in cycles
    for cycle in cycles:
        for edge in cycle:
            if edge in counts:
                counts[edge] += 1
            else:
                counts[edge] = 1

    while cycles:
        current_edge = max(counts.items(), key=operator.itemgetter(1))[0]
        S.append(current_edge)

        for cycle in cycles:
            if current_edge in cycle:
                cycles.remove(cycle)
                for edge in cycle:
                    counts[edge] -= 1

    copy_G.remove_edges_from(S)

    if has_cycle(copy_G):
        copy_G = greedy_edge(copy_G)

    return copy_G


if __name__ == '__main__':
    np.set_printoptions(precision=2, threshold=1e99, linewidth=100)
    # TODO: Parse args to get args
    N = 1000
    d = 3
    k = 16
    p_switch = 0.2

    X, X_comp, V, E = gen_data_comp(N=N, d=d, k=k)

    for e in E:
        if np.random.rand() < p_switch:
            e[0], e[1] = e[1], e[0]

    G = nx.DiGraph()
    G.add_edges_from(E)

    acyclic_G = greedy_edge(G)
    print('Running greedy_edge')
    print('len(G)', len(G))
    print('len acyclic_G', len(acyclic_G))
    print('edges of G', len(G.edges()))
    print('edges of acyclic_G', len(acyclic_G.edges()))
    print('cycle in G?', has_cycle(G))
    print('cycle in acyclic_G', has_cycle(acyclic_G))
