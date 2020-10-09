import pickle
from itertools import combinations
import numpy as np
from scipy.stats import mode
from minimumFeedbackArcSet import construct_graph
from networkx.algorithms.dag import is_directed_acyclic_graph

def obtain_majority_comparison(cmp_label_array_multi_expert, indices, ):
    cmp_label_array_multi = cmp_label_array_multi_expert[indices, :]
    cmp_label_majority, _ = mode(cmp_label_array_multi, axis=1)
    return cmp_label_majority

partition_file_name='../data/ROP/complexity.p'
partition_file = pickle.load(open(partition_file_name,'rb'))

class_feat = partition_file['class_feat']
RSD_labels = partition_file['RSD_labels']
cmp_feat = partition_file['cmp_feat']
cmp_label_array_multi_expert = partition_file['cmp_label_array_multi_expert']
cmp_pair_array_single_expert = partition_file['cmp_pair_array_single_expert']
RSD_train_plus_partition = partition_file['RSD_train_plus_partition']
RSD_test_plus_partition = partition_file['RSD_test_plus_partition']
cmp_train_plus_partition_list = partition_file['cmp_train_plus_partition_list']
cmp_test_plus_partition_list = partition_file['cmp_test_plus_partition_list']

k_acyclic = []
for k in range(5):
    ind_k = cmp_test_plus_partition_list[k]
    num_edges = len(ind_k)
    cmp_maj_label = obtain_majority_comparison(cmp_label_array_multi_expert,ind_k)
    # G = construct_graph(cmp_pair_array_single_expert, cmp_maj_label)
    num_edge_removed = 1
    acyclic_graph_found = False
    ind_k_acyclic = []
    k_acyclic_num_edges_removed = []
    while (not acyclic_graph_found):
        print str(k)+"-th fold with "+str(num_edge_removed) +" edges removed"
        for ind_removed in combinations(range(num_edges),num_edge_removed):
            ind_of_ind_k_kept_edges = np.ones((num_edges,),dtype=bool)
            ind_of_ind_k_kept_edges[np.array(ind_removed)] = False
            ind_k_edges_may_acyclic = ind_k[ind_of_ind_k_kept_edges]
            cmp_pair_kept =  cmp_pair_array_single_expert[ind_k_edges_may_acyclic,:]
            cmp_label_kept = obtain_majority_comparison(cmp_label_array_multi_expert,ind_k_edges_may_acyclic)
            G = construct_graph(cmp_pair_array_single_expert, cmp_maj_label)
            if is_directed_acyclic_graph(G):
                acyclic_graph_found = True
                ind_k_acyclic.append(1*ind_k_edges_may_acyclic)
        num_edge_removed += 1
        if num_edge_removed>=0.5*num_edges:
            print "Remove more than half edges in fold " +str(k)
    k_acyclic.append(ind_k_acyclic[:])
    k_acyclic_num_edges_removed.append(num_edge_removed)
save_file_name='../data/ROP/complexity_no_cycle_test.p'
partition_file['cmp_test_no_cycle_plus_partition_list'] = k_acyclic
partition_file['cmp_test_no_cycle_num_edges_removed'] = k_acyclic_num_edges_removed
pickle.dump(partition_file)




