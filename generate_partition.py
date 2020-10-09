from pandas import read_csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from networkx.generators.random_graphs import random_regular_graph
from sklearn.preprocessing import PolynomialFeatures
import os.path
from os import path
from scipy.io import savemat,loadmat


def generate_abs_labels(movie_scores):
    # This function generate absolute labels for each user
    # by using her average score. + 1 means above average -1 means below.

    # Input: - movie_scores: list of tuples. In each tuple, there is (movie_id, score by user).
    # Output - m x 3 ndarray, abs_label_user, the first column is the movie id
    # and the second column is the absolute labels, third column is the score.

    movie_ids = []
    absolute_labels = []
    scores = []
    average_score = 1.*sum(score for _, score in movie_scores)/len(movie_scores)

    for movie_id, movie_score in movie_scores:
        movie_ids.append(movie_id)
        scores.append(movie_score)
        if movie_score >= average_score:
            absolute_labels.append(1)
        else:
            absolute_labels.append(-1)
    abs_label_user = np.array([movie_ids, absolute_labels, scores]).T

    return abs_label_user


def generate_cmp_labels(movie_scores):
    # This function generates comparison labels per user.

    # Input: - movie_scores: list of tuples. In each tuple, there is (movie_id, score by user).
    # Output - list of tuples, cmp_data_user,
    # In each tuple, (i, j, cmp_label); i ,j are movie ids and cmp_label is in {+1, -1}.

    num_of_movies = len(movie_scores)
    cmp_data_user = []

    for movie_i_ind in range(num_of_movies - 1):
        id_movie_i, score_movie_i = movie_scores[movie_i_ind]
        for movie_j_ind in range(movie_i_ind + 1, num_of_movies):
            id_movie_j, score_movie_j = movie_scores[movie_j_ind]
            if score_movie_i > score_movie_j:
                cmp_data_user.append((id_movie_i, id_movie_j, 1))
            elif score_movie_i < score_movie_j:
                cmp_data_user.append((id_movie_i, id_movie_j, -1))

    return cmp_data_user


#### Netflix Dataset
def generate_features(name):
    data_file = pickle.load(open('../data/Netflix/NetflixFeature.p', 'rb'))
    netflix_one_labeler_feat_file = name
    if not path.exists(netflix_one_labeler_feat_file):
        labelers = data_file.keys()
        labelers.remove('Feature')
        # labeler = 'labeler500'
        max_labeler = labelers[0]
        for count, labeler in enumerate(labelers):
            if len(data_file[labeler]) > len(data_file[max_labeler]):
                max_labeler = labeler
        print max_labeler, len(data_file[max_labeler])
        labeler = max_labeler
        movies_labeler = data_file[labeler]
        print len(movies_labeler)
        abs_label_user = generate_abs_labels(movies_labeler)
        movie_id = abs_label_user[:, 0].astype(np.int)
        movie_scores = abs_label_user[:, 2]
        abs_feat_unnormalized = data_file['Feature']
        poly = PolynomialFeatures(2,include_bias=False)
        abs_feat_unnormalized_poly = poly.fit_transform(abs_feat_unnormalized)

        scaler = StandardScaler()
        # Normalize the features (centralize (zero mean) and standardize (unit variance))
        feat_abs = scaler.fit_transform(abs_feat_unnormalized_poly)[movie_id, :]
        num_samples = feat_abs.shape[0]

        cmp_pairs = []
        cmp_label = []
        cmp_feat = []
        for i in range(num_samples):
            for j in range(i+1,num_samples):
                if movie_scores[i] != movie_scores[j]:
                    cmp_pairs.append((i, j))
                    feat_diff = feat_abs[[i], :] - feat_abs[[j], :]
                    cmp_feat.append(1 * feat_diff)
                    if movie_scores[i] > movie_scores[j]:
                        cmp_label.append(1)
                    else:
                        cmp_label.append(-1)
        cmp_feat = np.concatenate(cmp_feat,axis=0)
        cmp_scaler = StandardScaler()
        cmp_feat_normalized = cmp_scaler.fit_transform(cmp_feat)
        cmp_label = np.array(cmp_label)
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        abs_partitions = list(kf.split(feat_abs))
        f = {'cmp_pairs': cmp_pairs, 'cmp_feat_normalized': cmp_feat_normalized,
                    'cmp_label': cmp_label,'num_samples':np.array(num_samples),'feat_abs':feat_abs,
                         'abs_partitions':abs_partitions}

        savemat(netflix_one_labeler_feat_file,f)
    else:
        f = loadmat(netflix_one_labeler_feat_file)
    return f


def generate_partitions(f,n,k):
    name_base = "../data/Netflix/Netflix_itemCV_k_" + str(k) + "_sub_" + str(n) + ".p"
    cmp_pairs = f['cmp_pairs'].tolist()
    feat_abs = f['feat_abs']
    abs_partitions = f['abs_partitions']
    num_samples = f['num_samples']
    np.random.seed(0)

    cmp_train_ind =[]
    cmp_test_ind = []
    num_train_cmp_fold = []
    num_test_cmp_fold = []
    if not str(k).isdigit():
        # k = \sqrt{n-1}
        if k == 'sqrt':
            k = int(np.sqrt(n-1))
        elif k == 'half':
            k = int(0.5*n)
    else:
        # Either 4 or 0(tournament)
        k = eval(str(k))
        if k == 0:
            k = n-1
    m = n*k/2

    if k!=0:
        while (k >= n):
            n += 1
        if n > num_samples:
            n = num_samples

        np.random.seed(1)
        # select_m_train = np.sort(np.random.choice(num_samples, m, replace=False))
        # abs_partitions = list(kf.split(feat_abs))
        for cv_ind in range(5):
            abs_train = abs_partitions[cv_ind][0][0,:]
            abs_test = abs_partitions[cv_ind][1][0,:]
            if n > abs_train.shape[0]:
                n = abs_train.shape[0]
            graph = random_regular_graph(k, n, 0)
            comps = graph.edges
            select_abs_train = np.sort(abs_train[:n]) #np.sort(np.random.choice(abs_train, n, replace=False))
            mapping = {index:val for index,val in enumerate(list(select_abs_train))}
            cmps_in_graph = [(mapping[i], mapping[j]) for i, j in comps]
            train_fold = []
            test_fold = []
            pair_id = 0
            for i, j in cmp_pairs:
                if i in select_abs_train and j in select_abs_train:
                    if (i, j) in cmps_in_graph or (j,i) in cmps_in_graph:
                        train_fold.append(pair_id)
                elif i in abs_test and j in abs_test:
                        test_fold.append(pair_id)
                pair_id += 1
            cmp_train_ind.append(train_fold[:])
            num_train_cmp_fold.append(len(train_fold))
            cmp_test_ind.append(test_fold[:])
            num_test_cmp_fold.append(len(test_fold))
        print("degree at " + str(k), "n=" + str(n), "m=" + str(n*k/2), "expected " + str(m), "train",num_train_cmp_fold)
    else:
        # Tournament
        # np.random.seed(1)
        # abs_partitions = list(kf.split(feat_abs))
        for cv_ind in range(5):
            abs_train = abs_partitions[cv_ind][0][0,:]
            abs_test = abs_partitions[cv_ind][1][0,:]
            if n > abs_train.shape[0]:
                n = abs_train.shape[0]
            select_abs_train = np.sort(abs_train[:n]) # np.sort(np.random.choice(abs_train, n, replace=False))
            train_fold = []
            test_fold = []
            pair_id = 0
            for i,j in cmp_pairs:
                if i in select_abs_train and j in select_abs_train:
                    train_fold.append(pair_id)
                elif i in abs_test and j in abs_test:
                    test_fold.append(pair_id)
                pair_id += 1
            cmp_train_ind.append(train_fold[:])
            num_train_cmp_fold.append(len(train_fold))
            cmp_test_ind.append(test_fold[:])
            num_test_cmp_fold.append(len(test_fold))
        print("degree at " + str(k), "n=" + str(n), "m=" + str(n*k/2), "expected " + str(m), "train",num_train_cmp_fold)
    out_dict = {'cmp_train_ind':cmp_train_ind,'cmp_test_ind':cmp_test_ind}
    pickle.dump(out_dict,open(name_base,'wb'))
    return


#### Netflix Dataset



if __name__ == "__main__":
    count = 0
    k_list = ['half']  # 0, 8, 'sqrt'
    n_list = [200,400,600,800,1000]#[i*100 for i in range(1,5)] #30, 80, 130, 200, 400, 800, 1200,
    # generate_features
    netflix_one_labeler_feat_file = "../data/Netflix/netflixpoly2_slice.mat"
    f = generate_features(netflix_one_labeler_feat_file)

    # f = loadmat(netflix_one_labeler_feat_file[:-2] + ".mat")
    for k in k_list:
        for n in n_list:
            count += 1
            print "Processing " + str(count) + '/' + str(len(k_list) * len(n_list))
            generate_partitions(f,n, k)