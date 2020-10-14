"""
Helper codes
"""
from os import path, listdir
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold


def cv_edges(edges, K):
    """
    Given a list of (u, v) find K fold cv split of nodes
    [[tr1, te1], [tr2, te2], ...]
    """
    unq_nodes = get_unq_nodes(edges)
    seed = int(np.random.rand()*1e6)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    split_edges = [[[], []] for _ in range(K)]
    for k, (train, test) in enumerate(kf.split(unq_nodes)):
        for edge in edges:
            u, v = edge
            if u in train and v in train:
                split_edges[k][0].append(edge)
            elif u in test and v in test:
                split_edges[k][1].append(edge)

    return split_edges


def get_unq_nodes(edges):
    """
    Get a list of unique nodes from a list of edges
    """
    unq_nodes = list({u for edge in edges for u in edge})

    return unq_nodes


def read_results_synth(path):
    """
    Read path for experiment results
    return in a dict
    """
    # Hard code metrics
    metrics = ['err_angle', 'err_norm', 'kt_dist']
    # Files in given path
    files = listdir(path)
    # IF ANY FILES ARE TO BE SKIPPED, ADD LINES HERE.
    # Split by dash
    sfiles = [f.split('-') for f in files]
    # Different seed values
    seeds = list({f[0] for f in sfiles})
    # Different ld values
    lds = list({f[1] for f in sfiles})
    # Different d values
    ds = list({f[2] for f in sfiles})
    # Different k values
    ks = list({f[-2] for f in sfiles})
    # Different method values
    methods = list({f[-1] for f in sfiles})
    # Different N and M values for each d
    Ns = {}
    Ms = {}

    # Form skeleton of results dict
    results = {}
    for seed in seeds:
        results[seed] = {}
        for ld in lds:
            results[seed][ld] = {}
            for d in ds:
                results[seed][ld][d] = {}
                for k in ks:
                    results[seed][ld][d][k] = {}
                    for method in methods:
                        results[seed][ld][d][k][method] = {}
                        for metric in metrics:
                            results[seed][ld][d][k][method][metric] = None

    for f in files:
        seed, ld, d, _, _, k, method = f.split('-')

        with open(path+f, 'rb') as fb:
            resd = pickle.load(fb)  # Results dictionary

        Ns[d] = resd['Ns']
        Ms[d] = {k: resd['Ms']}

        for metric in metrics:
            metric_values = np.zeros(Ns[d].size)
            for i, N in enumerate(resd['Ns']):
                metric_values[i] = resd[N][metric]
            results[seed][ld][d][k][method][metric] = metric_values

    return seeds, lds, ds, Ns, Ms, ks, methods, metrics, results


def read_results_synth_by_M(path):
    """
    Read path for experiment results
    return in a dict
    """
    # Hard code metrics
    metrics = ['err_angle', 'err_norm', 'kt_dist']
    # Files in given path
    files = listdir(path)
    # IF ANY FILES ARE TO BE SKIPPED, ADD LINES HERE.
    # Split by dash
    sfiles = [f.split('-') for f in files]
    # Different seed values
    seeds = list({f[0] for f in sfiles})
    # Different ld values
    lds = list({f[1] for f in sfiles})
    # Different d values
    ds = list({f[2] for f in sfiles})
    # Different method values
    methods = list({f[-1] for f in sfiles})
    # Different N and M values for each d
    Ns = {}
    Ms = {}

    # Form skeleton of results dict
    results = {}
    for seed in seeds:
        results[seed] = {}
        for ld in lds:
            results[seed][ld] = {}
            for d in ds:
                results[seed][ld][d] = {}
                for method in methods:
                    results[seed][ld][d][method] = {}
                    for metric in metrics:
                        results[seed][ld][d][method][metric] = None

    for f in files:
        seed, ld, d, _, M, method = f.split('-')

        with open(path+f, 'rb') as fb:
            resd = pickle.load(fb)  # Results dictionary

        Ns[d] = resd['Ns']
        Ms[d] = resd['Ms']

        for metric in metrics:
            metric_values = np.zeros(Ms[d].size)
            for i, M in enumerate(resd['Ms']):
                metric_values[i] = resd[M][metric]
            results[seed][ld][d][method][metric] = metric_values

    return seeds, lds, ds, Ns, Ms, methods, metrics, results


def check_exp(args, name):
    """
    If experiment is already finished, stop early.
    """
    file_path = get_exp_path(args, name)
    if not path.exists(file_path):
        return
    else:
        raise Exception('%s exists.' % file_path)


def save_results(results, args, name):
    """
    Save results dict in disk.
    """
    # Get file path to write
    file_path = get_exp_path(args, name)
    # Write to disk
    with open(file_path, 'wb+') as f:
        pickle.dump(dict(results), f)


def get_exp_path(args, name):
    """
    Generate file path for current experiment.
    """
    home_path = str(Path.home())
    if name == 'synth':
        # Find path to home dir
        file_path = home_path + '/Res-Synth/%i-%.3f-%i-%i-%i-%i-%i' \
            % (args.seed, args.ld, args.d, args.N1,
               args.N2, args.k, args.method)
    elif name == 'synth_by_M':
        # Find path to home dir
        file_path = home_path + '/Res-Synth-M/%i-%.3f-%i-%i-%i-%i' \
            % (args.seed, args.ld, args.d, args.N,
               args.M, args.method)
    elif name == 'sushi':
        file_path = home_path + '/Res-Sushi/%i' % args.method

    return file_path


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
    and smallest eigen value is ld.
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
    # N1 = np.log10(N1)
    # N2 = np.log10(N2)
    # N = np.ceil(np.logspace(N1, N2, 10)).astype(np.int32)

    if N2 > 4000:
        span1 = np.linspace(N1, N2/2, 15)
        span2 = np.linspace(N2/2 + N2/10, N2, 4)
        N = np.concatenate([span1, span2]).astype(np.int32)
    else:
        N = np.linspace(N1, N2, 10).astype(np.int32)

    if k == 1:
        M = N
    elif k == 2:
        M = np.ceil(N*np.log(np.log(N))).astype(np.int32)
    elif k == 3:
        M = np.ceil(N*np.log(N)).astype(np.int32)
    elif k == 4:
        M = np.ceil(N*N**.5).astype(np.int32)

    return N, M
