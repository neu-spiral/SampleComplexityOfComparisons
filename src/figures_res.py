"""
Read and plot synthetic experiment results
"""
import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_SNbM(path):
    """
    Plot synthetic N by M.

    Assumes all experiments are finished
    and all results are saved.

    Reads results from path.
    Plots synthetic results as
    N by metrics for each ld, d, k, method

    Each file in path is a dict with keys below
    seed, ld, d, k, method, Ns, Ms,
    then for each N in Ns 'err_angle', 'err_norm', 'kt_dist'
    """
    # Hardcode error metrics
    metrics = ['err_angle', 'err_norm', 'kt_dist']
    # Files in given path
    files = os.listdir(path)
    # IF ANY FILES ARE TO BE SKIPPED, ADD LINES HERE.
    # Split by dash
    sfiles = [f.split('-') for f in files]
    # Different seed values
    seeds = {f[0] for f in sfiles}
    # Different ld values
    lds = {f[1] for f in sfiles}
    # Different d values
    ds = {f[2] for f in sfiles}
    # Different k values
    ks = {f[-2] for f in sfiles}
    # Different method values
    methods = {f[-1] for f in sfiles}
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
        seed, ld, d, N1, N2, k, method = f.split('-')

        with open(path+f, 'rb') as fb:
            resd = pickle.load(fb)  # Results dictionary

        Ns[d] = resd['Ns']
        Ms[d] = {k: resd['Ms']}

        for metric in metrics:
            metric_values = np.zeros(Ns[d].size)
            for i, N in enumerate(resd['Ns']):
                metric_values[i] = resd[N][metric]
            results[seed][ld][d][k][method][metric] = metric_values

    # Start plotting
    # Settings of plt
    plt.rc('font', family='serif')
    plt.figure(figsize=(10, 10))

    # Plot trade-off curve.
    x, y = 1, 1
    plt.subplot(221)
    plt.plot(x, y)
    plt.xlabel(r'$\|x\|_1^1$', fontsize=16)
    plt.xticks(x[::10], ['a' for _ in x[::10]], rotation=20)
    plt.ylabel(r'$\|y\|^1_1$', fontsize=16)
    plt.xscale('log')
    plt.title('x vs y', fontsize=16)

    # Plot entries of x vs. gamma.
    plt.subplot(224)
    for i in range(1, 6):
        y = x**(1/i)
        plt.plot(x, y, label=r'$\gamma$ = %.2f' % (1/i))

    plt.fill_between(x, np.ones(100)*.5, np.ones(100)*1.5,
                     color='red', alpha=0.2)

    plt.xlabel(r'x', fontsize=16)
    plt.ylabel(r'y', fontsize=16)
    plt.xscale('log')
    plt.title(r'$x^\gamma$', fontsize=16)

    plt.grid()
    plt.legend(loc=2, fontsize=10)
    plt.tight_layout()
    plt.show()
    plt.savefig('./fig.pdf', format='pdf', transparent=True)
    os.remove('./fig.pdf')


if __name__ == '__main__':
    home_path = str(Path.home())
    plot_SNbM(home_path + '/SCResults/')
