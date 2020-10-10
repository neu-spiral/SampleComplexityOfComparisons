"""
Read and plot synthetic experiment results
"""
import os
from collections import defaultdict
import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_N(path):
    """
    Read results from path.
    Each file in path is a dict with keys
    seed, ld, d, k, method, Ns, Ms, then for eacn N in Ns
    'err_angle', 'err_norm', 'kt_dist'
    """
    files = os.listdir(path)
    results = defaultdict(dict)

    for f in files:
        with open(path+f, 'rb') as op_f:
            c_res = pickle.load(op_f)
        
        


if __name__ == '__main__':
    # Some data to be plotted
    x = np.logspace(-3, .5, 100)
    y = x*2
    print('x:', x)

    # Setting of plt
    plt.rc('font', family='serif')
    plt.figure(figsize=(10, 10))

    # Plot trade-off curve.
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







