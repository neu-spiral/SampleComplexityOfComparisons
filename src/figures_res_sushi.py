"""
Read and plot synthetic experiment results
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_NbKT(path):
    """
    Plot sushi N by KT dist.

    Assumes all experiments are finished
    and all results are saved.
    """
    methods = ['1', '2']
    Ns = range(22, 41, 2)
    kts = [np.zeros((5, len(Ns))) for _ in range(len(methods))]

    for method in methods:
        with open(path+method, 'rb') as f:
            results = pickle.load(f)
            for i, N in enumerate(Ns):
                for cvk in range(5):
                    kts[eval(method)-1][cvk][i] = results[N][cvk]

    plt.rc('font', family='serif')
    _, ax = plt.subplots()
    mean_average = kts[0].mean(axis=0)
    sd_average = kts[0].std(axis=0)

    mean_logistic = kts[1].mean(axis=0)
    sd_logistic = kts[1].std(axis=0)
    plt.plot(Ns, mean_average, label='Average')
    plt.fill_between(Ns, mean_average - sd_average,
                     mean_average + sd_average, alpha=0.2)

    plt.plot(Ns, mean_logistic, label='Logistic')
    plt.fill_between(Ns, mean_logistic - sd_logistic,
                     mean_logistic + sd_logistic, alpha=0.2)

    ax.annotate(r'$N$', xy=(.95, 0), xytext=(15, -5),
                ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points', fontsize=16)

    plt.ylabel(r'$\tau(\hat\beta, \beta)$', fontsize=16)
    plt.ylim(0, 1)
    plt.legend(loc=2, fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path+'../Res-Sushi.pdf', format='pdf', transparent=True)
    plt.close()


if __name__ == '__main__':
    home_path = str(Path.home())
    plot_NbKT(home_path + '/Res-Sushi/')
