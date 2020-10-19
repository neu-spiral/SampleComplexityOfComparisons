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
    methods = ['1']
    Ns = range(22, 41, 2)
    accs = [np.zeros((5, len(Ns))) for _ in range(len(methods))]
    kts = [np.zeros((5, len(Ns))) for _ in range(len(methods))]

    for method in methods:
        with open(path+method, 'rb') as f:
            results = pickle.load(f)
            for i, N in enumerate(Ns):
                for cvk in range(5):
                    accs[eval(method)-1][cvk][i] = results[N]['acc'][cvk]
                    kts[eval(method)-1][cvk][i] = results[N]['kt_dist'][cvk]

    plt.rc('font', family='serif')
    _, ax = plt.subplots()
    mean_average = kts[0].mean(axis=0)
    sd_average = kts[0].std(axis=0)

    plt.plot(Ns, mean_average, label='Average')
    plt.fill_between(Ns, mean_average - sd_average,
                     mean_average + sd_average, alpha=0.2)

    ax.annotate(r'$N$', xy=(.95, 0), xytext=(18, -5),
                ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points', fontsize=16)

    plt.ylabel(r"Kendall's Tau", fontsize=16)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path+'../Res-Sushi-KT.pdf', format='pdf', transparent=True)
    plt.close()

    _, ax = plt.subplots()
    mean_average = accs[0].mean(axis=0)
    sd_average = accs[0].std(axis=0)

    plt.plot(Ns, mean_average, label='Average')
    plt.fill_between(Ns, mean_average - sd_average,
                     mean_average + sd_average, alpha=0.2)

    plt.plot(Ns, 0.677*np.ones(len(Ns)), 'k--')

    ax.annotate(r'$N$', xy=(.95, 0), xytext=(18, -5),
                ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points', fontsize=16)

    plt.ylabel(r'Accuracy', fontsize=16)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path+'../Res-Sushi-Acc.pdf', format='pdf', transparent=True)
    plt.close()


if __name__ == '__main__':
    home_path = str(Path.home())
    plot_NbKT(home_path + '/Res-Sushi/')
