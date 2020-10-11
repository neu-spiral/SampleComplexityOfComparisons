"""
Read and plot synthetic experiment results
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from helpers import read_results


def plot_SNbM(path):
    """
    Plot synthetic N by M.

    Assumes all experiments are finished
    and all results are saved.

    Plots synthetic results as
    N by metrics(d) for each ld, k, method, metric
    """
    seeds, lds, ds, Ns, _, ks, methods, metrics, results = read_results(path)

    plt.rc('font', family='serif')
    for metric in metrics:
        for method in methods:
            for k in ks:
                for ld in lds:
                    # Now in the same figure
                    _, ax = plt.subplots()
                    for d in ds:
                        x = Ns[d]
                        y = np.zeros((len(seeds), x.size))
                        for i, seed in enumerate(seeds):
                            y[i] = results[seed][ld][d][k][method][metric]
                        # Mean and std of metrics
                        my = y.mean(axis=0)
                        sdy = y.std(axis=0)

                        plt.plot(x, my, label='d = %s' % d)
                        plt.fill_between(x, my - sdy, my + sdy, alpha=0.2)
                    if metric == 'err_angle':
                        label = r'$\angle(\hat\beta, \beta)$'
                        lim = 2
                    elif metric == 'err_norm':
                        if method == '1':
                            label = r'$||\hat\beta - c_1\beta||$'
                        elif method == '2':
                            label = r'$||\hat\beta - \beta||$'
                        lim = 4
                    else:
                        label = r'$\tau(\hat\beta, \beta)$'
                        lim = 0.6
                    ax.annotate(r'$N$', xy=(.95,0), xytext=(15, -5),
                                ha='left', va='top', xycoords='axes fraction',
                                textcoords='offset points', fontsize=16)
                    # plt.ylabel(label, fontsize=16)
                    plt.xscale('log')
                    plt.ylim(0, lim)
                    plt.legend(loc=2, fontsize=10)
                    plt.tight_layout()
                    plt.savefig(path+'../Syn-%s-%s-%s-%s.pdf'
                                % (metric, ld, k, method),
                                format='pdf', transparent=True)


if __name__ == '__main__':
    home_path = str(Path.home())
    plot_SNbM(home_path + '/SCResults/')
