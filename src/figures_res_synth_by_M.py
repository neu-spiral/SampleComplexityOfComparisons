"""
Read and plot synthetic experiment results
"""
from pathlib import Path
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from helpers import read_results_synth_by_M


def plot_SmbM(path):
    """
    Plot synthetic metric by M.

    Assumes all experiments are finished
    and all results are saved.
    """
    seeds, lds, ds, Ns, Ms, methods, _, results = \
        read_results_synth_by_M(path)
    ds.sort(key=float)

    plt.rc('font', family='serif')
    markers_list = ['x-', 's-', '^-', 'o-', '>-']
    for method in methods:
        for ld in lds:
            # Now in the same figure
            markers = cycle(markers_list)
            _, ax = plt.subplots()
            for d in ds:
                N = Ns[d]
                M = Ms[d]
                x = M
                y = np.zeros((len(seeds), x.size))
                for i, seed in enumerate(seeds):
                    y[i] = results[seed][ld][d][method]['err_norm']
                # Mean and std of metrics
                my = y.mean(axis=0)
                sdy = y.std(axis=0)

                plt.plot(x, my, next(markers), label=r'$d = %s$' % d,
                         markersize=3)
                plt.fill_between(x, my - sdy, my + sdy, alpha=0.2)
            # plt.axvline(N)
            plt.axvline(N*np.log(N), ls='--', color='k', label=r'$M=N\log N$')
            if method == '1':
                label = r'$||\hat\beta - c_1\beta||$'
            elif method == '2':
                label = r'$||\hat\beta - \beta||$'
            ax.annotate(r'$M$', xy=(.95, 0), xytext=(15, -5),
                        ha='left', va='top', xycoords='axes fraction',
                        textcoords='offset points', fontsize=16)
            plt.ylabel(label, fontsize=16)
            plt.xscale('log')
            plt.ylim(0, 3)
            plt.grid()
            plt.legend(loc='upper right', fontsize=10)
            plt.tight_layout()
            plt.savefig(path+'../Syn-mbM-%s-%s.pdf' % (ld, method),
                        format='pdf', transparent=True)
            plt.close()


if __name__ == '__main__':
    home_path = str(Path.home())
    plot_SmbM(home_path + '/Res-Synth-M/')
