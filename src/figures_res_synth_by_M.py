"""
Read and plot synthetic experiment results
"""
from pathlib import Path
from argparse import ArgumentParser
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from helpers import read_results_synth_by_M


def plot_SmbM(path, legend, y_label):
    """
    Plot synthetic metric by M.

    Assumes all experiments are finished
    and all results are saved.
    """
    seeds, lds, pes, ds, Ns, Ms, methods, _, results = \
        read_results_synth_by_M(path)
    ds.sort(key=float)

    markers_list = ['x-', 's--', '^-.', 'o:']

    plt.rc('font', family='serif')
    for metric in ['err_norm', 'err_angle']:
        for method in methods:
            for ld in lds:
                if ld not in ['0.005', '1.000']:
                    continue
                for pe in pes:
                    # Now in the same figure
                    markers = cycle(markers_list)
                    _, ax = plt.subplots()
                    for d in ds:
                        if d not in ['10', '90', '250']:
                            continue
                        N = Ns[d]
                        M = Ms[d]
                        x = M
                        y = np.zeros((len(seeds), x.size))
                        for i, seed in enumerate(seeds):
                            y[i] = results[seed][ld][pe][d][method][metric]
                        # Mean and std of metrics
                        my = y.mean(axis=0)
                        sdy = y.std(axis=0)

                        plt.plot(x, my, next(markers), label=r'$d = %s$' % d,
                                 markersize=3)
                        plt.fill_between(x, my - sdy, my + sdy, alpha=0.2)
                    # plt.axvline(N)
                    plt.axvline(N*np.log(N), ls='--', color='k')
                    if metric == 'err_norm':
                        lim = 4
                        if method == '1':
                            label = r'$||\hat\beta - c_1\beta||$'
                        elif method == '2':
                            label = r'$||\hat\beta - \beta||$'
                    elif metric == 'err_angle':
                        lim = 1.5
                        label = r'$\angle (\hat\beta, \beta)$'
                    ax.annotate(r'$M$', xy=(.95, 0), xytext=(15, -5),
                                ha='left', va='top', xycoords='axes fraction',
                                textcoords='offset points', fontsize=20)
                    if y_label:
                        plt.ylabel(label, fontsize=20)
                    plt.xscale('log')
                    plt.ylim(0, lim)
                    plt.grid()
                    if legend:
                        plt.legend(loc='upper right', fontsize=20)
                    plt.tight_layout()
                    plt.savefig(path+'../Syn-mbM-%s-%s-%s-%s.pdf'
                                % (metric, ld, pe, method),
                                format='pdf', transparent=True)
                    plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-l', type=int)
    parser.add_argument('-y', type=int)
    args = parser.parse_args()

    home_path = str(Path.home())
    plot_SmbM(home_path + '/Res-Synth-M/', args.l, args.y)
