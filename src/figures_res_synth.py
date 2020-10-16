"""
Read and plot synthetic experiment results
"""
from pathlib import Path
from argparse import ArgumentParser
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from helpers import read_results_synth


def plot_SmbN(path, eps1, eps2, legend, y_axis):
    """
    Plot synthetic metrics as a function of N.

    Assumes all experiments are finished
    and all results are saved.

    Plots synthetic results as
    N by metrics(d) for each ld, k, method, metric
    """
    seeds, lds, pes, ds, Ns, _, ks, methods, metrics, results = \
        read_results_synth(path)
    ds.sort(key=float)

    plt.rc('font', family='serif')
    markers_list = ['x-', 's--', '^-.', 'o:']

    for metric in metrics:
        if metric == 'kt_dist':
            continue
        for method in methods:
            for k in ks:
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
                            # Remove some d here
                            # So that the plot is not dense
                            if (int(d)-10) % 80 != 0:
                                continue
                            if ld == '1.00':
                                line = np.ones(Ns[d].size)
                            else:
                                line = eps2*np.ones(Ns[d].size)
                            x = Ns[d]
                            y = np.zeros((len(seeds), x.size))
                            for i, seed in enumerate(seeds):
                                y[i] = results[seed][ld][pe][d][k][method][metric]
                            # Mean and std of metrics
                            my = y.mean(axis=0)
                            sdy = y.std(axis=0)

                            plt.plot(x, my, next(markers), label=r'$d = %s$' % d,
                                     markersize=3)
                            plt.fill_between(x, my - sdy, my + sdy, alpha=0.2)
                        if metric == 'err_angle':
                            plt.plot(x, line, 'k-.')
                            label = r'$\angle(\hat\beta, \beta)$'
                            lim = 1.5
                        elif metric == 'err_norm':
                            if ld != '1.000':
                                plt.plot(x, line, 'k-.')
                            if method == '1':
                                label = r'$||\hat\beta - c_1\beta||$'
                            elif method == '2':
                                label = r'$||\hat\beta - \beta||$'
                            lim = 4
                        else:
                            label = r'$\tau(\hat\beta, \beta)$'
                            lim = 0.4
                        ax.annotate(r'$N$', xy=(.95, 0), xytext=(18, -5),
                                    ha='left', va='top', xycoords='axes fraction',
                                    textcoords='offset points', fontsize=16)
                        if y_axis:
                            plt.ylabel(label, fontsize=16)
                        plt.xscale('log')
                        plt.ylim(0, lim)
                        plt.grid()
                        if legend:
                            plt.legend(loc='upper right', fontsize=16)
                        plt.tight_layout()
                        plt.savefig(path+'../Syn-mbN-%s-%s-%s-%s-%s.pdf'
                                    % (metric, ld, pe, k, method),
                                    format='pdf', transparent=True)
                        plt.close()


def plot_SNbd(path, eps1, eps2, legend, y_axis):
    """
    Plot minimum N that achieves epsilon by d
    """
    seeds, lds, pes, ds, Ns, _, ks, _, _, results = read_results_synth(path)
    size = len(Ns[ds[0]])
    lds.sort(key=float)
    ds.sort(key=float)
    pes.sort(key=float)

    markers_list = ['x-', 's--', '^-.', 'o:']

    x = [int(d) for d in ds]
    for k in ks:
        for pe in pes:
            # Now in the same figure
            _, ax = plt.subplots()
            markers = cycle(markers_list)
            for ld in lds:
                if ld not in ['0.005', '0.100', '1.000']:
                    continue
                if float(ld) == 1:
                    epsilon = eps1
                else:
                    epsilon = eps2
                min_N = np.zeros(len(ds))
                for j, d in enumerate(ds):
                    y = np.zeros((len(seeds), size))
                    for i, seed in enumerate(seeds):
                        value = results[seed][ld][pe][d][k]['1']['err_angle']
                        y[i] = value
                    # Mean and std of metric
                    my = y.mean(axis=0)
                    loc = np.where(epsilon > my)[0][0]
                    min_N[j] = Ns[ds[0]][loc]
                label = r'$\lambda_d = %s$' % ld
                plt.plot(x, min_N, next(markers), label=label, markersize=3)
            if legend:
                plt.legend(loc='upper right', fontsize=16)
            plt.grid()
            if y_axis:
                plt.ylabel(r'$N$', fontsize=16)
            plt.ylim(0, 30500)
            ax.annotate(r'$d$', xy=(.95, 0), xytext=(18, -5),
                        ha='left', va='top', xycoords='axes fraction',
                        textcoords='offset points', fontsize=16)
            plt.tight_layout()
            plt.savefig(path+'../Syn-Nbd-%.2f-%s-%s.pdf'
                        % (epsilon, pe, k),
                        format='pdf', transparent=True)
            plt.close()


def plot_SNbld(path, eps1, eps2, legend, y_axis):
    """
    Plot minimum N that achieves epsilon by lambda d
    """
    seeds, lds, pes, ds, Ns, _, ks, _, _, results = read_results_synth(path)
    size = len(Ns[ds[0]])
    ds.sort(key=float)
    lds.sort(key=lambda x: -1*float(x))
    pes.sort(key=float)

    markers_list = ['x-', 's--', '^-.', 'o:']

    x = [float(ld) for ld in lds]
    for k in ks:
        for pe in pes:
            # Now in the same figure
            _, ax = plt.subplots()
            markers = cycle(markers_list)
            for d in ds:
                if d not in ['10', '90', '250']:
                    continue
                min_N = np.zeros(len(lds))
                for j, ld in enumerate(lds):
                    if float(ld) == 1:
                        epsilon = eps1
                    else:
                        epsilon = eps2
                    y = np.zeros((len(seeds), size))
                    for i, seed in enumerate(seeds):
                        value = results[seed][ld][pe][d][k]['1']['err_angle']
                        y[i] = value
                    # Mean and std of metric
                    my = y.mean(axis=0)
                    loc = np.where(epsilon > my)[0][0]
                    min_N[j] = Ns[ds[0]][loc]
                label = r'$d = %s$' % d
                plt.plot(x, min_N, next(markers), label=label, markersize=3)
            if legend:
                plt.legend(loc='upper right', fontsize=16)
            plt.grid()
            if y_axis:
                plt.ylabel(r'$N$', fontsize=16)
            plt.ylim(0, 30500)
            ax.annotate(r'$\lambda_d$', xy=(.95, 0), xytext=(18, -5),
                        ha='left', va='top', xycoords='axes fraction',
                        textcoords='offset points', fontsize=16)
            plt.tight_layout()
            plt.savefig(path+'../Syn-Nbld-%.2f-%s-%s.pdf'
                        % (epsilon, pe, k),
                        format='pdf', transparent=True)
            plt.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Run synthetic experiments.')
    parser.add_argument('-l', type=int)
    parser.add_argument('-y', type=int)
    parser.add_argument('-eps', type=float, default=0.3,
                        help='Epsilon val for ld=1.')
    args = parser.parse_args()

    home_path = str(Path.home())
    plot_SmbN(home_path + '/Res-Synth/', args.eps, args.eps, args.l, args.y)
    plot_SNbd(home_path + '/Res-Synth/', args.eps, args.eps, args.l, args.y)
    plot_SNbld(home_path + '/Res-Synth/', args.eps, args.eps, args.l, args.y)
