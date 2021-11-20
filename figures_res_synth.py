"""
Read and plot synthetic experiment results
"""
from pathlib import Path
from argparse import ArgumentParser
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from src.helpers import read_results_synth


def plot_SmbN(path, legend, y_axis):
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
        for method in methods:
            if method == '1':
                epsilon = 0.7
            else:
                epsilon = 0.5
            for k in ks:
                for ld in lds:
                    if ld not in ['0.005', '1.000']:
                        continue
                    for pe in pes:
                        if pe not in ['0.0', '0.4']:
                            continue
                        # Now in the same figure
                        markers = cycle(markers_list)
                        _, ax = plt.subplots()
                        for d in ds:
                            if d not in ['10', '100']:
                                continue
                            # Remove some d here
                            # So that the plot is not dense
                            line = epsilon*np.ones(Ns[d].size)
                            x = Ns[d]
                            y = np.zeros((len(seeds), x.size))
                            for i, seed in enumerate(seeds):
                                y[i] = results[seed][ld][pe][d][k][
                                    method][metric]
                            # Mean and std of metrics
                            my = y.mean(axis=0)
                            sdy = y.std(axis=0)

                            plt.plot(x, my, next(markers),
                                     label=r'$d = %s$' % d, markersize=3)
                            plt.fill_between(x, my - sdy, my + sdy, alpha=0.2)
                        if metric == 'err_angle':
                            plt.plot(x, line, 'k-.')
                            label = r'$\angle(\hat\beta, \beta)$'
                            lim = 1.5
                        elif metric == 'err_norm':
                            if method == '1':
                                label = r'$||\hat\beta - c_1\beta||$'
                            else:
                                label = r'$c_1||\hat\beta - \beta||$'
                            lim = 2
                        elif metric == 'kt_dist':
                            label = r'$\tau(\hat\beta, \beta)$'
                            lim = 0.4
                        else:
                            label = 'Duration'

                        ax.annotate(r'$N$', xy=(.95, 0), xytext=(18, -5),
                                    ha='left', va='top',
                                    xycoords='axes fraction',
                                    textcoords='offset points', fontsize=20)
                        if y_axis:
                            plt.ylabel(label, fontsize=20)
                        plt.xscale('log')
                        plt.ylim(0, lim)
                        plt.grid()
                        if legend:
                            plt.legend(loc='upper right', fontsize=20)
                        plt.tight_layout()
                        plt.savefig(path+'../Syn-mbN-%s-%s-%s-%s.pdf'
                                    % (metric, ld, pe, method),
                                    format='pdf', transparent=True)
                        plt.close()


def plot_SNbd(path, legend, y_axis):
    """
    Plot minimum N that achieves epsilon by d
    """
    seeds, lds, pes, ds, Ns, _, ks, methods, metrics, results = \
        read_results_synth(path)
    size = len(Ns[ds[0]])
    lds.sort(key=float)
    ds.sort(key=float)
    pes.sort(key=float)

    markers_list = ['x-', 's--', '^-.', 'o:']

    x = [int(d) for d in ds]
    for k in ks:
        for pe in pes:
            for method in methods:
                if method == '1':
                    epsilon = 0.7
                else:
                    epsilon = 0.5
                # Now in the same figure
                _, ax = plt.subplots()
                markers = cycle(markers_list)
                for ld in lds:
                    if ld not in ['0.005', '1.000']:
                        continue
                    min_N = np.zeros(len(ds))
                    for j, d in enumerate(ds):
                        y = np.zeros((len(seeds), size))
                        for i, seed in enumerate(seeds):
                            value = results[seed][ld][pe][d][k][method]['err_angle']
                            y[i] = value
                        # Mean and std of metric
                        my = y.mean(axis=0)
                        loc = np.where(epsilon > my)[0][0]
                        min_N[j] = Ns[ds[0]][loc]
                    label = r'$\lambda_d = %s$' % ld
                    plt.plot(x, min_N, next(markers), label=label, markersize=3)
                if legend:
                    plt.legend(loc='upper left', fontsize=20)
                plt.grid()
                if y_axis:
                    plt.ylabel(r'$N$', fontsize=20)
                plt.ylim(0, 2200)
                ax.annotate(r'$d$', xy=(.95, 0), xytext=(18, -5),
                            ha='left', va='top', xycoords='axes fraction',
                            textcoords='offset points', fontsize=20)
                plt.tight_layout()
                plt.savefig(path+'../Syn-Nbd-%.2f-%s-%s.pdf'
                            % (epsilon, pe, method),
                            format='pdf', transparent=True)
                plt.close()


def plot_SNbld(path, legend, y_axis):
    """
    Plot minimum N that achieves epsilon by lambda d
    """
    seeds, lds, pes, ds, Ns, _, ks, methods, metrics, results = read_results_synth(path)
    size = len(Ns[ds[0]])
    ds.sort(key=float)
    lds.sort(key=lambda x: -1*float(x))
    pes.sort(key=float)

    markers_list = ['x-', 's--', '^-.', 'o:']

    x = [float(ld) for ld in lds]
    for k in ks:
        for pe in pes:
            for method in methods:
                if method == '1':
                    epsilon = 0.7
                else:
                    epsilon = 0.5
                # Now in the same figure
                _, ax = plt.subplots()
                markers = cycle(markers_list)
                for d in ds:
                    if d not in ['10', '100']:
                        continue
                    min_N = np.zeros(len(lds))
                    for j, ld in enumerate(lds):
                        y = np.zeros((len(seeds), size))
                        for i, seed in enumerate(seeds):
                            value = results[seed][ld][pe][d][k][method]['err_angle']
                            y[i] = value
                        # Mean and std of metric
                        my = y.mean(axis=0)
                        loc = np.where(epsilon > my)[0][0]
                        min_N[j] = Ns[ds[0]][loc]
                    label = r'$d = %s$' % d
                    plt.plot(x, min_N, next(markers), label=label, markersize=3)
                if legend:
                    plt.legend(loc='upper right', fontsize=20)
                plt.grid()
                if y_axis:
                    plt.ylabel(r'$N$', fontsize=20)
                plt.ylim(0, 2200)
                ax.annotate(r'$\lambda_d$', xy=(.95, 0), xytext=(18, -5),
                            ha='left', va='top', xycoords='axes fraction',
                            textcoords='offset points', fontsize=20)
                plt.tight_layout()
                plt.savefig(path+'../Syn-Nbld-%.2f-%s-%s.pdf'
                            % (epsilon, pe, method),
                            format='pdf', transparent=True)
                plt.close()


def plot_SmbNcomp(path, legend, y_axis):
    """
    Plot synthetic metrics as a function of N.
    Keep all methods in the same figure.

    Assumes all experiments are finished
    and all results are saved.

    Plots synthetic results as
    N by metrics(d) for each ld, k, method, metric
    """
    seeds, lds, pes, ds, Ns, _, ks, methods, metrics, results = \
        read_results_synth(path)
    ds.sort(key=float)
    methods.sort(key=float)

    plt.rc('font', family='serif')
    markers_list = ['x-', 's--', '^-.', 'o:']

    for metric in metrics:
        for k in ks:
            for ld in lds:
                if ld not in ['0.005', '1.000']:
                    continue
                for pe in pes:
                    if pe not in ['0.0', '0.4']:
                        continue
                    # Now in the same figure
                    markers = cycle(markers_list)
                    _, ax = plt.subplots()
                    for d in ds:
                        if d not in ['100']:
                            continue
                        for method in methods:
                            x = Ns[d]
                            y = np.zeros((len(seeds), x.size))
                            for i, seed in enumerate(seeds):
                                y[i] = results[seed][ld][pe][d][k][
                                    method][metric]
                            # Mean and std of metrics
                            my = y.mean(axis=0)
                            sdy = y.std(axis=0)

                            if method == '1':
                                mname = 'Our method'
                            elif method == '3':
                                mname = 'RABF-log'

                            plt.plot(x, my, next(markers),
                                     label=r'%s' % mname, markersize=3)
                            plt.fill_between(x, my - sdy, my + sdy, alpha=0.2)
                        if metric == 'err_angle':
                            label = r'$\angle(\hat\beta, \beta)$'
                            lim = 1.5
                            plt.yscale('linear')
                        elif metric == 'err_norm':
                            if method == '1':
                                label = r'$||\hat\beta - c_1\beta||$'
                            else:
                                label = r'$c_1||\hat\beta - \beta||$'
                            lim = 2
                            plt.yscale('linear')
                        elif metric == 'kt_dist':
                            label = r'$\tau(\hat\beta, \beta)$'
                            lim = 0.4
                            plt.yscale('linear')
                        else:
                            label = 'Duration'
                            plt.yscale('log')
                            lim = 1000

                        ax.annotate(r'$N$', xy=(.95, 0), xytext=(18, -5),
                                    ha='left', va='top',
                                    xycoords='axes fraction',
                                    textcoords='offset points', fontsize=20)
                        if y_axis:
                            plt.ylabel(label, fontsize=20)
                        plt.xscale('log')
                        plt.ylim(0, lim)
                        plt.grid()
                        if legend:
                            plt.legend(loc='upper right', fontsize=20)
                        plt.tight_layout()
                        plt.savefig(path+'../Syn-mbNcomp-%s-%s-%s.pdf'
                                    % (metric, ld, pe),
                                    format='pdf', transparent=True)
                        plt.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Run synthetic experiments.')
    parser.add_argument('-l', type=int)
    parser.add_argument('-y', type=int)
    args = parser.parse_args()

    home_path = str(Path.home())
    plot_SmbN(home_path + '/Res-Synth/', args.l, args.y)
    plot_SNbd(home_path + '/Res-Synth/', args.l, args.y)
    plot_SNbld(home_path + '/Res-Synth/', args.l, args.y)
    plot_SmbNcomp(home_path + '/Res-Synth/', args.l, args.y)
