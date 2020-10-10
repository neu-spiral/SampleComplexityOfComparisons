"""
Helper codes
"""
from os import path
import pickle
from pathlib import Path
import numpy as np


def check_exp(args):
    """
    If experiment is already finished, stop early.
    """
    file_path = get_exp_path(args)
    if not path.exists(file_path):
        return
    else:
        raise Exception('%s exists.' % file_path)


def save_results(results, args):
    """
    Save results dict in disk.
    """
    # Get file path to write
    file_path = get_exp_path(args)
    # Write to disk
    with open(file_path, 'wb+') as f:
        pickle.dump(dict(results), f)


def get_exp_path(args):
    """
    Generate file path for current experiment.
    """
    # Find path to home dir
    home_path = str(Path.home())
    file_path = home_path + '/SCResults/%i-%.2f-%i-%i-%i-%i-%i' \
        % (args.seed, args.ld, args.d, args.N1, args.N2, args.k, args.method)

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
    and smallest eigen value is 0.
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
    N1 = np.log10(N1)
    N2 = np.log10(N2)
    N = np.ceil(np.logspace(N1, N2, 10)).astype(np.int32)

    if k == 1:
        M = N
    elif k == 2:
        M = np.ceil(N*np.log(np.log(N))).astype(np.int32)
    elif k == 3:
        M = np.ceil(N*np.log(N)).astype(np.int32)
    elif k == 4:
        M = np.ceil(N*N**.5).astype(np.int32)

    return N, M
