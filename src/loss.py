"""
Loss functions of interest
"""
import numpy as np
from helpers import get_c1


def beta_error(e_beta, beta, f_cov, method):
    """
    Find the cosine of angle between e_beta and beta
    Estimate c1 by the approximation of sigmoid integral
    Calculate ||e_beta - c1*beta||
    """
    # Get norms of beta
    eb_norm = np.linalg.norm(e_beta)
    b_norm = np.linalg.norm(beta)
    # Find the angle
    err_angle = np.arccos(e_beta/eb_norm @ beta/b_norm)
    # If averaging
    if method == 1:
        # Get estimated c1
        e_c1 = get_c1(beta, f_cov)
        err_beta = np.linalg.norm(e_beta - e_c1*beta)
    # else logistic
    else:
        err_beta = np.linalg.norm(e_beta - beta)

    return err_angle, err_beta


def kt_distance(scores, e_scores):
    """
    Kendall Tau distance
    Z pairs exist where for a pair i, j is
    true score of i < true score of j, for such i, j
    sum count of estimated score of i > estimated score of j
    then divide by Z
    """
    N = len(scores)

    num = 0
    denum = 0
    for i in range(N):
        for j in range(i+1, N):
            if scores[i] < scores[j]:
                denum += 1
                if e_scores[i] > e_scores[j]:
                    num += 1

    return num/denum
