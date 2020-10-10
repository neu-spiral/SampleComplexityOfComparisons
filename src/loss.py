"""
Loss functions of interest
"""
import numpy as np
from src.helpers import get_c1


def beta_error(e_beta, beta, f_cov):
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
    # Get estimated c1
    e_c1 = get_c1(beta, f_cov)
    err_beta = np.linalg.norm(e_beta - e_c1*beta)

    return err_angle, err_beta
