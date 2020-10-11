"""
Methods for estimating beta from pairwise comparisons
"""
import numpy as np
from sklearn.linear_model import LogisticRegression as logistic_reg


def averaging(X, XC, yn):
    """
    Estimate covariance from X, beta from XC and yn.
    """
    N = X.shape[0]
    _, d = XC.shape
    # Estimated mean
    e_mean = X.mean(axis=0)
    # Estimated covariance, scaling for unbiased precision
    X -= e_mean
    e_cov = X.T@X/(N-d-2)
    # Estimated precision
    e_prec = np.linalg.inv(e_cov)
    # Estimated beta
    e_beta = e_prec @ (yn[:, None]*XC).mean(axis=0)

    return e_beta


def logistic(XC, yn):
    """
    Estimate beta by logistic regression over comparison labels
    """
    # C infty implies no regularization
    model = logistic_reg(C=np.inf, fit_intercept=False, solver="lbfgs")
    model.fit(XC, yn)
    beta_est = model.coef_[0]
    return beta_est


def estimate_beta(X, XC, yn, method):
    """
    Choose method and return beta
    """
    if method == 1:
        e_beta = averaging(X, XC, yn)
    elif method == 2:
        e_beta = logistic(XC, yn)

    return e_beta
