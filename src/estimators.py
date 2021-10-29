"""
Methods for estimating beta from pairwise comparisons
"""
import numpy as np
from sklearn.linear_model import LogisticRegression as logistic_reg
from scipy.optimize import minimize


def averaging(X, XC, yn):
    """
    Estimate covariance from X, beta from XC and yn.
    """
    N, d = X.shape
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
    model = logistic_reg(C=np.inf, fit_intercept=False,
                         solver="lbfgs", max_iter=10000)
    model.fit(XC, yn)
    beta_est = model.coef_[0]
    return beta_est

def RABF_LOG(N, u, v, XC, yn):
    """
    Estimate beta and item scores with the method outlined in the paper
    Rank Aggregation and Prediction with Item Features
    """
    M, d = XC.shape
    
    M_train = int(M*0.6)
    
    u_train = u[:M_train]
    u_valid = u[M_train:]
    v_train = v[:M_train]
    v_valid = v[M_train:]
    XC_train = XC[:M_train]
    XC_valid = XC[M_train:]
    yn_train = yn[:M_train]
    yn_valid = yn[M_train:]

    valid_results = {}
    lambda_w = np.logspace(-6, 3, 4)
    lambda_r = np.logspace(-6, 3, 4)
    
    x = np.random.randn(N+d)

    for lw in lambda_w:
        for lr in lambda_r:
            loss = lambda inp: RABF_loss(inp, d, u_train, v_train, XC_train, yn_train, lw, lr)
            grad = lambda inp: RABF_grad(inp, d, u_train, v_train, XC_train, yn_train, lw, lr)

            res = minimize(loss, x, method='L-BFGS-B', jac=grad)
            x = res.x
            e_beta = x[:d]
            e_scores = x[d:]
            e_yn = np.sign(e_scores[v_valid] - e_scores[u_valid] + XC_valid@e_beta)
            accuracy = np.sum(e_yn == yn_valid)/yn_valid.size
            valid_results[(lw, lr)] = accuracy

    lw, lr = max(valid_results, key=valid_results.get)
    loss = lambda inp: RABF_loss(inp, d, u, v, XC, yn, lw, lr)
    grad = lambda inp: RABF_grad(inp, d, u, v, XC, yn, lw, lr)

    res = minimize(loss, x, method='L-BFGS-B', jac=grad)
    x = res.x
    e_beta = x[:d]
 
    return e_beta

def RABF_loss(x, d, u, v, XC, yn, lw, lr):
    """
    Total loss over dataset
    """
    e_beta = x[:d]
    e_scores = x[d:]
    loss = lw*e_beta@e_beta + lr*e_scores@e_scores
    loss += np.log(1 + np.exp(-yn*(e_scores[v] - e_scores[u] + XC@e_beta))).sum()
    
    return loss

def RABF_grad(x, d, u, v, XC, yn, lw, lr):
    """
    Total grad over dataset
    """
    e_beta = x[:d]
    e_scores = x[d:]

    grad = np.zeros(x.shape)
    grad[:d] += 2*lw*e_beta
    grad[d:] += 2*lr*e_scores

    exp = np.exp(-yn*(e_scores[v]-e_scores[u] + XC@e_beta))
    scale = exp/(1+exp)
    grad[:d] += -((scale*yn)[:, None]*XC).sum(0)
    for i in range(yn.size):
        grad[d + v[i]] += -scale[i]*yn[i]
        grad[d + u[i]] += scale[i]*yn[i]

    return grad

def estimate_beta(X1, u, v, XC, yn, method):
    """
    Choose method and return beta
    """
    if method == 1:
        e_beta = averaging(X1, XC, yn)
    elif method == 2:
        e_beta = logistic(XC, yn)
    elif method == 3:
        e_beta = RABF_LOG(X1.shape[0], u, v, XC, yn)
    
    return e_beta
