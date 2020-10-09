import numpy as np
from synthetic import generate_comparison_fixed_degree, synthetic_independent_comparisons

d = 100
num_beta = 10
n = int(1e5)
k = 16
m = n / k
error_method = 'BT'
alphas = np.array([5e-8,0.001,0.02,0.03,0.039,0.04,0.041,0.05,0.08,0.095,0.1,0.15,0.18,0.185,0.2,0.3,0.35,0.4,0.41,0.45])
alpha_ave_error = np.concatenate((alphas[:,np.newaxis],np.zeros((alphas.shape[0],1))),axis=1)
error_prev = 2


def BradleyTerryLikelihood(X,label,beta,alpha):
    return 1. / (1. + np.exp(alpha * (-np.multiply(label,X.dot(beta)))))

for count_alpha in range(alphas.shape[0]):
    alpha = alphas[count_alpha]
    error_per_alpha = np.zeros((num_beta,))
    error_per_alpha[:] = np.nan
    error_ave_rate_prev = 2
    for count_beta in range(num_beta):
        synthetic_pairs = generate_comparison_fixed_degree(m, k, count_beta)
        synthetic = synthetic_independent_comparisons(m, k, d, synthetic_pairs,
                                                      seed=count_beta)
        prob_correct_label = BradleyTerryLikelihood(synthetic.pair_feat,synthetic.noiseless_labels,synthetic.beta,alpha)
        # noisy_label, ground_truth_ind_error_labels = synthetic.generate_label(error_method, alpha)
        prob_error_label = 1-prob_correct_label
        error_rate = np.mean(prob_error_label)
        # error_rate = 1.*ground_truth_ind_error_labels.shape[0]/noisy_label.shape[0]
        error_per_alpha[count_beta] = 1.*error_rate
        error_ave_rate_curr = np.nanmean(error_per_alpha)
        if ((error_ave_rate_curr-error_ave_rate_prev)/error_ave_rate_prev)<1e-3:
            break
        else:
            error_ave_rate_prev = 1*error_ave_rate_curr
    alpha_ave_error[count_alpha,1] = 1*error_ave_rate_curr
np.savetxt("../data/BT_prob_error.csv",alpha_ave_error,delimiter=",")
print alpha_ave_error

# P error = [0.4021,0.3011,0.2004,0.1029]
# alpha = [0.042,0.095,0.185,0.41]
print "done"