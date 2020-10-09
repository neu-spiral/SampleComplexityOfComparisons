import pickle
import numpy as np
import os

rop_result_dir = "../result/ROP/"
algorithms = ['greedyCycle','greedyCycleBeta','No Correction','Repeated MLE']
correct_method = 'flip'
estimate_method = 'MLE'
k_fold = 5


num_alg = len(algorithms)
auc_RSD_plus = np.zeros((num_alg,k_fold))
auc_RSD_plus[:] = np.nan
auc_RSD_prep = np.zeros((num_alg,k_fold))
auc_RSD_prep[:] = np.nan
auc_comparison = np.zeros((num_alg,k_fold))
auc_comparison[:] = np.nan

for ind_alg in range(num_alg):
    for fold in range(k_fold):
        file_name = "ROP_"+algorithms[ind_alg]+"_"+estimate_method+"_fold_"+str(fold)+"_"+correct_method+".p"
        full_file_name = rop_result_dir+algorithms[ind_alg] +"/"+file_name
        if os.path.isfile(full_file_name):
            # out_dict = pickle.load(open(full_file_name,'rb'))
            with open(full_file_name, "rb") as f:
                out_dict = pickle.load(f)
            auc_RSD_plus[ind_alg,fold] = 1* out_dict['auc_class_plus']
            auc_RSD_prep[ind_alg,fold] = out_dict['auc_class_normal']
            auc_comparison[ind_alg,fold] = out_dict['auc_cmp']


print("class_plus", np.mean(auc_RSD_plusaxis=1))
print("class_prep", np.mean(auc_RSD_plusaxis=1))
print("comparison", np.mean(auc_RSD_plusaxis=1))

        # print algorithms[ind_alg] + "-auc_plus: "+np.mean(auc_RSD_plus[ind_alg,:])

print "done"

# ROP_No Correction_MLE_fold_0_flip.p
