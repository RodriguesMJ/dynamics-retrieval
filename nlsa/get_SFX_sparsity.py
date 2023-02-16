# -*- coding: utf-8 -*-
import joblib
import random
import numpy

label = 'light'
root_folder = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2'
results_path = '%s/results_LPSA/bR_%s_dI'%(root_folder, label)
S = 119507
m = 22727

n_obs = joblib.load('%s/boost_factors_%s.jbl'%(results_path, label))
sparsity_array = n_obs/S

idxs = random.sample(range(0, m), 7000)
idxs = sorted(idxs)

sparsity_array_sampled = sparsity_array[idxs]

thrs = 1-sparsity_array_sampled

print max(thrs)
print numpy.average(sparsity_array)
print numpy.average(sparsity_array_sampled)

joblib.dump(thrs, '/das/work/p17/p17491/Cecilia_Casadei/NLSA/synthetic_data_jitter/test12/sparsity_thrs.jbl')