# -*- coding: utf-8 -*-
import numpy
import pickle
import random

import settings

print '\n\n\n'
print '****** RUNNING MAKE SPARSE ******'
print '\n\n\n'
    
results_path = settings.results_path
sparsity = settings.sparsity

f = open('%s/T_anomaly.pkl'%results_path, 'rb')
T_full = pickle.load(f)
f.close()

m = T_full.shape[0]
S = T_full.shape[1]

n_sparse_per_frame = int(sparsity * m)

T_sparse = numpy.zeros((m,S))

for i in range(S):
    frame = T_full[:,i]
    NaN_idxs = random.sample(range(m), n_sparse_per_frame)
    frame[NaN_idxs] = numpy.NaN
    T_sparse[:,i] = frame
    
f = open('%s/T_anomaly_sparse_%.2f.pkl'%(results_path, sparsity), 'wb')
pickle.dump(T_sparse, f)
f.close()