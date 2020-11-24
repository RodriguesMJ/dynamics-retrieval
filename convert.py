# -*- coding: utf-8 -*-
import scipy.io
import joblib
import h5py
import numpy
from scipy import sparse

import settings_rho_light as settings

datatype = settings.datatype
label = settings.label

folder = '../data_rho/data_converted_%s'%label

file_mat = '%s/miller_idxs_%s.mat'%(folder, label)
dictionary = scipy.io.loadmat(file_mat)
h_miller = dictionary['miller_h']
k_miller = dictionary['miller_k']
l_miller = dictionary['miller_l']

joblib.dump(h_miller, '%s/miller_h_%s.jbl'%(folder, label))
joblib.dump(k_miller, '%s/miller_k_%s.jbl'%(folder, label))
joblib.dump(l_miller, '%s/miller_l_%s.jbl'%(folder, label))

file_mat = '%s/t_uniform_light.mat'%folder
dictionary = scipy.io.loadmat(file_mat)
t_uniform = dictionary['t_uniform']

joblib.dump(t_uniform, '%s/t_uniform_light.jbl'%folder)

file_mat = '%s/T_%s_full.mat'%(folder, label)
f = h5py.File(file_mat, 'r') 
T_full = f['/T']
T_full = numpy.asarray(T_full, dtype=datatype)
print 'T_full: ', T_full.shape, T_full.dtype
print 'Make sparse'
T_sparse = sparse.csr_matrix(T_full)
joblib.dump(T_sparse, '%s/T_sparse_%s.jbl'%(folder, label))
print 'T_sparse: ', T_sparse.shape, T_sparse.dtype
print 'T_sparse nonzero: ', T_sparse.count_nonzero()
print 'T_sparse done'

T_avg = numpy.average(T_full, axis=1)
print 'T_avg: ', T_avg.shape, T_avg.dtype
joblib.dump(T_avg, '%s/T_avg_%s.jbl'%(folder, label))
T_avg = numpy.repeat(T_avg[:,numpy.newaxis], T_full.shape[1], axis=1)
print 'T_avg: ', T_avg.shape, T_avg.dtype
dT_full = T_full - T_avg

file_mat = '%s/M_%s_full.mat'%(folder, label)
f = h5py.File(file_mat, 'r') 
M_full = f['/M']
M_full = numpy.asarray(M_full, dtype=numpy.uint8)
print 'M_full: ', M_full.shape, M_full.dtype

dT_full = dT_full * M_full
print 'dT_full: ', dT_full.shape, dT_full.dtype
dT_sparse = sparse.csr_matrix(dT_full)
print 'dT_sparse: ', dT_sparse.shape, dT_sparse.dtype
joblib.dump(dT_sparse, '%s/dT_sparse_%s.jbl'%(folder, label))

print 'Make sparse' 
M_sparse = sparse.csr_matrix(M_full)
joblib.dump(M_sparse, '%s/M_sparse_%s.jbl'%(folder, label))
print 'M_sparse: ', M_sparse.shape, M_sparse.dtype
print 'M_sparse nonzero: ', M_sparse.count_nonzero()
print 'M_sparse done'