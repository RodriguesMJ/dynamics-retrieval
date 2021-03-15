# -*- coding: utf-8 -*-
import h5py
import numpy

import joblib
import settings_bR_dark as settings

folder = settings.results_path

D = joblib.load('%s/D_parallel.jbl'%folder)
N = joblib.load('%s/N_parallel.jbl'%folder)

file_mat = '%s/dataY_dark_nS99616_nN5000_iB1.mat'%folder
f = h5py.File(file_mat, 'r') 

yInd = f['yInd']
yVal = f['yVal']
yInd = numpy.asarray(yInd)
yInd  = yInd.T
yInd = yInd.reshape(N.shape)
yInd = yInd - 1
diff = abs(yInd-N)
test = numpy.argwhere(diff > 0)
print 'Number of index disagreements: ', test.shape[0]
print 'Fraction index disagreement: ', float(test.shape[0])/(N.shape[0]*N.shape[1])

yVal = numpy.asarray(yVal)
yVal  = yVal.T
yVal = numpy.sqrt(yVal)
yVal = yVal.reshape(D.shape)

diff_D = yVal - D
print 'Differences in D in range: ', numpy.amax(diff_D), numpy.amin(diff_D)
diff_D_rel = (yVal[:,1:] - D[:,1:]) / yVal[:,1:]
print 'Relative differences in D in range: ', numpy.amax(diff_D_rel), numpy.amin(diff_D_rel)

print '\nFirst 10 index disagreements:'
for i in range(10):
    i0 = test[i,0]
    i1 = test[i,1]
    print '%10d %10d %12.4f %12.4f'%(yInd[i0, i1], N[i0,i1], yVal[i0,i1], D[i0,i1])

# SMALL DIFFERENCEIN DISTANCE CALCULATION ALTER THE ORDER OF ABOUT 1% SORTED DISTANCES




#f = open('%s/T_sparse_light.pkl'%folder, 'rb')
#T_sparse = pickle.load(f)
#f.close()
#print T_sparse.count_nonzero()
#T_dense = T_sparse.todense()
#print T_dense.shape
#
#f = open('%s/M_sparse_light.pkl'%folder, 'rb')
#M_sparse = pickle.load(f)
#f.close()
#print M_sparse.count_nonzero()
#M_dense = M_sparse.todense()
#print M_dense.shape
#
#M_dense[T_dense==0]=0
#M_sparse = sparse.csr_matrix(M_dense)
#print M_sparse.count_nonzero()