# -*- coding: utf-8 -*-
import h5py
import numpy
import matplotlib.pyplot
import joblib

import settings_bR_light as settings

folder = settings.results_path

U = joblib.load('%s/U.jbl'%folder)
S = joblib.load('%s/S.jbl'%folder)
VT = joblib.load('%s/VT_final.jbl'%folder)

print VT.shape

file_mat = '../../SVD_light_nN5000_26300.0000_chronosVer1.mat'
f = h5py.File(file_mat, 'r') 

print f.keys()
U_m = f['U']
S_m = f['S']
VT_m = f['V']
U_m = numpy.asarray(U_m).T
S_m = numpy.diag(numpy.asarray(S_m))
VT_m = numpy.asarray(VT_m)

U_test = U[0:U_m.shape[0], :]

nmodes = VT.shape[0]
s = VT.shape[1]

print 'nmodes: ', nmodes
print 's: ', s

for i in range(nmodes):
    chrono = VT[i,:]
    chrono_m = VT_m[i,:]
    matplotlib.pyplot.figure(figsize=(30,10))
    matplotlib.pyplot.plot(range(s), chrono, 'bo', markersize=10)
    matplotlib.pyplot.plot(range(s), chrono_m, 'mo', markersize=3)
    matplotlib.pyplot.savefig('%s/chronos/cmp_matlab_py_chrono_%d.png'%(folder, i), dpi=2*96)
    matplotlib.pyplot.close()