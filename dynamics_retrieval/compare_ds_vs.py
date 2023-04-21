# -*- coding: utf-8 -*-
import pickle
import numpy
import joblib

import settings_NP as settings

results_path = settings.results_path
print results_path

f = open('%s/D_optimised.pkl'%results_path, 'rb')
D_ref = pickle.load(f)
f.close()

#f = open('%s/dist_sparse_algo/D_optimised.pkl'%results_path, 'rb')
#D = pickle.load(f)
#f.close()
#
#diff = D-D_ref
#print numpy.amax(diff), numpy.amin(diff)

D = joblib.load('%s/D.jbl'%results_path)

diff = D-D_ref
print numpy.amax(diff), numpy.amin(diff)


#D = joblib.load('%s/D_parallel.jbl'%results_path)
#diff = D-D_ref
#print numpy.amax(diff), numpy.amin(diff)


f = open('%s/N_optimised.pkl'%results_path, 'rb')
N_ref = pickle.load(f)
f.close()

#f = open('%s/dist_sparse_algo/N_optimised.pkl'%results_path, 'rb')
#N = pickle.load(f)
#f.close()
#
#diff = N-N_ref
#print numpy.amax(diff), numpy.amin(diff)

N = joblib.load('%s/N.jbl'%results_path)

diff = N-N_ref
print numpy.amax(diff), numpy.amin(diff)

#N = joblib.load('%s/N_parallel.jbl'%results_path)
#diff = N-N_ref
#print numpy.amax(diff), numpy.amin(diff)


#f = open('%s/v_optimised.pkl'%results_path, 'rb')
#v_ref = pickle.load(f)
#f.close()
#
#f = open('%s/v_standard.pkl'%results_path, 'rb')
#v_std = pickle.load(f)
#f.close()
#
#v = joblib.load('%s/v.jbl'%results_path)
#v_par = joblib.load('%s/v_parallel.jbl'%results_path)
#
#diff = v.flatten()-v_ref.flatten()
#print numpy.amax(diff), numpy.amin(diff)
#diff = v.flatten()-v_par.flatten()
#print numpy.amax(diff), numpy.amin(diff)
#diff = v.flatten()-v_std.flatten()
#print numpy.amax(diff), numpy.amin(diff)