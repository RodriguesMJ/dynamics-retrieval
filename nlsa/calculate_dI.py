#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:27:33 2023
@author: casadei_c
"""

from scipy import sparse
import joblib
import numpy

def main(settings):
    
    T = joblib.load('%s/T_sel_sparse_%s.jbl'%(settings.results_path, settings.label))
    M = joblib.load('%s/M_sel_sparse_%s.jbl'%(settings.results_path, settings.label))
    print 'T: is sparse: ', sparse.issparse(T), T.shape, T.dtype
    print 'M: is sparse: ', sparse.issparse(M), M.shape, M.dtype
    print 'T[100,58]: ', T[100,58], 'T[100,57]: ', T[100,57]
    ns = numpy.sum(M, axis=1)
    avgs = numpy.sum(T, axis=1)/ns
    print 'avgs: ', avgs.shape
    print 'avgs[100,0]:', avgs[100,0]
    T = T-avgs
    print 'After avg subtraction: '
    print 'T[100,58]: ', T[100,58], 'T[100,57]: ', T[100,57]
    print 'T: is sparse: ', sparse.issparse(T), T.shape, T.dtype
    for i in range(M.shape[1]):
        if i%500 == 0:
            print i
        T[:,i] = numpy.multiply(T[:,i], (M[:,i].todense()))
    print 'After mask multiplication: '
    print 'T[100,58]: ', T[100,58], 'T[100,57]: ', T[100,57]
    print 'T: is sparse: ', sparse.issparse(T), T.shape, T.dtype
    dT_sparse = sparse.csr_matrix(T)
    print 'dT_sparse: is sparse: ', sparse.issparse(dT_sparse), dT_sparse.shape, dT_sparse.dtype
    joblib.dump(dT_sparse, '%s/dT_sparse_%s.jbl'%(settings.results_path, settings.label))
