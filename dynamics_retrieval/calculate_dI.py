#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:27:33 2023
@author: casadei_c
"""

import joblib
import numpy
from scipy import sparse


def main(settings):
    T_fn = "%s/T_sparse_LTD_%s.jbl" % (settings.results_path, settings.label)  # TO LOAD
    M_fn = "%s/M_sparse_%s.jbl" % (settings.results_path, settings.label)  # TO LOAD
    dT_fn = "%s/dT_sparse_LTD_%s.jbl" % (
        settings.results_path,
        settings.label,
    )  # TO SAVE

    T = joblib.load(T_fn)
    M = joblib.load(M_fn)
    print "T: is sparse: ", sparse.issparse(T), T.shape, T.dtype
    print "M: is sparse: ", sparse.issparse(M), M.shape, M.dtype

    if sparse.issparse(T) == False:
        T = sparse.csr_matrix(T)
    print "T, is sparse:", sparse.issparse(T), T.dtype, T.shape
    if sparse.issparse(M) == False:
        M = sparse.csr_matrix(M)
    print "M, is sparse:", sparse.issparse(M), M.dtype, M.shape

    ns = numpy.sum(M, axis=1)
    print "ns: ", ns.shape, ns.dtype
    avgs = numpy.sum(T, axis=1) / ns
    print "avgs: ", avgs.shape
    print "avgs[100,0]:", avgs[100, 0]
    print "T[100,0:100]: ", T[100, 0:100]
    T = T - avgs
    for i in range(M.shape[1]):
        if i % 500 == 0:
            print i
        T[:, i] = numpy.multiply(T[:, i], (M[:, i].todense()))
    dT_sparse = sparse.csr_matrix(T)
    print "After mask multiplication: "
    print "dT_sparse: is sparse: ", sparse.issparse(
        dT_sparse
    ), dT_sparse.shape, dT_sparse.dtype
    print "T[100,0:100]: ", T[100, 0:100]
    joblib.dump(dT_sparse, dT_fn)
