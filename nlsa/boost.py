# -*- coding: utf-8 -*-
import scipy.io
import joblib
import h5py
import numpy
from scipy import sparse

def main(settings):

    datatype = settings.datatype
    label = settings.label
    folder = settings.results_path
    print datatype, label
    print folder
    
    # LIGHT
    # T = joblib.load( '%s/T_sel_sparse_%s.jbl'%(folder, label))
    # M = joblib.load( '%s/M_sel_sparse_%s.jbl'%(folder, label))
    # DARK
    T = joblib.load( '%s/T_sparse_%s.jbl'%(folder, label))
    M = joblib.load( '%s/M_sparse_%s.jbl'%(folder, label))
    
    print sparse.issparse(T), T.dtype, T.shape
    
    n_obs = M.sum(axis=1)
    print sparse.issparse(n_obs), n_obs.dtype, n_obs.shape
    
    T_bst = T / n_obs
    T_bst = 1000 * T_bst
    
    print sparse.issparse(T_bst), T_bst.dtype, T_bst.shape
       
    print T[11, 0:100]
    print n_obs[11, 0]
    print T_bst[11, 0:100]
    
    T_bst_sparse = sparse.csr_matrix(T_bst)
    joblib.dump(T_bst_sparse, '%s/T_bst_sparse_%s.jbl'%(folder, label))
      
    print sparse.issparse(T_bst_sparse), T_bst_sparse.dtype, T_bst_sparse.shape
    print sparse.issparse(n_obs), n_obs.dtype, n_obs.shape
    n_obs = numpy.asarray(n_obs, dtype=numpy.float)
    joblib.dump(n_obs, '%s/boost_factors_%s.jbl'%(folder, label))
    
        