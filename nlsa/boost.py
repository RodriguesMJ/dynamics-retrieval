# -*- coding: utf-8 -*-
import joblib
import numpy
from scipy import sparse

def main(settings):

    datatype = settings.datatype
    label = settings.label
    folder = settings.results_path
    print datatype, label
    print folder
    
    # LIGHT
    if label == 'light':
        #T = joblib.load( '%s/T_sel_sparse_%s.jbl'%(folder, label))
        T = joblib.load( '%s/dT_sparse_%s.jbl'%(folder, label))
        M = joblib.load( '%s/M_sel_sparse_%s.jbl'%(folder, label))
    # DARK
    # if label == 'dark':
    #     T = joblib.load( '%s/T_sparse_%s.jbl'%(folder, label))
    #     M = joblib.load( '%s/M_sparse_%s.jbl'%(folder, label))
    
    print 'T: is sparse: ', sparse.issparse(T), T.dtype, T.shape
    print 'M: is sparse: ', sparse.issparse(M), M.dtype, M.shape
    n_obs = M.sum(axis=1)
    print 'n_obs: is sparse: ', sparse.issparse(n_obs), n_obs.dtype, n_obs.shape
    
    T_bst = T / n_obs
    T_bst = settings.S * T_bst
    
    print 'T_bst: is sparse: ', sparse.issparse(T_bst), T_bst.dtype, T_bst.shape
       
    print T[11, 0:100]
    print n_obs[11, 0]
    print T_bst[11, 0:100]
    
    T_bst_sparse = sparse.csr_matrix(T_bst)
    joblib.dump(T_bst_sparse, '%s/dT_bst_sparse_%s.jbl'%(folder, label))
      
    print 'dT_bst_sparse: is sparse: ',sparse.issparse(T_bst_sparse), T_bst_sparse.dtype, T_bst_sparse.shape
    
        
def main_syn_data(settings):

    datatype = settings.datatype
    folder = settings.results_path
    print folder
    
    # Syn data
    T = joblib.load( '%s/x.jbl'%(folder))
    M = joblib.load( '%s/mask.jbl'%(folder))
    
    print sparse.issparse(T), T.dtype, T.shape
    if sparse.issparse(T) == False:
        print 'T is not sparse.'
        T = sparse.csr_matrix(T)
        print 'T, is sparse:', sparse.issparse(T), T.dtype, T.shape
    if sparse.issparse(M) == False:
        print 'M is not sparse.'
        M = sparse.csr_matrix(M)
        print 'M, is sparse:', sparse.issparse(M), M.dtype, M.shape    
    
    n_obs = M.sum(axis=1)
    print 'n_obs, is sparse: ', sparse.issparse(n_obs), n_obs.dtype, n_obs.shape
    
    T_bst = T / n_obs
    T_bst = 1000 * T_bst
    
    print 'T_bst: is sparse: ', sparse.issparse(T_bst), T_bst.dtype, T_bst.shape
       
    print T[22, 0:100]
    print n_obs[22, 0]
    print T_bst[22, 0:100]
    
    T_bst_sparse = sparse.csr_matrix(T_bst)
    joblib.dump(T_bst_sparse, '%s/T_bst_sparse.jbl'%(folder))
    print 'T_bst_sparse: is sparse: ',sparse.issparse(T_bst_sparse), T_bst_sparse.dtype, T_bst_sparse.shape
    print 'n_obs: is sparse: ', sparse.issparse(n_obs), n_obs.dtype, n_obs.shape
    
    n_obs = numpy.asarray(n_obs, dtype=datatype)
    joblib.dump(n_obs, '%s/boost_factors.jbl'%(folder))    