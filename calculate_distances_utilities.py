# -*- coding: utf-8 -*-
import numpy
import joblib
import pickle


def concatenate_backward():
    import util_concatenate_backward
    util_concatenate_backward.f()


    
def calculate_D_N_v_optimised_dense():
    import util_calculate_D_N_v_optimised_dense
    util_calculate_D_N_v_optimised_dense.f()
    
def calculate_D_N_v_optimised_sparse():
    import util_calculate_D_N_v_optimised_sparse
    util_calculate_D_N_v_optimised_sparse.f()


    
def calculate_D_N_v_dense():
    import util_calculate_D_N_v_dense
    util_calculate_D_N_v_dense.f()
    
def calculate_D_N_v_sparse():
    import util_calculate_D_N_v_sparse
    util_calculate_D_N_v_sparse.f()


    
def calculate_d_sq_dense():
    import settings_NP as settings
    import util_calculate_d_sq
    
    results_path = settings.results_path
    f = open('%s/T_anomaly.pkl'%results_path, 'rb')
    #f = open('%s/T.pkl'%results_path, 'rb')
    x = pickle.load(f)
    f.close()
    print 'Data: ', x.shape
    
    mask = numpy.ones(x.shape)       
    
    d_sq = util_calculate_d_sq.f(x, mask)
    print 'd_sq: ', d_sq.shape, d_sq.dtype
    
    print 'Saving d_sq'    
    joblib.dump(d_sq, '%s/d_sq.jbl'%results_path)
    print '\n'
    
def calculate_d_sq_sparse():
    import settings
    import util_calculate_d_sq
    
    results_path = settings.results_path
    sparsity = settings.sparsity
    f = open('%s/T_anomaly_sparse_%.2f.pkl'%(results_path, sparsity), 'rb')
    x = pickle.load(f)
    f.close()
    print 'Data: ', x.shape
    
    mask = numpy.ones(x.shape) 
    mask[numpy.isnan(x)] = 0
    x[numpy.isnan(x)] = 0      
    
    d_sq = util_calculate_d_sq.f(x, mask)
    print 'd_sq: ', d_sq.shape, d_sq.dtype
    
    print 'Saving d_sq'    
    joblib.dump(d_sq, '%s/d_sq.jbl'%results_path)
    print '\n'  
    
def calculate_d_sq_SFX():
    import settings_rho_light as settings
    import util_calculate_d_sq
    
    results_path = settings.results_path
    datatype = settings.datatype
    label = settings.label
       
    # EXTRACT DATA
    T_sparse = joblib.load('../data_rho/data_converted_%s/dT_sparse_%s.jbl'%(label,
                                                                             label))
    print 'T_sparse: ', T_sparse.shape
    print 'T_sparse nonzero: ', T_sparse.count_nonzero()
    x = T_sparse[:,:].todense()
    print 'x: ', x.shape, x.dtype
    x = numpy.asarray(x, dtype=datatype)
    print 'x: ', x.shape, x.dtype
    
    M_sparse = joblib.load('../data_rho/data_converted_%s/M_sparse_%s.jbl'%(label,
                                                                            label))
    print 'M_sparse: ', M_sparse.shape
    print 'M_sparse nonzero: ', M_sparse.count_nonzero()
    mask = M_sparse[:,:].todense()
    print 'mask: ', mask.shape, mask.dtype
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    print 'mask: ', mask.shape, mask.dtype
    
    # CALCULATE d_sq
    d_sq = util_calculate_d_sq.f(x, mask)
    print 'd_sq: ', d_sq.shape, d_sq.dtype
        
    print 'Saving d_sq'    
    joblib.dump(d_sq, '%s/d_sq.jbl'%results_path)
    print '\n' 



    
def calculate_D_sq():  
    import settings_NP as settings
    import util_calculate_D_sq
    
    results_path = settings.results_path
    q = settings.q
    datatype = settings.datatype
    d_sq = joblib.load('%s/d_sq.jbl'%results_path)    
    print 'd_sq: ', d_sq.shape
    
    D_sq = util_calculate_D_sq.f(d_sq, q, datatype)
    print 'D_sq: ', D_sq.shape, D_sq.dtype
    
    print 'Saving D_sq'    
    joblib.dump(D_sq, '%s/D_sq.jbl'%results_path)
    print '\n'



    
def merge_D_sq():
    import util_merge_D_sq
    nproc = 17
    util_merge_D_sq.f(nproc)



    
def sort_D_sq():
    import settings_rho_light as settings
    import util_sort_D_sq
    
    results_path = settings.results_path
    b = settings.b
    D_sq = joblib.load('%s/D_sq_parallel.jbl'%results_path)
    D, N, v = util_sort_D_sq.f(D_sq, b)
    
    print 'Saving'    
    joblib.dump(D, '%s/D.jbl'%results_path)
    joblib.dump(N, '%s/N.jbl'%results_path)
    joblib.dump(v, '%s/v.jbl'%results_path)
    print '\n'