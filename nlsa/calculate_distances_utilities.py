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
    
    #results_path = settings.results_path
    #datatype = settings.datatype
    label = settings.label
       
    # EXTRACT DATA
    #T_sparse = joblib.load('../data_rho/data_converted_%s/dT_sparse_%s.jbl'%(label,
    #                                                                         label))
#    T_sparse = joblib.load('/das/work/p18/p18594/cecilia-offline/NLSA/data_rho/data_converted_%s_nS214205/T_sparse_%s.jbl'
#                           %(label, label))
    #T_sparse = joblib.load('/das/work/p18/p18594/cecilia-offline/NLSA/data_rho/data_converted_%s/T_sparse_%s.jbl'
    #                        %(label, label))
    
    #
    #T_sparse = T_sparse[0:70000, 0:10]
    #
    
#    print 'T_sparse: ', T_sparse.shape, T_sparse.dtype
#    print 'T_sparse nonzero: ', T_sparse.count_nonzero()
    
#    x = T_sparse[:,:].todense()
#    print 'x: ', x.shape, x.dtype
#    x = numpy.asarray(x, dtype=datatype)
#    print 'x: ', x.shape, x.dtype
    
    #M_sparse = joblib.load('../data_rho/data_converted_%s/M_sparse_%s.jbl'%(label,
    #                                                                        label))
#    M_sparse = joblib.load('/das/work/p18/p18594/cecilia-offline/NLSA/data_rho/data_converted_%s_nS214205/M_sparse_%s.jbl'
#                           %(label, label))
    #M_sparse = joblib.load('/das/work/p18/p18594/cecilia-offline/NLSA/data_rho/data_converted_%s/M_sparse_%s.jbl'
    #                       %(label, label))
    
    #
    #M_sparse = M_sparse[0:70000, 0:10]
    #
    
#    print 'M_sparse: ', M_sparse.shape, M_sparse.dtype
#    print 'M_sparse nonzero: ', M_sparse.count_nonzero()
    
#    mask = M_sparse[:,:].todense()
#    print 'mask: ', mask.shape, mask.dtype
#    mask = numpy.asarray(mask, dtype=numpy.uint8)
#    print 'mask: ', mask.shape, mask.dtype
    
#    # CALCULATE d_sq SPARSE INPUT
#    d_sq_sparse = util_calculate_d_sq.f_sparse(T_sparse, M_sparse)
#    print 'd_sq_sparse: ', d_sq_sparse.shape, d_sq_sparse.dtype
#        
#    print 'Saving d_sq_sparse'    
#    joblib.dump(d_sq_sparse, '%s/d_sq_sparseinput.jbl'%results_path)
#    print '\n' 
    
    
    # CALCULATE TERM BY TERM
    #util_calculate_d_sq.f_sparse_x_T_x(T_sparse)
    #util_calculate_d_sq.f_sparse_x_sq_T_mask(T_sparse, M_sparse)
    #util_calculate_d_sq.f_sparse_mask_T_x_sq(T_sparse, M_sparse)
    #util_calculate_d_sq.f_add_1()
    util_calculate_d_sq.f_add_2()
    
#    term_xTx = util_calculate_d_sq.f_sparse_x_T_x(T_sparse)
#    print 'Saving x_T_x'    
#    joblib.dump(term_xTx, '%s/term_xTx.jbl'%results_path)
    
    
     # CALCULATE d_sq DENSE INPUT  
#    d_sq_dense = util_calculate_d_sq.f_dense(x, mask)
#    print 'd_sq_dense: ', d_sq_dense.shape, d_sq_dense.dtype
#        
#    print 'Saving d_sq_dense'    
#    joblib.dump(d_sq_dense, '%s/d_sq_dense.jbl'%results_path)
#    print '\n'     
#        
#    ### CHECK
#    #d_sq_denseinput = joblib.load('%s/d_sq.jbl'%results_path)
#    diff = abs(d_sq_dense-d_sq_sparse)
#    print 'Difference sparse to dense input: ', numpy.amax(diff)
#    print d_sq_sparse
#    print d_sq_dense
#    print diff
#    numpy.fill_diagonal(d_sq_sparse, 1)
#    diff = diff/d_sq_sparse
#    print 'Difference sparse to dense input (fraction): ', numpy.nanmax(diff)
#    
#    d_sq_dense_old = joblib.load('%s/d_sq_bkp/d_sq.jbl'%results_path)
#    diff = abs(d_sq_dense-d_sq_dense_old)
#    print 'Difference dense input (new/old): ', numpy.amax(diff)
#    numpy.fill_diagonal(d_sq_dense_old, 1)
#    diff = diff/d_sq_dense_old
#    print 'Difference old/new dense input (fraction): ', numpy.nanmax(diff)
#    ### CHECK



    
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
    nproc = 5
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