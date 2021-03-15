# -*- coding: utf-8 -*-
import joblib
import numpy

import settings_rho_light as settings

def calculate_W_localspeeds(D, N, v, results_path, datatype, sigma_sq):
    s = D.shape[0]
    b = D.shape[1]    
    print s, b
    
    W = numpy.empty((s,s), dtype=datatype)
    W[:] = numpy.nan
    
    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        v_i = v[i]
        for j in range(b):
            d_i_neighbor = D[i,j]
            neighbor_idx = N[i,j]
            v_neighbor = v[neighbor_idx] 
            W[i, neighbor_idx] = numpy.exp(-(d_i_neighbor*d_i_neighbor)/
                                            (sigma_sq*v_i*v_neighbor))
    
    return W


def calculate_W(D, N, results_path, datatype, sigma_sq):
    s = D.shape[0]
    b = D.shape[1]    
    print s, b, sigma_sq
    
    W = numpy.empty((s,s), dtype=datatype)
    W[:] = numpy.nan
    
    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        for j in range(b):
            d_i_neighbor = D[i,j]
            neighbor_idx = N[i,j]
            W[i, neighbor_idx] = numpy.exp(-(d_i_neighbor*d_i_neighbor)/
                                            (sigma_sq))
    
    return W
    


def symmetrise_W(W, dataype):
    s = W.shape[0]
    W_sym = numpy.empty((s,s), dtype=datatype)
    W_sym[:] = numpy.nan

    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        for j in range(s):
            if not numpy.isnan(W[i,j]): 
                W_sym[i,j] = W[i,j]
            elif not numpy.isnan(W[j,i]):
                W_sym[i,j] = W[j,i]
    return W_sym



def symmetrise_W_optimised(W, dataype):
    s = W.shape[0]
    W_sym = numpy.empty((s,s), dtype=datatype)
    W_sym[:] = numpy.nan
    
    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        row_i = W[i,:].flatten()
        col_i = W[:,i].flatten()
        m_row_i = numpy.ones((s,))
        m_col_i = numpy.ones((s,))
        
        m_row_i[numpy.isnan(row_i)]=0
        m_col_i[numpy.isnan(col_i)]=0
        
        m_i = m_row_i + m_col_i
        
        row_i[numpy.isnan(row_i)]=0
        col_i[numpy.isnan(col_i)]=0
        
        sym_i = (row_i+col_i)/m_i # Divisions by zero will be nan
        
        W_sym[i,:] = sym_i
        W_sym[:,i] = sym_i

    return W_sym
    
    

if __name__ == '__main__':
    
    print '\n****** RUNNING tansition_matrix ****** '
    
    sigma_sq = settings.sigma_sq
    print 'Sigma_sq: ', sigma_sq
    epsilon = sigma_sq/2
    print 'Epsilon: ', epsilon
        
    results_path = settings.results_path
    datatype = settings.datatype
    
#    N = joblib.load('%s/N.jbl'%results_path)
#    D = joblib.load('%s/D.jbl'%results_path)
#    v = joblib.load('%s/v.jbl'%results_path)
#    
#    print 'Calculate W (local speeds)'
#    W = calculate_W_localspeeds(D, N, v, results_path, datatype, sigma_sq)
#    joblib.dump(W, '%s/W.jbl'%results_path)
         
#    print 'Calculate W'
#    W = calculate_W(D, N, results_path, datatype, sigma_sq)
#    joblib.dump(W, '%s/W.jbl'%results_path)

    W = joblib.load('%s/W.jbl'%results_path)

#    print 'Symmetrise W'
#    W_sym = symmetrise_W(W, datatype)
#    joblib.dump(W_sym, '%s/W_sym.jbl'%results_path)
    
    print 'Symmetrise W (optimised)'
    W_sym = symmetrise_W_optimised(W, datatype)
    joblib.dump(W_sym, '%s/W_sym.jbl'%results_path)

#    W_sym_opt = joblib.load('%s/W_sym_opt.jbl'%results_path)
    
#    print 'Checks:'
#    
#    W[numpy.isnan(W)] = 0
#    W_sym[numpy.isnan(W_sym)] = 0
#    W_sym_opt[numpy.isnan(W_sym_opt)] = 0
#    
#    print 'Check that W is not symmetric:'
#    W_T = W.T
#    diff = W-W_T
#    print numpy.amax(diff), numpy.amin(diff)
#        
#    print 'Check that W_sym is symmetric:'
#    W_sym_T = W_sym.T
#    diff = W_sym - W_sym_T
#    print numpy.amax(diff), numpy.amin(diff)
#    
#    print 'Check W_sym_opt is symmetric '
#    W_sym_opt_T = W_sym_opt.T    
#    diff = W_sym_opt - W_sym_opt_T    
#    print numpy.amax(diff), numpy.amin(diff)
#    
#    print 'Check that W_sym = W_sym_opt'
#    diff = W_sym - W_sym_opt
#    print numpy.amax(diff), numpy.amin(diff)
    
    
#    import pickle
#    
#    print 'Check W-W_ref'
#    f = open('../data_Giannakis_PNAS_2012/results_test_8/W.pkl', 'rb')
#    W_ref = pickle.load(f)
#    f.close()
#    
#    W_ref[numpy.isnan(W_ref)] = 0    
#    diff = W_ref - W
#    print numpy.amax(diff), numpy.amin(diff)
#
#    print 'Check W_sym_opt - W_sym_ref'
#    f = open('../data_Giannakis_PNAS_2012/results_test_8/W_sym.pkl', 'rb')
#    W_sym_ref = pickle.load(f)
#    f.close()
#     
#    W_sym_ref[numpy.isnan(W_sym_ref)] = 0
#    diff = W_sym_ref - W_sym_opt
#    print numpy.amax(diff), numpy.amin(diff)   