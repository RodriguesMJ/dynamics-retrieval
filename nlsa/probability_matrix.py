# -*- coding: utf-8 -*-
import joblib
import numpy
import scipy.linalg
import time


def density_normalisation_memopt(W_sym, datatype):    
    s = W_sym.shape[0]
    #Q = numpy.sum(W_sym, axis=0)
    # CHANGE 07/10/2021
    Q = numpy.sum(W_sym, axis=1)
    print 'Q', Q.shape
    W_norm = numpy.zeros((s,s), dtype=datatype)
    start = time.time()
    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        Q_i = Q[i]
        #print 'Q_i', Q_i
        for j in range(s):
            if abs(W_sym[i,j]) > 0:
                Q_j = Q[j]
                #print 'Q_i, Q_j, Q_i*Q_j', Q_i, Q_j, Q_i*Q_j
                W_norm[i,j] = W_sym[i,j]/(Q_i*Q_j)
    print 'It took: ', time.time()-start
    return W_norm
        

def density_normalisation_timeopt(W_sym):
    s = W_sym.shape[0]
    Q = numpy.sum(W_sym, axis=0)
    Q = Q[:, numpy.newaxis]
    Q_rep = numpy.repeat(Q, s, axis=1)
    
    norm_factor = numpy.multiply(Q_rep, Q_rep.T)
    W_norm_opt = W_sym/norm_factor    
    
    return W_norm_opt
    
 
    
def row_normalisation_memopt(W_norm, datatype):   
    s = W_norm.shape[0]
    Q = numpy.sum(W_norm, axis=1)
    P = numpy.zeros((s,s), dtype=datatype)
    print 'P: ', P.shape, P.dtype
    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        Q_i = Q[i]
        P[i,:] = W_norm[i,:]/Q_i
    return P, Q

def get_W_tilde_memopt(W_norm, datatype):   
    s = W_norm.shape[0]
    sqrt_Q = numpy.sqrt(numpy.sum(W_norm, axis=1))
    W_tilde = numpy.zeros((s,s), dtype=datatype)
    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        sqrt_Q_i = sqrt_Q[i]
        W_tilde[i,:] = W_norm[i,:]/sqrt_Q_i
    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        sqrt_Q_i = sqrt_Q[i]
        W_tilde[:,i] = W_tilde[:,i]/sqrt_Q_i
    return W_tilde



def row_normalisation_timeopt(W_norm):   
    s = W_norm.shape[0]
    Q = numpy.sum(W_norm, axis=1)
    Q_temp = Q[:, numpy.newaxis]
    Q_rep = numpy.repeat(Q_temp, s, axis=1)
    P = W_norm/Q_rep
    return P, Q
    

def get_mu_P(Q):
    Q_sum = Q.sum()
    mu = Q/Q_sum 
    return mu

    
def symmetrize_P_timeopt(P, Q):
    print 'P: ', P.shape, P.dtype
    print 'Q: ', Q.shape, Q.dtype
    sqrt_Q = numpy.sqrt(Q)
    inv_sqrt_Q = 1.0/sqrt_Q
    P_sym_temp = numpy.matmul(P, numpy.diag(inv_sqrt_Q))
    P_sym = numpy.matmul(numpy.diag(sqrt_Q), P_sym_temp)
    return P_sym
    

def symmetrize_P_memopt(P, Q):
    print 'P: ', P.shape, P.dtype
    print 'Q: ', Q.shape, Q.dtype
    sqrt_Q = numpy.sqrt(Q)
    inv_sqrt_Q = 1.0/sqrt_Q
    s = P.shape[0]
    start = time.time()
    for i in range(s):
        P[:,i] = inv_sqrt_Q[i]*P[:,i]
        if i%10000 == 0:
            print i, '/', s
    print 'First loop done'
    for i in range(s):
        P[i,:] = sqrt_Q[i]*P[i,:]
        if i%10000 == 0:
            print i, '/', s
    print 'Second loop done'
    print 'It took: ', time.time()-start
    return P



def get_mu_P_sym(P_sym):    
    evals_P_sym_left, evecs_P_sym_left = scipy.linalg.eig(P_sym, 
                                                          left=True, 
                                                          right=False)
    mu_P_sym = evecs_P_sym_left[:,0]
    return evals_P_sym_left, mu_P_sym


def main(settings):
        
    results_path = settings.results_path
    datatype = settings.datatype
    
    flag_W_norm = 1
    if flag_W_norm == 1:
        print 'Load W_sym'
        W_sym = joblib.load('%s/W_sym.jbl'%results_path) 
        print 'Density normalization (low memory use, long running time)'
        W_norm = density_normalisation_memopt(W_sym, datatype) 
        joblib.dump(W_norm, '%s/W_norm.jbl'%results_path)
    
    flag_check_W_norm = 1
    if flag_check_W_norm == 1:
        W_norm = joblib.load('%s/W_norm.jbl'%results_path)        
        print 'Check that W_norm is symmetric'            
        diff = W_norm - W_norm.T
        print numpy.amax(diff), numpy.amin(diff)     
        
    flag_W_tilde = 1
    if flag_W_tilde == 1:
        W_norm = joblib.load('%s/W_norm.jbl'%results_path)
        print 'Get W_tilde'
        W_tilde = get_W_tilde_memopt(W_norm, datatype)        
        joblib.dump(W_tilde, '%s/W_tilde.jbl'%results_path)
        
    flag_check_W_tilde = 1
    if flag_check_W_tilde == 1:
        W_tilde = joblib.load('%s/W_tilde.jbl'%results_path)        
        print 'Check that W_tilde is symmetric'            
        diff = W_tilde - W_tilde.T
        print numpy.amax(diff), numpy.amin(diff)     
        
    
