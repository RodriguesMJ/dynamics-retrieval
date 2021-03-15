# -*- coding: utf-8 -*-
import joblib
import numpy
import scipy.linalg

import settings_rho_light as settings

def density_normalisation(W_sym, datatype):    
    s = W_sym.shape[0]
    Q = numpy.sum(W_sym, axis=0)
    W_norm = numpy.zeros((s,s), dtype=datatype)
    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        Q_i = Q[i]
        for j in range(s):
            if abs(W_sym[i,j]) > 0:
                Q_j = Q[j]
                W_norm[i,j] = W_sym[i,j]/(Q_i*Q_j)
    return W_norm
        

def density_normalisation_opt(W_sym):
    s = W_sym.shape[0]
    Q = numpy.sum(W_sym, axis=0)
    Q = Q[:, numpy.newaxis]
    Q_rep = numpy.repeat(Q, s, axis=1)
    
    norm_factor = numpy.multiply(Q_rep, Q_rep.T)
    W_norm_opt = W_sym/norm_factor
    
    return W_norm_opt
 
    
def row_normalisation(W_norm):   
    s = W_norm.shape[0]
    Q = numpy.sum(W_norm, axis=1)
    P = numpy.zeros((s,s))
    for i in range(s):
        if i%1000 == 0:
            print i, '/', s
        Q_i = Q[i]
        P[i,:] = W_norm[i,:]/Q_i
    return P, Q


def row_normalisation_opt(W_norm):   
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


def symmetrize_P(P, Q):
    sqrt_Q = numpy.sqrt(Q)
    inv_sqrt_Q = 1.0/sqrt_Q
    P_sym_temp = numpy.matmul(P, numpy.diag(inv_sqrt_Q))
    P_sym = numpy.matmul(numpy.diag(sqrt_Q), P_sym_temp)
    return P_sym


def get_mu_P_sym(P_sym):    
    evals_P_sym_left, evecs_P_sym_left = scipy.linalg.eig(P_sym, 
                                                          left=True, 
                                                          right=False)
    mu_P_sym = evecs_P_sym_left[:,0]
    return evals_P_sym_left, mu_P_sym


if __name__ == '__main__':
        
    results_path = settings.results_path
    datatype = settings.datatype
    
#    print 'Load W_sym'
#    W_sym = joblib.load('%s/W_sym.jbl'%results_path)
#    
#    print 'Set NaN values to zero'
#    W_sym[numpy.isnan(W_sym)] = 0

#    print 'Check that W_sym is symmetric:'
#    diff = W_sym - W_sym.T
#    print numpy.nanmax(diff), numpy.nanmin(diff)
    
#    print 'Density normalization'
#    W_norm = density_normalisation(W_sym, datatype)    
#    print 'Check that W_norm is symmetric'            
#    diff = W_norm - W_norm.T
#    print numpy.amax(diff), numpy.amin(diff)       
#    
    #print 'Density normalization (optimised)'
    #W_norm_opt = density_normalisation_opt(W_sym)
#    print 'Check that W_norm is symmetric'            
#    diff = W_norm_opt - W_norm_opt.T
#    print numpy.amax(diff), numpy.amin(diff)   
#    print 'Check results'      
#    diff = W_norm - W_norm_opt
#    print numpy.amax(diff), numpy.amin(diff)
#   
#    print 'Row normalisation'
#    P, Q = row_normalisation(W_norm)
    
    #print 'Row normalisation (optimised)'
    #P, Q = row_normalisation_opt(W_norm_opt)
    #print 'P: ', P.shape, P.dtype
    #print 'Q: ', Q.shape, Q.dtype
#    print 'Check results'      
#    diff = P - P_opt
#    print numpy.amax(diff), numpy.amin(diff)
        
    #joblib.dump(P, '%s/P.jbl'%results_path)
    #joblib.dump(Q, '%s/Q.jbl'%results_path)
    P = joblib.load('%s/P.jbl'%results_path)
    Q = joblib.load('%s/Q.jbl'%results_path)
    
#    print 'Calculate mu_P'
#    mu_P = get_mu_P(Q)
#    
#    joblib.dump(mu_P, '%s/mu_P.jbl'%results_path)
#
#    print 'Check mu_P P = mu_P'
#    mu_P_T = mu_P.T
#    dot = numpy.dot(mu_P_T, P)
#    diff = dot - mu_P_T
#    print numpy.amax(diff), numpy.amin(diff)

    print 'Symmetrise P'
    P_sym = symmetrize_P(P, Q)

    joblib.dump(P_sym, '%s/P_sym.jbl'%results_path)
    
#    print 'Check that P_sym is symmetric'            
#    diff = P_sym - P_sym.T
#    print numpy.amax(diff), numpy.amin(diff)    

    #P_sym = joblib.load('%s/P_sym.jbl'%results_path)
    
#    print 'Calculate mu_P_sym'    
#    evals_P_sym_left, mu_P_sym = get_mu_P_sym(P_sym)
#
#    joblib.dump(evals_P_sym_left, '%s/evals_P_sym_left.jbl'%results_path)
#    joblib.dump(mu_P_sym, '%s/mu_P_sym.jbl'%results_path)
#    
#    print 'Check mu_P_sym P_sym = mu_P_sym'
#    mu_P_sym_T = mu_P_sym.T
#    dot = numpy.dot(mu_P_sym_T, P_sym)
#    diff = dot - mu_P_sym_T
#    print numpy.amax(diff), numpy.amin(diff)