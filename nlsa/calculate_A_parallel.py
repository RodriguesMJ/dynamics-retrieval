# -*- coding: utf-8 -*-
import getopt
import sys
import numpy
import time
import joblib
#import pickle

import settings_rho_light as settings
 
def f(myArguments):
    
    try:
        optionPairs, leftOver = getopt.getopt(myArguments, "h", ["index="])
    except getopt.GetoptError:
        print 'Usage: python ....py --index <index>'
        sys.exit(2)   
    for option, value in optionPairs:
        if option == '-h':
            print 'Usage: python ....py --index <index>'
            sys.exit()
        elif option == "--index":
            loop_idx = int(value)

    print 'loop_idx: ', loop_idx
    
    q = settings.q    
    datatype = settings.datatype
    results_path = settings.results_path    
    step = settings.paral_step_A 
    nmodes = settings.nmodes
    toproject  = settings.toproject
    label = settings.label
    
    #mu = joblib.load('%s/mu_P_sym.jbl'%(results_path))
    mu = joblib.load('%s/mu.jbl'%(results_path))
    print 'mu (s, ): ', mu.shape
    
    evecs_norm = joblib.load('%s/P_sym_ARPACK_evecs_normalised.jbl'%(results_path))
    print 'evecs_norm (s, s): ', evecs_norm.shape
    Phi = numpy.zeros((evecs_norm.shape[0], nmodes), dtype=datatype)
    for j in range(nmodes):
        ev_toproject = toproject[j]
        print 'Evec: ', ev_toproject
        Phi[:, j] = evecs_norm[:, ev_toproject]
    #Phi = evecs_norm[:,0:nmodes]

#    f = open('%s/T_anomaly.pkl'%results_path,'rb')
#    x = pickle.load(f)
#    f.close()
#    
#    # For sparse data
#    if numpy.any(numpy.isnan(x)):
#        print 'NaN values in x'    
#        x[numpy.isnan(x)] = 0
#        print 'Set X NaNs to zero'
    
    T_sparse = joblib.load('../data_rho/data_converted_%s/T_sparse_%s.jbl'%(label, 
                                                                            label))
    x = T_sparse[:,:].todense()
    print 'x: ', x.shape, x.dtype
    x = numpy.asarray(x, dtype=datatype)
    print 'x (m, S): ', x.shape, x.dtype
       
    m = x.shape[0]
    S = x.shape[1]
    s = S-q
    n = m*q
    
    print 'Calculate A, index: ', loop_idx
    
    q_start = loop_idx * step
    q_end = q_start + step
    if q_end > q:
        q_end = q
    print 'q_start: ', q_start, 'q_end: ', q_end
    
    A = numpy.zeros((n,nmodes), dtype=datatype)
    mu_Phi = numpy.matmul(numpy.diag(mu), Phi)
    print 'mu_Phi (s, nmodes): ', mu_Phi.shape
    starttime = time.time()
    for i in range(q_start, q_end):
        if i%100 == 0:
            print i
        A[i*m : (i+1)*m, :] = numpy.matmul(x[:, q-i : q-i+s], mu_Phi)
            
    print 'Time: ', time.time() - starttime   
    
    print 'A: ', A.shape
    joblib.dump(A, '%s/A_chunck_idx_%d.jbl'%(results_path,loop_idx))    
    print 'Time: ', time.time() - starttime   
    
if __name__== "__main__":
    
    f(sys.argv[1:])