# -*- coding: utf-8 -*-
import argparse
import importlib
import numpy
import time
import joblib

 
def f(loop_idx, settings):
        
    print 'loop_idx: ', loop_idx
    
    q = settings.q    
    datatype = settings.datatype
    results_path = settings.results_path    
    step = settings.paral_step_A 
    nmodes = settings.nmodes
    toproject = settings.toproject
    data_file = settings.data_file
    
    f_max_considered = settings.f_max_considered
    filterfuncs = joblib.load('%s/F_on_qr.jbl'%results_path)
    filterfuncs = filterfuncs[:,:(2*f_max_considered+1)]
    print 'Filterfuncs:', filterfuncs.shape

    
    mu = joblib.load('%s/mu.jbl'%(results_path))
    print 'mu (s, ): ', mu.shape
    
    evecs_norm = joblib.load('%s/P_evecs_normalised.jbl'%(results_path))
    print 'evecs_norm (s, s): ', evecs_norm.shape
    Phi = numpy.zeros((evecs_norm.shape[0], nmodes), dtype=datatype)
    for j in range(nmodes):
        ev_toproject = toproject[j]
        print 'Evec: ', ev_toproject
        Phi[:, j] = evecs_norm[:, ev_toproject]
    
#    f = open('%s/T_anomaly.pkl'%results_path,'rb')
#    x = pickle.load(f)
#    f.close()
#    
#    # For sparse data
#    if numpy.any(numpy.isnan(x)):
#        print 'NaN values in x'    
#        x[numpy.isnan(x)] = 0
#        print 'Set X NaNs to zero'
    try:
        T_sparse = joblib.load(data_file)
        x = T_sparse[:,:].todense()
        print 'x (sparse -> dense): ', x.shape, x.dtype
        x = numpy.asarray(x, dtype=datatype)
        print 'x (m, S): ', x.shape, x.dtype
    except:
        x = joblib.load(data_file)
        print 'x (dense): ', x.shape, x.dtype
        
       
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
    
    temp = numpy.matmul(filterfuncs.T, mu_Phi)
    filtered_mu_Phi = numpy.matmul(filterfuncs, temp)
    print 'filtered_mu_Phi (s, nmodes): ', mu_Phi.shape
    
    starttime = time.time()
    for i in range(q_start, q_end):
        if i%100 == 0:
            print i
        A[i*m : (i+1)*m, :] = numpy.matmul(x[:, q-i : q-i+s], filtered_mu_Phi)
            
    print 'Time: ', time.time() - starttime   
    
    print 'A: ', A.shape
    joblib.dump(A, '%s/A_chunck_idx_%d.jbl'%(results_path,loop_idx))    
    print 'Time: ', time.time() - starttime   
    
def main(args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("worker_ID", help="Worker ID")
    parser.add_argument("settings", help="settings file")
    args = parser.parse_args(args)

    print 'Worker ID: ', args.worker_ID
    
    # Dynamic import based on the command line argument
    settings = importlib.import_module(args.settings)

    #print("Label: %s"%settings.label)
    
    f(int(args.worker_ID), settings)


if __name__ == "__main__":
    main()