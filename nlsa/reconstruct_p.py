# -*- coding: utf-8 -*-

import numpy
import time
import joblib
 
def f(settings):
    results_path = settings.results_path
    q = settings.q
    p = settings.p
    print results_path
    
    datatype = settings.datatype
    
    U = joblib.load('%s/U.jbl'%results_path)
    if U.shape[1] > 20:
        U = U[:,0:20]
    print 'U:', U.shape
    S = joblib.load('%s/S.jbl'%results_path)
    print 'S: ', S.shape
    VT_final = joblib.load('%s/VT_final.jbl'%results_path)
    print 'VT_final: ', VT_final.shape
        
    for k in settings.modes_to_reconstruct:
        print 'Mode: ', k    
        
        s = VT_final.shape[1]
        print 's: ', s
        m = U.shape[0]/q
        print 'm: ', m
        
        u_k = U[:,k]
        s_k = S[k]
        v_k = VT_final[k,:]
        print u_k.shape, v_k.shape
        
        block_idx_center = (q-1)/2
        print 'block_idx_center:', block_idx_center
        block_idxs = range(int(block_idx_center)-p, int(block_idx_center)+p+1)
        print block_idxs
        
        starttime = time.time()
        
        ncopies = 2*p+1
        x_r = numpy.zeros((m,s-ncopies+1), dtype=datatype)
        print 'x_r:', x_r.shape
        
        for i in range(-p, p+1):
            
            block_idx = block_idx_center + i
            print i, block_idx
            i1 =  block_idx*m
            i2 = (block_idx+1)*m
            term = s_k * numpy.outer(u_k[i1:i2], v_k)
            print 'term: ', term.shape
            print 'starting column:', i+p
            print 'ending column (included):', i+p+(s-ncopies+1)-1
            
            x_r += term[:,i+p:i+p+(s-ncopies+1)]
        x_r = x_r/ncopies
        joblib.dump(x_r, '%s/movie_p_%d_mode_%d.jbl'%(results_path, p, k))
        
        print 'Time: ', time.time() - starttime

def f_ts(settings):        
    S = settings.S
    q = settings.q  
    s = S-q+1
    
    ts_meas = joblib.load('%s/ts_meas.jbl'%settings.results_path)
    
    ts_svs = []
    for i in range(s):
        t_sv = numpy.average(ts_meas[i:i+q])
        ts_svs.append(t_sv)
        
    ts_svs = numpy.asarray(ts_svs)
    #t = numpy.asarray(range(s))
    print 'ts_meas:', ts_meas.shape, ts_meas[0:10]
    print 'ts_svs:',  ts_svs.shape,  ts_svs[0:10]
    
    p = settings.p
    
    block_idx_center = (q-1)/2
    print 'block_idx_center:', block_idx_center
    block_idxs = range(int(block_idx_center)-p, int(block_idx_center)+p+1)
    print block_idxs
    
    ncopies = 2*p+1
    t_r = numpy.zeros((s-ncopies+1,), dtype=settings.datatype)
    print 't_r:', t_r.shape
        
    for i in range(-p, p+1):
        print i
        
        print 'starting column:', i+p
        print 'ending column (included):', i+p+(s-ncopies+1)-1
        
        t_r += ts_svs[i+p:i+p+(s-ncopies+1)]
    t_r = t_r/ncopies
    joblib.dump(t_r, '%s/t_r_p_%d.jbl'%(settings.results_path, p))
    
    print 't_r:',  t_r.shape,  t_r[0:10], '...', t_r[-1]
    
        



