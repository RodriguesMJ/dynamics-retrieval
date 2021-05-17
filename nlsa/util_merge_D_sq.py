# -*- coding: utf-8 -*-
import joblib
import numpy
import time

def f(settings):
    print '\n****** RUNNING merge_D_sq ******'
    results_path = settings.results_path
    datatype = settings.datatype
    nproc = settings.n_workers
    n = settings.n
    
    starttime = time.time()
    
    # fn = '%s/D_sq_loop_idx_0.jbl'%(results_path)
    # D_sq = joblib.load(fn)
    # print D_sq.shape, D_sq.dtype
    # n = D_sq.shape[0]
    
    print 'Merge D_sq'
    D_sq = numpy.zeros((n,n), dtype=datatype)
    for i in range(nproc):
        print i
        fn = '%s/D_sq_loop_idx_%d.jbl'%(results_path, i)
        print fn
        temp = joblib.load(fn)
        D_sq += temp
    print 'Done.'
        
    print 'Saving.'
    joblib.dump(D_sq, '%s/D_sq_parallel.jbl'%results_path)
    print 'Done.'
    print 'It took: ', time.time() - starttime