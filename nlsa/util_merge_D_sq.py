# -*- coding: utf-8 -*-
import joblib
import numpy
import time

import settings_rho_light as settings

def f(nproc):
    print '\n****** RUNNING merge_D_sq ******'
    results_path = settings.results_path
    datatype = settings.datatype
    
    starttime = time.time()
    
#    fn = '%s/D_sq_loop_idx_0.jbl'%(results_path)
#    test = joblib.load(fn)
#    print test.shape, test.dtype
    
    n = 181438
    
    print 'Merge d_sq'
    #D_sq = numpy.zeros(test.shape, dtype=datatype)
    D_sq = numpy.zeros((n,n), dtype=datatype)
    for i in range(nproc):
        print i
        fn = '%s/D_sq_loop_idx_%d.jbl'%(results_path, i)
        print fn
        temp = joblib.load(fn)
        #D_sq = D_sq + temp
        D_sq += temp
    print 'Done.'
        
    print 'Saving.'
    joblib.dump(D_sq, '%s/D_sq_parallel.jbl'%results_path)
    print 'Done.'
    print 'It took: ', time.time() - starttime