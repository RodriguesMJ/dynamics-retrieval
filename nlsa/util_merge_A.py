# -*- coding: utf-8 -*-
import joblib
import numpy
import time

import settings_rho_light as settings

def f(nproc):

    print '\n****** RUNNING merge_A ******'
    results_path = settings.results_path
    datatype = settings.datatype
    
    starttime = time.time()
    
    fn = '%s/A_chunck_idx_0.jbl'%(results_path)
    test = joblib.load(fn)
    print test.shape, test.dtype
    
    print 'Merge A'
    A = numpy.zeros(test.shape, dtype=datatype)
    for i in range(nproc):
        print i
        fn = '%s/A_chunck_idx_%d.jbl'%(results_path, i)
        print fn
        temp = joblib.load(fn)
        A = A + temp
    print 'Done.'
     
    print 'A: ', A.shape, A.dtype
    print 'Saving.'
    joblib.dump(A, '%s/A_parallel.jbl'%results_path)
    print 'Done.'
    print 'It took: ', time.time() - starttime
    
#    A_loop = joblib.load('%s/A_loop.jbl'%results_path)
#    diff = A - A_loop
#    print numpy.amax(diff), numpy.amin(diff)
#    
#    A_chuncks = joblib.load('%s/A_chuncks.jbl'%results_path)
#    diff = A - A_chuncks
#    print numpy.amax(diff), numpy.amin(diff)
#    
#    diff = A_loop - A_chuncks
#    print numpy.amax(diff), numpy.amin(diff)