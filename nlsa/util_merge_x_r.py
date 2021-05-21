# -*- coding: utf-8 -*-
import joblib
import numpy
import time

def f(settings, mode):   
    print '\n****** RUNNING merge_x_r ******'
    results_path = settings.results_path
    datatype = settings.datatype
    nproc = settings.n_workers_reconstruction 
    
    print 'Mode: ', mode
    starttime = time.time()
    
    fn = '%s/movie_mode_%d_chunck_0.jbl'%(results_path, mode)   
    test = joblib.load(fn)
    print test.shape, test.dtype
    
    print 'Merge x_r'
    x_r = numpy.zeros(test.shape, dtype=datatype)
    for i in range(nproc):
        print i
        fn = '%s/movie_mode_%d_chunck_%d.jbl'%(results_path, mode, i)
        print fn
        temp = joblib.load(fn)
        x_r = x_r + temp
    print 'Done.'
        
    print 'Saving.'
    joblib.dump(x_r, '%s/movie_mode_%d_parallel.jbl'%(results_path, mode))
    print 'Done.'
    print 'It took: ', time.time() - starttime