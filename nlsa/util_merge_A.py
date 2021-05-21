# -*- coding: utf-8 -*-
import joblib
import numpy
import time


def main(settings):

    print '\n****** RUNNING merge_A ******'
    results_path = settings.results_path
    datatype = settings.datatype
    n_workers_A = settings.n_workers_A
    
    starttime = time.time()
    
    fn = '%s/A_chunck_idx_0.jbl'%(results_path)
    A = joblib.load(fn)
    print A.shape, A.dtype
    nrows = A.shape[0]
    ncols = A.shape[1]
    
    print 'Merge A'
    A = numpy.zeros((nrows, ncols), dtype=datatype)
    for i in range(n_workers_A):
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