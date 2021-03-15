# -*- coding: utf-8 -*-
import numpy
import time

def f_unoptimised(D_sq, b):
    print '\n****** RUNNING sort_D_sq ******'
    
    starttime = time.time()
    
    print 'D_sq: ', D_sq.shape, D_sq.dtype
    
    diff = D_sq - D_sq.T
    print 'D_sq is symmetric:'
    print numpy.amin(diff), numpy.amax(diff)    
    
    print 'Extract velocities.'
    v = numpy.sqrt(numpy.diag(D_sq, 1))
    
    print 'Remove first sample.'
    D_sq = D_sq[1:, 1:]    
    print 'D_sq: ', D_sq.shape, D_sq.dtype    
    
    print 'Sqrt of D_sq.'
    D_all = numpy.sqrt(D_sq)
    
    print 'Sorting: '
    idxs = numpy.argsort(D_all, axis=1)
    D_all_sorted = numpy.sort(D_all, axis=1)
    
    D = D_all_sorted[:, 0:b]
    N = idxs[:, 0:b]
     
    D[:,0] = 0   
    
    print 'It took: ', time.time() - starttime
    
    return D, N, v


def f(D, b): # It is D_sq
    print '\n****** RUNNING sort_D_sq ******'
    
    starttime = time.time()
    
    print 'D_sq: ', D.shape, D.dtype
    
#    diff = D_sq - D_sq.T
#    print 'D_sq is symmetric:'
#    print numpy.amin(diff), numpy.amax(diff)    
#    
#    print 'Extract velocities.'
#    v = numpy.sqrt(numpy.diag(D_sq, 1))
    
    print 'Remove first sample.'
    D = D[1:, 1:]    
    print 'D_sq: ', D.shape, D.dtype    
    
    print 'Sqrt of D_sq.'
    D = numpy.sqrt(D)
    
    print 'Sorting: '
    N = numpy.argsort(D, axis=1)
    N = N[:, 0:b]
    
    D = numpy.sort(D, axis=1)    
    D = D[:, 0:b]
         
    D[:,0] = 0   
    
    print 'It took: ', time.time() - starttime
    
    return D, N#, v