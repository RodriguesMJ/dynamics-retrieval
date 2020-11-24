# -*- coding: utf-8 -*-
import numpy

def f(x, mask):
    print '\n****** RUNNING calculate_d_sq ******'
    m = x.shape[0]    
    S = x.shape[1]

    print m, ' pixels'
    print S, ' samples'
    
    print 'Calculate d^2 (sparse data in general)'
    
    print 'Start'

    x_sq = numpy.multiply(x,x)

    d_sq = numpy.matmul(x_sq.T, mask) \
         + numpy.matmul(mask.T, x_sq) \
         - 2 * numpy.matmul(x.T, x)

    print 'd_sq: ', d_sq.shape
    test = numpy.argwhere(d_sq < 0)
    print '\n'
    print test.shape[0], 'negative values in d_sq'
    test = numpy.argwhere(numpy.diag(d_sq) < 0)
    print test.shape[0], 'negative values in diag(d_sq)'
    
    print 'd_sq min value: ', numpy.amin(d_sq)
    print 'd_sq max value: ', numpy.amax(d_sq)

    print '\nSet diag d_sq values to zero'
    numpy.fill_diagonal(d_sq, 0)
    
    test = numpy.argwhere(d_sq < 0)
    print test.shape[0], 'negative values in d_sq'
    test = numpy.argwhere(numpy.diag(d_sq) < 0)
    print test.shape[0], 'negative values in diag(d_sq)'
    
    print 'd_sq min value: ', numpy.amin(d_sq)
    print 'd_sq max value: ', numpy.amax(d_sq)
    
    print '\nSet negative values to zero.'
    d_sq[d_sq < 0] = 0
    
    diff = d_sq - d_sq.T
    print 'd_sq is symmetric:'
    print numpy.amin(diff), numpy.amax(diff)
    print 'd_sq calculation done.'
    
    return d_sq