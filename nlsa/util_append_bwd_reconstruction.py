# -*- coding: utf-8 -*-
import joblib
import numpy

import settings_rho_light as settings


def f(mode):
    results_path = settings.results_path
    x_r_bkw = joblib.load('%s/movie_mode_%d_optimised_chunk_bwd.jbl'%(results_path, mode))
    print x_r_bkw.shape
    
    x_r_fwd = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode))
    print x_r_fwd.shape
    
    m = x_r_bkw.shape[0]
    
    x_r_extended = numpy.zeros((m, x_r_bkw.shape[1]+x_r_fwd.shape[1]))
    x_r_extended[:, 0:x_r_bkw.shape[1]] = x_r_bkw
    x_r_extended[:, x_r_bkw.shape[1]:] = x_r_fwd
    
    print x_r_extended.shape
    
    joblib.dump(x_r_extended, '%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, mode))
    
#if __name__ == '__main__':
#    for i in range(4): 
#        print 'Mode', i
#        f(i)