# -*- coding: utf-8 -*-
import joblib
import numpy


def f(settings, mode):
    results_path = settings.results_path
    datatype = settings.datatype
    
    x_r_bkw = joblib.load('%s/movie_mode_%d_optimised_chunk_bwd.jbl'%(results_path, mode))
    print x_r_bkw.shape
    
    x_r_fwd = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode))
    print x_r_fwd.shape
    
    m = x_r_bkw.shape[0]
    
    x_r_extended = numpy.zeros((m, x_r_bkw.shape[1]+x_r_fwd.shape[1]), dtype=datatype)
    x_r_extended[:, 0:x_r_bkw.shape[1]] = x_r_bkw
    x_r_extended[:, x_r_bkw.shape[1]:] = x_r_fwd
    
    print x_r_extended.shape, x_r_extended.dtype
    
    joblib.dump(x_r_extended, '%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, mode))