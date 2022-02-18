#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:45:10 2022

@author: casadei_c
"""
import numpy

def local_linearity_measure(x):
    approx_l = x[:, 0:-3] - \
               (   x[:, 1:-2] - 1*(x[:,2:-1]-x[:,1:-2])  )
    #print 'approx_l:', approx_l.shape
    approx_r = x[:, 3:]   - \
               (   x[:, 1:-2] + 2*(x[:,2:-1]-x[:,1:-2])  )
    #print 'approx_r:', approx_r.shape
    L = 0.5*(abs(approx_l) + abs(approx_r))
    #print L.shape
    L = numpy.average(L)
    print L
    return L