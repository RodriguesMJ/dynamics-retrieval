#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:35:30 2022

@author: casadei_c
"""

import matplotlib.pyplot
import joblib
import numpy

results_path = '../../synthetic_data_4/test6'

flag = 1
if flag == 1:
    CCs_SVD = joblib.load('%s/svd/reconstruction_CC_vs_nmodes.jbl'%results_path)
    CCs_fourier = joblib.load('%s/fourier/reconstruction_CC_vs_nmodes.jbl'%results_path)
    #CCs_nlsa_t = joblib.load('%s/nlsa/b_1500_time_nns/reconstruction_CC_vs_nmodes.jbl'%results_path)
    CCs_nlsa_t = joblib.load('%s/nlsa/distances_onlymeasured_normalised/b_time_nns/reconstruction_CC_vs_nmodes.jbl'%results_path)
    
    Nmodes = len(CCs_nlsa_t)+1
    
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_SVD, 'o-', c='b', label='SVD')
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_fourier, 'o-', c='m', label='Fourier (q=4000)')
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_t, 'x-', c='c', label='NLSA (q=4000, 1500 time nns)')
    matplotlib.pyplot.legend(frameon=False)
    matplotlib.pyplot.xticks(range(1,Nmodes,2))
    matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_nmodes_sparsepartial.png'%results_path, dpi=96*3)
    matplotlib.pyplot.close()
    
    # CCs_nlsa_eu = joblib.load('%s/nlsa/b_eu_nns/reconstruction_CC_vs_nmodes.jbl'%results_path)
    # #CCs_nlsa_eu = joblib.load('%s/nlsa/distances_onlymeasured_normalised/b_eu_nns/reconstruction_CC_vs_nmodes.jbl'%results_path)
    
    # matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_t, 'o-', c='b', label='NLSA (time nns)')
    # matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_eu, 'o-', c='m', label='NLSA (Euclidean nns)')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.xticks(range(1,Nmodes,2))
    # matplotlib.pyplot.savefig('%s/reconstruction_NLSA_CC_vs_nmodes.png'%results_path, dpi=96*3)
    # matplotlib.pyplot.close()
    
    # CCs_nlsa_t = numpy.asarray(CCs_nlsa_t)
    # CCs_nlsa_eu = numpy.asarray(CCs_nlsa_eu)
    
    # avg = (CCs_nlsa_t+CCs_nlsa_eu) / 2
    # matplotlib.pyplot.plot(range(1,Nmodes), (CCs_nlsa_t-CCs_nlsa_eu)/avg, 'o', c='b')
    # matplotlib.pyplot.xticks(range(1,Nmodes,2))
    # matplotlib.pyplot.savefig('%s/reconstruction_NLSA_CC_vs_nmodes_relativedelta.png'%results_path, dpi=96*3)
    # matplotlib.pyplot.close()

flag = 0
if flag == 1:
    CCs_nlsa_eu_1500 = joblib.load('%s/nlsa/b_1500_eu_nns/reconstruction_CC_vs_nmodes.jbl'%results_path)
    CCs_nlsa_eu_3000 = joblib.load('%s/nlsa/b_3000_eu_nns/reconstruction_CC_vs_nmodes.jbl'%results_path)
    CCs_nlsa_eu_4000_6p5 = joblib.load('%s/nlsa/b_4000_eu_nns/logeps_6p5/reconstruction_CC_vs_nmodes.jbl'%results_path)
    CCs_nlsa_eu_4000_6p6 = joblib.load('%s/nlsa/b_4000_eu_nns/logeps_6p6/reconstruction_CC_vs_nmodes.jbl'%results_path)
    CCs_nlsa_eu_4000_7p2 = joblib.load('%s/nlsa/b_4000_eu_nns/logeps_7p2/reconstruction_CC_vs_nmodes.jbl'%results_path)
    CCs_nlsa_eu_4000_6p0 = joblib.load('%s/nlsa/b_4000_eu_nns/logeps_6p0/reconstruction_CC_vs_nmodes.jbl'%results_path)
    CCs_nlsa_t_1500 = joblib.load('%s/nlsa/b_1500_time_nns/reconstruction_CC_vs_nmodes.jbl'%results_path)
    Nmodes = len(CCs_nlsa_eu_1500)+1
    matplotlib.pyplot.figure(figsize=(15,15))
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_eu_1500, 'o-', c='b', label='NLSA (1500 Euclidean nns)')
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_eu_3000, 's-', c='m', label='NLSA (3000 Euclidean nns)')
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_eu_4000_6p5, 'x-', c='y', label='NLSA (4000 Euclidean nns, logEps=6.5)')
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_eu_4000_7p2, 'x-', c='k', label='NLSA (4000 Euclidean nns, logEps=7.2)')
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_eu_4000_6p0, 'x-', c='g', label='NLSA (4000 Euclidean nns, logEps=6.0)')
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_eu_4000_6p6, 'x-', c='b', label='NLSA (4000 Euclidean nns, logEps=6.6)')
    matplotlib.pyplot.plot(range(1,Nmodes), CCs_nlsa_t_1500, '*-', c='c', label='NLSA (1500 time nns)')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xticks(range(1,Nmodes,2))
    matplotlib.pyplot.savefig('%s/reconstruction_NLSA_eu_CC_vs_nmodes.png'%results_path, dpi=96*3)
    matplotlib.pyplot.close()
    
flag = 0
if flag == 1:
    qs = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    n_qs = len(qs)
    import matplotlib.pylab
    colors = matplotlib.pylab.cm.Blues(numpy.linspace(0,1,n_qs))   
    matplotlib.pyplot.figure(figsize=(10,10))  
    n = 12   
    matplotlib.pyplot.xticks(range(1,n+1,2))   
    matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
    for i, q in enumerate(qs):
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        CCs = joblib.load('../../synthetic_data_4/test5/fourier_para_search/f_max_100_q_%d/reconstruction_CC_vs_nmodes.jbl'%q)
        matplotlib.pyplot.plot(range(1, len(CCs)+1), CCs, '-o', c=colors[i], label='q=%d'%q)  
    matplotlib.pyplot.legend(frameon=False)
    matplotlib.pyplot.savefig('../../synthetic_data_4/reconstruction_CC_vs_nmodes_fourier_benchmark_q_scan.png', dpi=96*4)
    matplotlib.pyplot.close()
