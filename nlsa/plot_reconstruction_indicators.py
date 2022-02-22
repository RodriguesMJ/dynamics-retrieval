#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:35:30 2022

@author: casadei_c
"""

import matplotlib.pyplot
import matplotlib.pylab
import joblib
import numpy

results_path = '../../synthetic_data_4/test6'

flag = 0
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

def makelabel(test_number):
    if test_number == 5:
        l = 'benchmark'
    elif test_number == 6:
        l = 'sparsepartial'
    else:
        l = 'undefined'
    return l

root_f = '../../synthetic_data_4'

#### q-scan, CC ####    
flag = 1
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5, 6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        f_max = 100
        qs = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]
        n_curves = len(qs)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves)) 
        
        matplotlib.pyplot.figure(figsize=(10,10))  
        matplotlib.pyplot.xticks(range(1,n_m+1,2))           
        matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.ylim(bottom=0.64, top=1.02)
        matplotlib.pyplot.xlabel('n. modes', fontsize=20)
        matplotlib.pyplot.ylabel('CC', fontsize=20)
        
        for i, q in enumerate(qs):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            CCs = joblib.load('%s/reconstruction_CC_vs_nmodes.jbl'%folder)
            matplotlib.pyplot.plot(range(1, len(CCs)+1), CCs, '-o', c=colors[i], label='q=%d'%q)  
            
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=18)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/reconstruct_CC_vs_nmodes_fourier_%s_q_scan_fmax_%d.png'%(root_f, label, f_max), dpi=96*4)
        matplotlib.pyplot.close()

#### jmax-scan, CC ####
flag = 1
if flag == 1:
    matplotlib.pyplot.style.use('classic')  
    for test_n in [5,6]:
    
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        q = 4000
        f_max_s = [1, 5, 10, 50, 100, 150, 300]
        n_curves = len(f_max_s)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))      
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.ylim(bottom=0.64, top=1.02)
        matplotlib.pyplot.xlabel('n. modes', fontsize=20)
        matplotlib.pyplot.ylabel('CC', fontsize=20)
        
        for i, f_max in enumerate(f_max_s):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            CCs = joblib.load('%s/reconstruction_CC_vs_nmodes.jbl'%folder)
            matplotlib.pyplot.plot(range(1, len(CCs)+1), CCs, '-o', c=colors[i], label='$j_{\mathrm{max}}=$%d'%f_max)  
        
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=18)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/reconstruct_CC_vs_nmodes_fourier_%s_fmax_scan_q_%d.png'%(root_f, label, q), dpi=96*4)
        matplotlib.pyplot.close()

#### q-scan, L ####    
flag = 1
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5, 6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        f_max = 100
        qs = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]
        n_curves = len(qs)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))          
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('n. modes', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(L)$', fontsize=20)
        
        for i, q in enumerate(qs):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            lls = joblib.load('%s/local_linearity_vs_nmodes.jbl'%folder)
            matplotlib.pyplot.plot(range(1, len(lls)+1), numpy.log10(lls), '-o', c=colors[i], label='q=%d'%q)  
        
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=18)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/reconstruct_L_vs_nmodes_fourier_%s_q_scan_fmax_%d.png'%(root_f, label, f_max), dpi=96*4)
        matplotlib.pyplot.close()
        
#### jmax-scan, L ####    
flag = 1
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5,6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        q = 4000
        f_max_s = [1, 5, 10, 50, 100, 150, 300]
        n_curves = len(f_max_s)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))  
        
        matplotlib.pyplot.figure(figsize=(10,10))  
        matplotlib.pyplot.xticks(range(1,n_m+1,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)       
        matplotlib.pyplot.xlabel('n. modes', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(L)$', fontsize=20)
        
        for i, f_max in enumerate(f_max_s):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            lls = joblib.load('%s/local_linearity_vs_nmodes.jbl'%folder)
            matplotlib.pyplot.plot(range(1, len(lls)+1), numpy.log10(lls), '-o', c=colors[i], label='$j_{\mathrm{max}}=$%d'%f_max)  
        
        matplotlib.pyplot.legend(frameon=False, loc='lower right', fontsize=18)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/reconstruct_L_vs_nmodes_fourier_%s_fmax_scan_q_%d.png'%(root_f, label, q), dpi=96*4)
        matplotlib.pyplot.close()


#### q-scan, SVs ####        
flag = 1
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5, 6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        f_max = 100
        qs = [1, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]
        n_curves = len(qs)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))  
        matplotlib.pyplot.xticks(range(0,n_m,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('mode', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(\sigma/\sigma_0)$', fontsize=20)
        
        for i, q in enumerate(qs):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            S = joblib.load('%s/S.jbl'%folder)
            matplotlib.pyplot.plot(range(0, n_m), numpy.log10(S/S[0])[0:n_m], '-o', c=colors[i], label='q=%d'%q)  
        
        matplotlib.pyplot.legend(frameon=False, loc='upper right', fontsize=18)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/SVs_fourier_%s_q_scan_fmax_%d.png'%(root_f, label, f_max), dpi=96*4)
        matplotlib.pyplot.close()
        

#### jmax-scan, SVs ####        
flag = 1
if flag == 1:
    matplotlib.pyplot.style.use('classic') 
    for test_n in [5,6]:
        
        label = makelabel(test_n)
        test_f = '%s/test%d/fourier_para_search'%(root_f, test_n)
        
        n_m = 20
        q = 4000
        f_max_s = [1, 5, 10, 50, 100, 150, 300]
        n_curves = len(f_max_s)
        
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        
        matplotlib.pyplot.figure(figsize=(10,10))  
        matplotlib.pyplot.xticks(range(0,n_m,2))   
        matplotlib.pyplot.xlim(left=0, right=n_m+1)
        matplotlib.pyplot.xlabel('mode', fontsize=20)
        matplotlib.pyplot.ylabel('log$_{10}(\sigma/\sigma_0)$', fontsize=20)
        
        for i, f_max in enumerate(f_max_s):
            folder = '%s/f_max_%d_q_%d'%(test_f, f_max, q)
            S = joblib.load('%s/S.jbl'%folder)
            if S.shape[0] < 20:
                n_m = S.shape[0]
            else:
                n_m = 20
            matplotlib.pyplot.plot(range(0, n_m), numpy.log10(S/S[0])[0:n_m], '-o', c=colors[i], label='$j_{\mathrm{max}}=$%d'%f_max)  
        
        matplotlib.pyplot.legend(frameon=False, loc='upper right', fontsize=18)
        matplotlib.pyplot.gca().tick_params(axis='both', labelsize=18)
        matplotlib.pyplot.savefig('%s/SVs_fourier_%s_fmax_scan_q_%d.png'%(root_f, label, q), dpi=96*4)
        matplotlib.pyplot.close()