#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:53:29 2022

@author: casadei_c
"""
import joblib
import numpy
import matplotlib.pyplot as plt
import settings_synthetic_data_jitter as settings

path = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/synthetic_data_jitter/test18/LPSA_I_missing_obs_to_avg'
print path



x = joblib.load('%s/T_filled.jbl'%path)
#x = joblib.load('%s/x.jbl'%path)
##
#x = x.todense()
##

i_s = [200, 300, 500, 700, 1000, 3500, 4000, 5000, 5500, 6000]

#summ = 0
for i in i_s:#range(x.shape[0]):#i_s:
    
    x_i = x[i, :]
    
    print x_i.shape
    
    ##
    x_i = numpy.ravel(x_i.T)
    print x_i.shape
    ##
    
    X_i = numpy.fft.rfft(x_i)
    
    #summ += x_i
    
    
    fs = numpy.fft.rfftfreq(x_i.shape[0], 1.0/x_i.shape[0])

# plt.plot(fs, numpy.abs(X_i)**2)
# plt.gca().set_xlim([-500, 16000])
# plt.savefig('%s/pixel_all_fft.png'%(path))
# plt.close()

# plt.plot(fs, numpy.abs(X_i)**2)
# plt.gca().set_xlim([0, 40])
# plt.xticks(range(0,40,4))
# plt.savefig('%s/pixel_all_fft_zoom.png'%(path))
# plt.close()

    plt.plot(fs, numpy.abs(X_i)**2)
    plt.gca().set_xlim([-500, 16000])
    plt.savefig('%s/pixel_%d_fft.png'%(path, i))
    plt.close()
    
    plt.figure(figsize=(10,5))
    plt.gca().set_xlim([0, 100])
    #plt.gca().set_ylim([min(numpy.log10(numpy.abs(X_i)**2)[0:40])-1, max(numpy.log10(numpy.abs(X_i)**2)[0:40])+1])
    ref = numpy.abs(X_i[0])**2
    plt.plot(fs, (numpy.abs(X_i)**2)/ref)
    plt.xticks(range(0,100,4))
    plt.locator_params(axis='x', nbins=8)
    plt.locator_params(axis='y', nbins=3)
    plt.gca().tick_params(axis='both', labelsize=22)
    #plt.gca().axes.get_xaxis().set_ticks([])
    spectrum = numpy.abs(X_i)**2/ref
    plt.ylim([0,0.01+max(spectrum[1:99])])
    plt.savefig('%s/pixel_%d_fft_zoom_ratio_yzoom.png'%(path, i))
    plt.close()