# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot
from scipy.optimize import curve_fit
import os

t_d_y = 0.245

def f(x, A, B):
    return A + B*numpy.cos(2*numpy.pi*x*t_d_y)

def get_I_avg_vs_k(fn):
    Is_avg = []
    
    for k in range(0, 40):
        print k
        fo = open(fn, 'r')
        Is = []
        for i in fo:
            #print i
            if int(i.split()[1]) == k:
                I = float(i.split()[3])
                #print k, I
                Is.append(I)
        if len(Is) > 0:
            Is = numpy.asarray(Is)
            I_avg = numpy.average(Is)
            Is_avg.append(I_avg)
        fo.close()
    return Is_avg


def fit(Is_avg):
    
    xdata = range(10)
    ydata = numpy.asarray(Is_avg)[0:10]
    popt, pcov = curve_fit(f, xdata, ydata)
    
    A_opt = popt[0]
    B_opt = popt[1]
    
    r = B_opt/A_opt
    fraction = 0.5 * numpy.sqrt( (1-r)/(1+r) )
    fraction = 0.5 - fraction
    
    return A_opt, B_opt, fraction

def correct(fo_w_fn, fo_r_fn, A, B):
    fo_w = open(fo_w_fn, 'w')
    fo_r = open(fo_r_fn, 'r')
    
    for i in fo_r:
        
        isplit = i.split()
        h = int(isplit[0])
        k = int(isplit[1])
        l = int(isplit[2])
        I = float(isplit[3])
        sigI = float(isplit[5])
       
        #print i
        #print h, k, l, I, sigI
        factor = 100.0/( A + B * numpy.cos( 2 * numpy.pi * k * t_d_y ) )
        I_corr = factor*I
        sigI_corr = factor*sigI
        #print I_corr, sigI_corr
        #fo_w.write('%4d%5d%5d%11.2f        -%11.2f%8d\n'%(h, k, l, I_corr, sigI_corr, n))
        #fo_w.write('%4d%5d%5d%11.2f%11.2f\n'%(h, k, l, I_corr, sigI_corr))
        fo_w.write('%4d%5d%5d%15.2f%15.2f\n'%(h, k, l, I_corr, sigI_corr))
    
    fo_w.close()
    fo_r.close()
    
def rewrite(fo_w_fn, fo_r_fn,):
    fo_w = open(fo_w_fn, 'w')
    fo_r = open(fo_r_fn, 'r')
    
    for i in fo_r:
        
        isplit = i.split()
        h = int(isplit[0])
        k = int(isplit[1])
        l = int(isplit[2])
        I = float(isplit[3])
        sigI = float(isplit[5])
        fo_w.write('%4d%5d%5d%15.2f%15.2f\n'%(h, k, l, I, sigI))
    
    fo_w.close()
    fo_r.close()    

def plot(I_avg, fign):
    matplotlib.pyplot.figure(figsize=(40, 10))  
    matplotlib.pyplot.plot(range(len(Is_avg)), Is_avg, '-o')
    matplotlib.pyplot.xlabel(r'k'),
    matplotlib.pyplot.ylabel(r'<I>_k')
    matplotlib.pyplot.savefig(fign)  
    matplotlib.pyplot.close()
    
label = '348394_pushres1.9'    

# Get average I vs k
filename = './%s_ed.hkl'%label
Is_avg = get_I_avg_vs_k(filename)

# Plot
fign = './%s_original_Is.png'%label
plot(Is_avg, fign)

#### Method 1 ###
flag = 1
if flag == 1:
    
    # Fit
    A_opt, B_opt, fraction = fit(Is_avg)
    
    # Correct intensities
    fo_w_fn = './%s_corrected.txt'%label
    correct(fo_w_fn, filename, A_opt, B_opt)
    
    # Rewrite original in different format
    fo_w_fn_original = './%s_original.txt'%label
    rewrite(fo_w_fn_original, filename)
    
    # Get average corrected I vs k
    Is_avg = get_I_avg_vs_k(fo_w_fn)
    
    # Plot
    fign = './%s_corrected_Is.png'%label
    plot(Is_avg, fign)
    
    print 'Method 1 fraction: ', fraction

#### Method 2 ###
flag = 0
if flag == 1:
    
    outfolder = 'I_correction_method_2'
    if not os.path.exists('./%s'%outfolder):
        os.mkdir('./%s'%outfolder)
    
    fractions = numpy.arange(0.01, 0.50, 0.01)
    for fraction in fractions:
        print fraction
        A = 2 * fraction * fraction - 2 * fraction + 1
        B = 2 * fraction * (1 - fraction)
        
        # Correct intensities
        fo_w_fn = './%s/%s_corrected_frac_%.2f.txt'%(outfolder, label, fraction)
        correct(fo_w_fn, filename, A, B)
        
        # Get average corrected I vs k
        Is_avg = get_I_avg_vs_k(fo_w_fn)
        
        # Plot
        fign = './%s/%s_corrected_Is_frac_%.2f.png'%(outfolder, label, fraction)
        plot(Is_avg, fign)   
    
    ## Summary figure    
    matplotlib.pyplot.figure(figsize=(40, 10))
    matplotlib.pyplot.xlabel(r'k'),
    matplotlib.pyplot.ylabel(r'<I>_k')
    
    fractions = numpy.arange(0.19, 0.26, 0.01)
    for fraction in fractions:
        print fraction
        
        # Get average corrected I vs k
        fo_w_fn = './%s/%s_corrected_frac_%.2f.txt'%(outfolder, label, fraction)
        Is_avg = get_I_avg_vs_k(fo_w_fn)
        
        # Plot
        matplotlib.pyplot.plot(range(len(Is_avg)), Is_avg, '-o', label='%.2f'%fraction)
    
    matplotlib.pyplot.legend() 
    fign = './%s/%s_corrected_Is_frac_0p19_0p26.png'%(outfolder, label)
    matplotlib.pyplot.savefig(fign)  
    matplotlib.pyplot.close()