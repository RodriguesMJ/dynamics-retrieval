# -*- coding: utf-8 -*-
import numpy
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot
from scipy.optimize import curve_fit
import os

# VALUES FOR RHODOPSIN
t_d_y = 0.244
a =   61.45
b =   90.81
c =  150.82

def get_reso_P222(a, b, c, h, k, l):
    return 1.0/numpy.sqrt( (float(h)/a)**2 + (float(k)/b)**2 + (float(l)/c)**2 )

def f(x, A, B):
    return A + B*numpy.cos(2*numpy.pi*x*t_d_y)

def get_I_avg_vs_k(fn):
    Is_avg = []
    
    for k in range(0, 40):
        #print k
        fo = open(fn, 'r')
        Is = []
        for i in fo:
            #print i
            if int(i.split()[1]) == k:
                h = int(i.split()[0])
                l = int(i.split()[2])
                I = float(i.split()[3])
                d = get_reso_P222(a, b, c, h, k, l)
                if d > 1.8:
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
        
        # DEPENDING ON FORMAT
        sigI = float(isplit[5])
        #:qsigI = float(isplit[4])
       
        #print i
        #print h, k, l, I, sigI
        factor = 1.0/( A + B * numpy.cos( 2 * numpy.pi * k * t_d_y ) )
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
    
path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/translation_corr_det'

# Fixed correction 
### GUARANTEES A+B = 1
#fraction = 0.18
#A = 2 * fraction * fraction - 2 * fraction + 1
#B = 2 * fraction * (1 - fraction)

#for ts in range(0, 138300, 100):
#    print 'Timestamp: ', ts
#    label = 'rho_light_mode_0_2_timestep_%0.6d'%ts    
label = 'rho'
    
# Get average I vs k
filename = '%s/%s.txt'%(path, label)
Is_avg = get_I_avg_vs_k(filename)

# Plot
fign = '%s/%s_original_Is.png'%(path, label)
plot(Is_avg, fign)

#### Method 1 ###
flag = 0
if flag == 1:
    
    # Fit 
    # DOES NOT GUARANTEE A+B = 1
    A, B, fraction = fit(Is_avg)
            
    # Correct intensities
    fo_w_fn = '%s/%s_corrected.txt'%(path, label)
    correct(fo_w_fn, filename, A, B)
    
    # Rewrite original in different format
    fo_w_fn_original = '%s/%s_original.txt'%(path, label)
    rewrite(fo_w_fn_original, filename)
    
    # Get average corrected I vs k
    Is_avg = get_I_avg_vs_k(fo_w_fn)
    
    # Plot
    fign = '%s/%s_corrected_Is.png'%(path, label)
    plot(Is_avg, fign)
    
    print 'Method 1 fraction: ', fraction, A, B, A+B

#### Method 2 ###
flag = 1
if flag == 1:
    
    outfolder = '%s/I_correction_fraction_tests'%path
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    
    fractions = numpy.arange(0.01, 0.50, 0.01)
    for fraction in fractions:
        print fraction
        A = 2 * fraction * fraction - 2 * fraction + 1
        B = 2 * fraction * (1 - fraction)
        
        # Correct intensities
        fo_w_fn = '%s/%s_corrected_frac_%.2f.txt'%(outfolder, label, fraction)
        correct(fo_w_fn, filename, A, B)
        
        # Get average corrected I vs k
        Is_avg = get_I_avg_vs_k(fo_w_fn)
        
        # Plot
        fign = '%s/%s_corrected_Is_frac_%.2f.png'%(outfolder, label, fraction)
        plot(Is_avg, fign)   
    
    ## Summary figure    
    matplotlib.pyplot.figure(figsize=(40, 10))
    matplotlib.pyplot.xlabel(r'k'),
    matplotlib.pyplot.ylabel(r'<I>_k')
    
    fractions = numpy.arange(0.14, 0.22, 0.01)
    for fraction in fractions:
        print fraction
        
        # Get average corrected I vs k
        fo_w_fn = '%s/%s_corrected_frac_%.2f.txt'%(outfolder, label, fraction)
        Is_avg = get_I_avg_vs_k(fo_w_fn)
        
        # Plot
        matplotlib.pyplot.plot(range(len(Is_avg)), Is_avg, '-o', label='%.2f'%fraction)
    
    matplotlib.pyplot.legend() 
    fign = '%s/%s_corrected_Is_frac_range.png'%(outfolder, label)
    matplotlib.pyplot.savefig(fign)  
    matplotlib.pyplot.close()