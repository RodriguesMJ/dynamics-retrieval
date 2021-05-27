# -*- coding: utf-8 -*-
import numpy
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot
from scipy.optimize import curve_fit
import os
import warnings
warnings.filterwarnings("ignore")

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
    # The first 4 columns must be: h k l I
    # Subsequent columns and number of spaces do not matter.
    # Calculate, for each value of Miller index k, the average intensity.
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


def fit(Is_avg, k_range):
    
    xdata = range(k_range)
    ydata = numpy.asarray(Is_avg)[0:k_range]
    popt, pcov = curve_fit(f, xdata, ydata)
    
    A_opt = popt[0]
    B_opt = popt[1]
    
    r = B_opt/A_opt
    fraction = 0.5 * numpy.sqrt( (1-r)/(1+r) )
    fraction = 0.5 - fraction
    
    return A_opt, B_opt, fraction

def correct(f_w, f_r, A, B):
    print 'Writing corrected intensities in: ', f_w
    fo_w = open(f_w, 'w')
    fo_r = open(f_r, 'r')
    
    for i in fo_r:        
        isplit = i.split()
        h = int(isplit[0])
        k = int(isplit[1])
        l = int(isplit[2])
        I = float(isplit[3])
        sigI = float(isplit[4])
        
        factor = 1.0/( A + B * numpy.cos( 2 * numpy.pi * k * t_d_y ) )
        I_corr = factor*I
        sigI_corr = factor*sigI
        fo_w.write('%4d%5d%5d%15.2f%15.2f\n'%(h, k, l, I_corr, sigI_corr))
    
    fo_w.close()
    fo_r.close()
    
def rewrite(f_w, f_r,):
    fo_w = open(f_w, 'w')
    fo_r = open(f_r, 'r')
    
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
    print 'Plotting average intensities in: ', fign
    matplotlib.pyplot.figure(figsize=(40, 10))  
    matplotlib.pyplot.plot(range(len(Is_avg)), Is_avg, '-o')
    matplotlib.pyplot.xlabel(r'k', fontsize=22),
    matplotlib.pyplot.ylabel(r'<I>_k', fontsize=22)
    matplotlib.pyplot.savefig(fign)  
    matplotlib.pyplot.close()

def get_A(fraction):
    return 2 * fraction * fraction - 2 * fraction + 1

def get_B(fraction):
    return 2 * fraction * (1 - fraction)

if __name__ == '__main__':  
    
    path = './test'
    label = 'rho-alldark'
    
    # Get average I vs k
    filename = '%s/%s.txt'%(path, label)
    Is_avg = get_I_avg_vs_k(filename)
    
    # Plot
    fign = '%s/%s_original_Is.png'%(path, label)
    plot(Is_avg, fign)
    
    # Rewrite original in different format
    fo_w_fn_original = '%s/%s_original.txt'%(path, label)
    rewrite(fo_w_fn_original, filename)
    
    print '\n*** Determine translated domain fraction. ***'
    fn_r = '%s/%s_original.txt'%(path, label)
    
### METHOD 1: FIT ###
    # DOES NOT GUARANTEE A+B = 1
    out_path = '%s/method_fit'%path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print '\n*** Method 1: fit (%s) ***'%out_path
    print '(does not guarantee A+B=1)'
    
    # Fit 
    k_max = 10 # For fitting purposes
    A, B, fraction_fit = fit(Is_avg, k_max)
    
    print 'Translated domain fraction: ', fraction_fit
    print 'A: ', A
    print 'B: ', B
    print 'A+B', A+B
    
    # Correct intensities
    fn_w = '%s/%s_corrected.txt'%(out_path, label)
    correct(fn_w, fn_r, A, B)
    
    # Get average corrected I vs k
    Is_avg = get_I_avg_vs_k(fn_w)
    
    # Plot
    fign = '%s/%s_corrected_Is.png'%(out_path, label)
    plot(Is_avg, fign)
            
### METHOD 2: BRUTE FORCE ###
    out_path = '%s/method_bf'%path
    if not os.path.exists(out_path):
        os.mkdir(out_path) 
    print '\n*** Method 2: brute force (%s) ***'%out_path
    print '(A+B=1)' 
     
    fractions = numpy.arange(0.01, 0.50, 0.01)
    for fraction in fractions:
        print '\nTesting translated domain fraction: %.2f'%fraction
        A = get_A(fraction)
        B = get_B(fraction)
        
        # Correct intensities
        fn_w = '%s/%s_corrected_frac_%.2f.txt'%(out_path, label, fraction)
        correct(fn_w, fn_r, A, B)
        
        # Get average corrected I vs k
        Is_avg = get_I_avg_vs_k(fn_w)
        
        # Plot
        fign = '%s/%s_corrected_Is_frac_%.2f.png'%(out_path, label, fraction)
        plot(Is_avg, fign) 
        
    # Plot summary figure  
    print '\n*** Summary figure ***'
    matplotlib.pyplot.figure(figsize=(40, 10))
    matplotlib.pyplot.xlabel(r'k', fontsize=22),
    matplotlib.pyplot.ylabel(r'<I>_k', fontsize=22)
    
    fractions = numpy.arange(numpy.round(fraction_fit, 2)-0.04, numpy.round(fraction_fit, 2)+0.04, 0.01)
    for fraction in fractions:            
        # Get average corrected I vs k
        fn = '%s/%s_corrected_frac_%.2f.txt'%(out_path, label, fraction)
        Is_avg = get_I_avg_vs_k(fn)           
        # Plot
        matplotlib.pyplot.plot(range(len(Is_avg)), Is_avg, '-o', label='%.2f'%fraction)
    
    matplotlib.pyplot.legend() 
    fign = '%s/%s_corrected_Is_frac_range.png'%(out_path, label)
    matplotlib.pyplot.savefig(fign)  
    matplotlib.pyplot.close()
    print 'Summary figure in:', fign

### APPLY CORRECTION ###
### GUARANTEES A+B = 1
    out_path = '%s/final'%path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print '\n*** Apply results from fit (%s) ***'%out_path
    print '(enforcing A+B=1)'
            
    A = get_A(fraction_fit)
    B = get_B(fraction_fit)
    
    # Correct intensities
    fn_w = '%s/%s_corrected_frac_%.2f.txt'%(out_path, label, fraction_fit)
    correct(fn_w, fn_r, A, B)
        
    # Get average corrected I vs k
    Is_avg = get_I_avg_vs_k(fn_w)
        
    # Plot
    fign = '%s/%s_corrected_Is_frac_%.2f.png'%(out_path, label, fraction_fit)
    plot(Is_avg, fign) 