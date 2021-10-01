# -*- coding: utf-8 -*-

import joblib
import numpy
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot

def get_hist(settings):

    results_path = settings.results_path
    
    D_sq = joblib.load('%s/D_sq_parallel.jbl'%results_path)
    
    print 'D_sq: ', D_sq.shape, D_sq.dtype
    print D_sq[50000,50003]
    print D_sq[60001,50007]
    print D_sq[50008,50049]
    print D_sq[5000,50013]
    
    n_iter = 100
    avg_Dsq_nn = 0
    for k_iter in range(n_iter):
        avg_Dsq_nn_k = numpy.average(numpy.diag(D_sq, k=k_iter+1))
        print 'D_sq avg', k_iter+1, 'nn in time:', avg_Dsq_nn_k
        avg_Dsq_nn = avg_Dsq_nn + avg_Dsq_nn_k
    avg_Dsq_nn = avg_Dsq_nn/n_iter # assuming s ~ s-1 ~ ... ~ s-10
    print 'D_sq overall avg',  avg_Dsq_nn
    # avg_Dsq_nn1 = numpy.average(numpy.diag(D_sq, k=1))
    # avg_Dsq_nn2 = numpy.average(numpy.diag(D_sq, k=2))
    # avg_Dsq_nn3 = numpy.average(numpy.diag(D_sq, k=3))
    
    # print 'D_sq avg 1st nn in time:', avg_Dsq_nn1
    # print 'D_sq avg 2nd nn in time:', avg_Dsq_nn2
    # print 'D_sq avg 3rd nn in time', avg_Dsq_nn3
    
    
    D_sq_flat = D_sq.flatten()
    print 'D_sq flattened: ', D_sq_flat.shape
    
    [p5, p25, p50, p75, p95] = numpy.percentile(D_sq_flat, [5, 25, 50, 75, 95])
    print  '5th percentile: ', p5
    print '25th percentile: ', p25
    print '50th percentile: ', p50
    print '75th percentile: ', p75
    print '95th percentile: ', p95


    matplotlib.pyplot.imshow(numpy.log10(D_sq+1), cmap='jet', vmin=numpy.log10(p5), vmax=numpy.log10(p95))
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/Dsq.png'%results_path)
    matplotlib.pyplot.close()
    
    # idxs = numpy.argwhere(D_sq<p95)
    # D_sq_p95 = D_sq[idxs]
    # print(D_sq_p95.shape)
    
    #mybins = numpy.linspace(0, p3, num=100)
    [p1, p96] = numpy.percentile(D_sq_flat, [1, 96])
    
    matplotlib.pyplot.figure(figsize=(10,4))
    matplotlib.pyplot.xlim((p1, p96))
    mybins = numpy.linspace(p1, p96, num=300, endpoint=True)
    
    matplotlib.pyplot.hist(D_sq_flat,bins=mybins,color='b')
    matplotlib.pyplot.axvline(x=p5,  ymin=0, ymax=1, c='m')
    matplotlib.pyplot.axvline(x=p25, ymin=0, ymax=1, c='m')
    matplotlib.pyplot.axvline(x=p50, ymin=0, ymax=1, c='m')
    matplotlib.pyplot.axvline(x=p75, ymin=0, ymax=1, c='m')
    matplotlib.pyplot.axvline(x=p95, ymin=0, ymax=1, c='m')
    matplotlib.pyplot.axvline(x=avg_Dsq_nn, ymin=0, ymax=1, c='k')
    #matplotlib.pyplot.axvline(x=avg_Dsq_nn2, ymin=0, ymax=1, c='k')
    #matplotlib.pyplot.axvline(x=avg_Dsq_nn3, ymin=0, ymax=1, c='k')
    matplotlib.pyplot.savefig('%s/Dsq_hist.png'%results_path)
    matplotlib.pyplot.close()
    
    xs = numpy.linspace(1, 100, num=100, endpoint=True)
    
    ps = numpy.percentile(D_sq_flat, xs)
    
    # matplotlib.pyplot.scatter(xs, ps)
    # matplotlib.pyplot.savefig('%s/Dsq_ps.png'%results_path)
    # matplotlib.pyplot.close()
    
    log_ps = numpy.log10(ps)
    matplotlib.pyplot.figure(figsize=(4,10))
    matplotlib.pyplot.scatter(xs, log_ps)
    matplotlib.pyplot.savefig('%s/Dsq_log_ps.png'%results_path)
    matplotlib.pyplot.close()
    
    return avg_Dsq_nn
    
def get_epsilon_curve(settings, Dsq_char):
    results_path = settings.results_path
    label = settings.label
    
    D_sq = joblib.load('%s/D_sq_parallel.jbl'%results_path)
    
    log_epsilon_list = numpy.linspace(11, 16, num=50)
    
    epsilons = []
    sigmas = []
    values = []
    for log_epsilon in log_epsilon_list:
        print log_epsilon
        epsilon = 10**log_epsilon
        sigma = numpy.sqrt(2*epsilon)
        
        exponent = D_sq/(2*epsilon)
        W = numpy.exp(-exponent)
        value =  W.sum()
        
        epsilons.append(epsilon)
        sigmas.append(sigma)
        values.append(value)
    
    values = numpy.asarray(values)    
    log_values = numpy.log10(values)
    
    max_val = numpy.amax(log_values)
    min_val = numpy.amin(log_values)
    opt_val = (max_val+min_val)/2
    
    diff = abs(log_values - opt_val)
    idx = numpy.argmin(diff)
    print idx
    
    epsilon_opt = epsilons[idx]
    sigma_opt = sigmas[idx]
    log_epsilon_opt = log_epsilon_list[idx]
    print 'Epsilon_opt = (sigma_opt**2) / 2: ', epsilon_opt
    print 'Log(epsilon_opt): ', log_epsilon_opt
    print 'Sigma_opt: ', sigma_opt
    
    log_epsilon_char = numpy.log10(Dsq_char/2)
    print 'Log(epsilon_char): ', log_epsilon_char
    print 'Sigma_char: ', Dsq_char
    
    matplotlib.pyplot.plot(log_epsilon_list, log_values)
    matplotlib.pyplot.xlabel(r"log$_{10}\epsilon$")
    matplotlib.pyplot.ylabel(r"log$_{10}\sum W_{ij}$")
    matplotlib.pyplot.axvline(x=log_epsilon_opt,  ymin=0, ymax=1)
    matplotlib.pyplot.axvline(x=log_epsilon_char, ymin=0, ymax=1, c='m')
    matplotlib.pyplot.savefig('%s/bilog_%s.png'%(results_path, label))
    matplotlib.pyplot.close()

def main(settings):
    char_Dsq = get_hist(settings)
    get_epsilon_curve(settings, char_Dsq)
    
# def main_old(settings):
#     results_path = settings.results_path
#     label = settings.label
    
#     D = joblib.load('%s/D_loop.jbl'%results_path)
#     D_sq = D*D
    
#     log_epsilon_list = numpy.linspace(4.5, 9, num=60)
    
#     epsilons = []
#     sigmas = []
#     values = []
#     for log_epsilon in log_epsilon_list:
#         print log_epsilon
#         epsilon = 10**log_epsilon
#         sigma = numpy.sqrt(2*epsilon)
        
#         exponent = D_sq/(2*epsilon)
#         W = numpy.exp(-exponent)
#         value =  W.sum()
        
#         epsilons.append(epsilon)
#         sigmas.append(sigma)
#         values.append(value)
    
#     values = numpy.asarray(values)    
#     log_values = numpy.log10(values)
    
#     max_val = numpy.amax(log_values)
#     min_val = numpy.amin(log_values)
#     opt_val = (max_val+min_val)/2
    
#     diff = abs(log_values - opt_val)
#     idx = numpy.argmin(diff)
#     print idx
    
#     epsilon_opt = epsilons[idx]
#     sigma_opt = sigmas[idx]
#     log_epsilon_opt = log_epsilon_list[idx]
#     print 'Epsilon_opt = (sigma_opt**2) / 2: ', epsilon_opt
#     print 'Log(epsilon_opt): ', log_epsilon_opt
#     print 'Sigma_opt: ', sigma_opt
    
#     matplotlib.pyplot.plot(log_epsilon_list, log_values)
#     matplotlib.pyplot.xlabel(r"log$_{10}\epsilon$")
#     matplotlib.pyplot.ylabel(r"log$_{10}\sum W_{ij}$")
#     matplotlib.pyplot.axvline(x=log_epsilon_opt, ymin=0, ymax=1)
#     matplotlib.pyplot.savefig('%s/bilog_%s.png'%(results_path, label))
#     matplotlib.pyplot.close()