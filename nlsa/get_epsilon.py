# -*- coding: utf-8 -*-
import joblib
import numpy
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot
from scipy import sparse

def get_N_Dsq_elements_distribution(settings):

    results_path = settings.results_path
    
    N_D_sq = joblib.load('%s/N_D_sq_parallel.jbl'%results_path)    
    print 'N_D_sq: ', N_D_sq.shape, N_D_sq.dtype # S-q+1 = s+1
    
    idxs = numpy.triu_indices_from(N_D_sq)    
    print 'idxs: ', len(idxs), len(idxs[0]), len(idxs[1])
    
    N_D_sq_flat = N_D_sq[idxs].flatten()
    print 'N_D_sq flattened: ', N_D_sq_flat.shape
    
    [p0p2, p99] = numpy.percentile(N_D_sq_flat, [0.2, 99])
    mybins = numpy.linspace(p0p2, p99, num=400, endpoint=True)    
    
    matplotlib.pyplot.figure(figsize=(10,10))
    matplotlib.pyplot.xlim((p0p2, p99))
    matplotlib.pyplot.hist(N_D_sq_flat,bins=mybins,color='b')
    matplotlib.pyplot.savefig('%s/N_D_sq_unique_hist.png'%results_path)
    matplotlib.pyplot.close()
  
def normalise(settings):
    results_path = settings.results_path
    
    D_sq = joblib.load('%s/D_sq_parallel.jbl'%results_path)    
    print 'D_sq: ', D_sq.shape, D_sq.dtype # S-q+1 = s+1
    
    N_D_sq = joblib.load('%s/N_D_sq_parallel.jbl'%results_path)    
    print 'N_D_sq: ', N_D_sq.shape, N_D_sq.dtype # S-q+1 = s+1
    
    D_sq = D_sq / N_D_sq
    print 'After normalisation D_sq:', D_sq.shape, D_sq.dtype
    joblib.dump(D_sq, '%s/D_sq_normalised.jbl'%results_path)
    
        
def get_distributions(settings):

    results_path = settings.results_path
    
    D_sq = joblib.load('%s/D_sq_normalised.jbl'%results_path)    
    print 'D_sq: ', D_sq.shape, D_sq.dtype # S-q+1 = s+1
    print 'N. nans', numpy.count_nonzero(numpy.isnan(D_sq))
    """
    idxs = numpy.triu_indices_from(D_sq)    
    print 'idxs: ', len(idxs), len(idxs[0]), len(idxs[1])
    
    D_flat = numpy.sqrt(D_sq[idxs].flatten())
    print 'D flattened: ', D_flat.shape
    
    [p99] = numpy.percentile(D_flat, [99])
    mybins = numpy.linspace(0, p99, num=400, endpoint=True)    
    
    matplotlib.pyplot.figure(figsize=(10,10))
    matplotlib.pyplot.xlim((0, p99))
    matplotlib.pyplot.hist(D_flat,bins=mybins,color='b')
    matplotlib.pyplot.savefig('%s/D_normalised_unique_hist.png'%results_path)
    matplotlib.pyplot.close()
    
    D_nn_r = numpy.sqrt(numpy.diag(D_sq, k=1 )[1:])
    print 'D_nn_r', D_nn_r.shape
    D_nn_l = numpy.sqrt(numpy.diag(D_sq, k=-1)[:-1])
    print 'D_nn_l', D_nn_l.shape
    
    avg_nn_D = (D_nn_r+D_nn_l)/2
    print 'avg_nn_D', avg_nn_D.shape    # S-q-1 = s-1
    
    #right_lim = max(numpy.amax(D_flat),numpy.amax(avg_nn_D))
    
    fig = matplotlib.pyplot.figure(figsize=(40,10))
    
    ax1 = fig.add_subplot(211)
    #ax1.set_xlim((p0p2, p99))
    ax1.set_xlim((0, p99))
    ax1.hist(D_flat,bins=mybins,color='b')
    
    ax2 = fig.add_subplot(212)
    ax2.set_xlim((0, p99))
    ax2.hist(avg_nn_D,bins=mybins,color='b')
    
    matplotlib.pyplot.savefig('%s/D_normalised_unique_and_nns_hist.png'%results_path)
    matplotlib.pyplot.close()
    
    flag = 0
    if flag == 1:
        D_sq = D_sq[1:-1, 1:-1]
        print 'exponent', D_sq.shape # s-1 x s-1
        
        for i in range(D_sq.shape[0]):
            #print i
            row = D_sq[i,:]
            row = row/avg_nn_D[i]
            row_final = numpy.divide(row, avg_nn_D)
            D_sq[i,:] = row_final
    
        joblib.dump(D_sq, '%s/exponent.jbl'%results_path)
        
        D_sq = numpy.asarray(D_sq, dtype=numpy.float32)
        
        idxs = numpy.triu_indices_from(D_sq)    
        print 'idxs: ', len(idxs), len(idxs[0]), len(idxs[1])
        D_sq = D_sq[idxs].flatten()
        [p1, p99] = numpy.percentile(D_sq, [1, 99])
        
        matplotlib.pyplot.figure(figsize=(20,10))
        matplotlib.pyplot.xlim((0, p99))
        mybins = numpy.linspace(0, p99, num=400, endpoint=True)    
        matplotlib.pyplot.hist(D_sq,bins=mybins,color='b')
        matplotlib.pyplot.savefig('%s/exponent_0_p99.png'%results_path)
        matplotlib.pyplot.close()
    """
def test(settings):
    results_path = settings.results_path
    
    
    flag_get_thr = 0
    if flag_get_thr == 1:
        exponent = joblib.load('%s/exponent.jbl'%results_path)
        exponent = exponent.flatten()
        [p20] = numpy.percentile(exponent, [20])
        print '20th percentile of exponent', p20
        
        threshold = numpy.exp(-p20)
        joblib.dump(threshold,'%s/W_threshold.jbl'%results_path)
    
    flag_get_W = 1
    if flag_get_W == 1:
        exponent = joblib.load('%s/exponent.jbl'%results_path)
        threshold = joblib.load('%s/W_threshold.jbl'%results_path)
        exponent = numpy.exp(-exponent)
        exponent[exponent<threshold]=0
        exponent = sparse.csr_matrix(exponent)
        print 'W:', exponent.shape, exponent.dtype
        print 'N.nonzero:', exponent.count_nonzero()
        joblib.dump(exponent, '%s/W_sym_local_distances.jbl'%results_path)
        
        counts = []
        for i in range(exponent.shape[0]):
            count = exponent[i,:].count_nonzero()
            counts.append(count)
        matplotlib.pyplot.figure(figsize=(30,10))
        matplotlib.pyplot.plot(range(exponent.shape[0]), counts)
        matplotlib.pyplot.axhline(y=0,xmin=0,xmax=1,c='m')
        matplotlib.pyplot.gca().set_ylim(bottom=-1000) 
        matplotlib.pyplot.savefig('%s/W_sym_local_distances_nns.png'%results_path)
        matplotlib.pyplot.close()
        

    

def get_hist(settings):

    results_path = settings.results_path
    
    D_sq = joblib.load('%s/D_sq_parallel.jbl'%results_path)
    
    print 'D_sq: ', D_sq.shape, D_sq.dtype
    
    avg_Dsq_nn_k = numpy.average(numpy.diag(D_sq, k=0))
    print 'D_sq avg main diagonal:', avg_Dsq_nn_k
    
    n_iter = 100
    avg_Dsq_nn = 0
    for k_iter in range(n_iter):
        avg_Dsq_nn_k = numpy.average(numpy.diag(D_sq, k=k_iter+1))
        print 'D_sq avg of', k_iter+1, 'nn in time:', avg_Dsq_nn_k
        avg_Dsq_nn = avg_Dsq_nn + avg_Dsq_nn_k
    Dsq_char_avg = avg_Dsq_nn/n_iter # assuming s ~ s-1 ~ ... ~ s-10
    print 'Characteristic D_sq (avg of 10 nns): ',  Dsq_char_avg
    
    Dsq_char_median = numpy.percentile(numpy.diag(D_sq, k=1), [50])
    print 'Characteristic D_sq (median of 1st nns): ',  Dsq_char_median
    
    D_sq_flat = D_sq.flatten()
    print 'D_sq flattened: ', D_sq_flat.shape
    
    [p5, p25, p50, p75, p95] = numpy.percentile(D_sq_flat, [5, 25, 50, 75, 95])
    print  '5th percentile: ', p5
    print '25th percentile: ', p25
    print '50th percentile: ', p50
    print '75th percentile: ', p75
    print '95th percentile: ', p95

    D_sq = numpy.asarray(D_sq, dtype=numpy.float32)
    matplotlib.pyplot.imshow(numpy.log10(D_sq+1), cmap='jet', vmin=numpy.log10(p5), vmax=numpy.log10(p95))
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/Dsq.png'%results_path)
    matplotlib.pyplot.close()
    
    xs = numpy.linspace(1, 100, num=100, endpoint=True)    
    ps = numpy.percentile(D_sq_flat, xs)
    joblib.dump(ps, '%s/Dsq_ps_1_100_percent.jbl'%results_path)
    
    log_ps = numpy.log10(ps)
    matplotlib.pyplot.figure(figsize=(4,10))
    matplotlib.pyplot.scatter(xs, log_ps)
    matplotlib.pyplot.savefig('%s/Dsq_log_ps.png'%results_path)
    matplotlib.pyplot.close()
    
    
    [p0p2, p96] = numpy.percentile(D_sq_flat, [0.2, 96])
    
    matplotlib.pyplot.figure(figsize=(10,4))
    matplotlib.pyplot.xlim((p0p2, p96))
    mybins = numpy.linspace(p0p2, p96, num=400, endpoint=True)
    
    matplotlib.pyplot.hist(D_sq_flat,bins=mybins,color='b')
    for i in range(5,100,5):
        #p = numpy.percentile(D_sq_flat, i)
        p = ps[i-1]
        matplotlib.pyplot.axvline(x=p,  ymin=0, ymax=1, c='m')
    # matplotlib.pyplot.axvline(x=p25, ymin=0, ymax=1, c='m')
    # matplotlib.pyplot.axvline(x=p50, ymin=0, ymax=1, c='m')
    # matplotlib.pyplot.axvline(x=p75, ymin=0, ymax=1, c='m')
    # matplotlib.pyplot.axvline(x=p95, ymin=0, ymax=1, c='m')
    # matplotlib.pyplot.axvline(x=Dsq_char_avg,    ymin=0, ymax=1, c='y')
    # matplotlib.pyplot.axvline(x=Dsq_char_median, ymin=0, ymax=1, c='c')
    matplotlib.pyplot.savefig('%s/Dsq_hist.png'%results_path)
    matplotlib.pyplot.close()
    
    
    joblib.dump(p25,             '%s/Dsq_p25.jbl'%results_path)
    joblib.dump(Dsq_char_avg,    '%s/Dsq_char_avg.jbl'%results_path)
    joblib.dump(Dsq_char_median, '%s/Dsq_char_median.jbl'%results_path)
    
    xs = numpy.linspace(0.1, 1.0, num=10, endpoint=True)    
    ps = numpy.percentile(D_sq_flat, xs)
    print 'D_sq percentiles from 0.1 to 1 percent:'
    print xs
    print ps
    joblib.dump(ps, '%s/Dsq_ps_0p1_1p0_percent.jbl'%results_path)
    
    
def get_epsilon_curve(settings):
    results_path = settings.results_path
    label = settings.label
    
    ps = joblib.load('%s/Dsq_ps_1_100_percent.jbl'%results_path)
    mylist = []
    for i in range(5,100,5):
        Dsq_p = ps[i-1]
        log_epsilon_p = numpy.log10(Dsq_p/2)  
        mylist.append(log_epsilon_p)
        
    Dsq_p25 = joblib.load('%s/Dsq_p25.jbl'%results_path)
    Dsq_char_avg = joblib.load('%s/Dsq_char_avg.jbl'%results_path)
    Dsq_char_median = joblib.load('%s/Dsq_char_median.jbl'%results_path)
    
    D_sq = joblib.load('%s/D_sq_parallel.jbl'%results_path)
    D_sq = numpy.asarray(D_sq, dtype=numpy.float32)
    
    log_epsilon_list = numpy.linspace(11, 16, num=50)
    
    epsilons = []
    sigmas = []
    values = []
    for log_epsilon in log_epsilon_list:
        print log_epsilon
        epsilon = 10**log_epsilon
        sigma = numpy.sqrt(2*epsilon)
        
        W = numpy.exp(-D_sq/(2*epsilon))
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
    
    sigma_old = 4*sigma_opt
    log_epsilon_old = numpy.log10((sigma_old**2)/2)
    
    log_epsilon_char_avg = numpy.log10(Dsq_char_avg/2)
    print 'Log(epsilon_char_avg): ', log_epsilon_char_avg
    print 'Sigma_char_avg: ', numpy.sqrt(Dsq_char_avg)
    
    log_epsilon_char_median = numpy.log10(Dsq_char_median/2)
    print 'Log(epsilon_char_median): ', log_epsilon_char_median
    print 'Sigma_char_median: ', numpy.sqrt(Dsq_char_median)
    
    log_epsilon_p25 = numpy.log10(Dsq_p25/2)
    print 'Log(epsilon_p25): ', log_epsilon_p25
    print 'Sigma_p25: ', numpy.sqrt(Dsq_p25)
    
    Dsq_final = Dsq_p25/3
    log_epsilon_final = numpy.log10(Dsq_final/2)
    print 'Log(epsilon_final): ', log_epsilon_final
    print 'Sigma_final: ', numpy.sqrt(Dsq_final)
    
    matplotlib.pyplot.figure(figsize=(30,10))
    matplotlib.pyplot.plot(log_epsilon_list, log_values)
    matplotlib.pyplot.xlabel(r"log$_{10}\epsilon$")
    matplotlib.pyplot.ylabel(r"log$_{10}\sum W_{ij}$")
    matplotlib.pyplot.axvline(x=log_epsilon_opt,  ymin=0, ymax=1, c='b')
    matplotlib.pyplot.axvline(x=log_epsilon_old,  ymin=0, ymax=1, c='b')
    for i in mylist:
        matplotlib.pyplot.axvline(x=i,  ymin=0, ymax=1, c='m')
    # matplotlib.pyplot.axvline(x=log_epsilon_p25,         ymin=0, ymax=1, c='m')
    # matplotlib.pyplot.axvline(x=log_epsilon_char_avg,    ymin=0, ymax=1, c='y')
    # matplotlib.pyplot.axvline(x=log_epsilon_char_median, ymin=0, ymax=1, c='c')
    # matplotlib.pyplot.axvline(x=log_epsilon_final,       ymin=0, ymax=1, c='k')
    matplotlib.pyplot.savefig('%s/bilog_%s.png'%(results_path, label))
    matplotlib.pyplot.close()

def get_hist_b_nns(settings):

    results_path = settings.results_path
    
    D_sq = joblib.load('%s/D_loop.jbl'%results_path)
    D_sq = D_sq*D_sq
    print 'D_sq: ', D_sq.shape, D_sq.dtype
    
    
    D_sq_flat = D_sq.flatten()
    print 'D_sq flattened: ', D_sq_flat.shape
    
    [p5, p25, p50, p75, p95] = numpy.percentile(D_sq_flat, [5, 25, 50, 75, 95])
    print  '5th percentile: ', p5
    print '25th percentile: ', p25
    print '50th percentile: ', p50
    print '75th percentile: ', p75
    print '95th percentile: ', p95

    D_sq = numpy.asarray(D_sq, dtype=numpy.float32)
    matplotlib.pyplot.imshow(numpy.log10(D_sq+1), cmap='jet', vmin=numpy.log10(p5), vmax=numpy.log10(p95))
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/Dsq_b_nns.png'%results_path)
    matplotlib.pyplot.close()
    
    xs = numpy.linspace(1, 100, num=100, endpoint=True)    
    ps = numpy.percentile(D_sq_flat, xs)
    joblib.dump(ps, '%s/Dsq_ps_1_100_percent_b_nns.jbl'%results_path)
    
    log_ps = numpy.log10(ps)
    matplotlib.pyplot.figure(figsize=(4,10))
    matplotlib.pyplot.scatter(xs, log_ps)
    matplotlib.pyplot.savefig('%s/Dsq_log_ps_b_nns.png'%results_path)
    matplotlib.pyplot.close()
        
    [p0p2, p96] = numpy.percentile(D_sq_flat, [0.2, 96])
    
    matplotlib.pyplot.figure(figsize=(10,4))
    matplotlib.pyplot.xlim((p0p2, p96))
    mybins = numpy.linspace(p0p2, p96, num=400, endpoint=True)
    
    matplotlib.pyplot.hist(D_sq_flat,bins=mybins,color='b')
    for i in range(5,100,5):
        p = ps[i-1]
        matplotlib.pyplot.axvline(x=p,  ymin=0, ymax=1, c='m')
    matplotlib.pyplot.savefig('%s/Dsq_hist_b_nns.png'%results_path)
    matplotlib.pyplot.close()
        
    xs = numpy.linspace(0.1, 1.0, num=10, endpoint=True)    
    ps = numpy.percentile(D_sq_flat, xs)
    print 'D_sq percentiles from 0.1 to 1 percent:'
    print xs
    print ps
    joblib.dump(ps, '%s/Dsq_ps_0p1_1p0_percent_b_nns.jbl'%results_path)


def get_epsilon_curve_b_nns(settings):
    results_path = settings.results_path
    label = settings.label
    
    ps = joblib.load('%s/Dsq_ps_1_100_percent_b_nns.jbl'%results_path)
    mylist = []
    for i in range(5,100,5):
        Dsq_p = ps[i-1]
        log_epsilon_p = numpy.log10(Dsq_p/2)  
        mylist.append(log_epsilon_p)
        
    
    D_sq = joblib.load('%s/D_loop.jbl'%results_path)
    D_sq = D_sq*D_sq
    #D_sq = numpy.asarray(D_sq, dtype=numpy.float32)
    
    log_epsilon_list = numpy.linspace(11, 16, num=50)
    
    epsilons = []
    sigmas = []
    values = []
    for log_epsilon in log_epsilon_list:
        print log_epsilon
        epsilon = 10**log_epsilon
        sigma = numpy.sqrt(2*epsilon)
        
        W = numpy.exp(-D_sq/(2*epsilon))
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
    
    sigma_old = 4*sigma_opt
    log_epsilon_old = numpy.log10((sigma_old**2)/2)
    
    
    matplotlib.pyplot.figure(figsize=(30,10))
    matplotlib.pyplot.plot(log_epsilon_list, log_values)
    matplotlib.pyplot.xlabel(r"log$_{10}\epsilon$")
    matplotlib.pyplot.ylabel(r"log$_{10}\sum W_{ij}$")
    matplotlib.pyplot.axvline(x=log_epsilon_opt,  ymin=0, ymax=1, c='b')
    matplotlib.pyplot.axvline(x=log_epsilon_old,  ymin=0, ymax=1, c='b')
    for i in mylist:
        matplotlib.pyplot.axvline(x=i,  ymin=0, ymax=1, c='m')
    matplotlib.pyplot.savefig('%s/bilog_%s_b_nns.png'%(results_path, label))
    matplotlib.pyplot.close()
    
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