# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot
import random
import joblib
import os
import time 
import settings_synthetic_data_spiral as settings


def Correlate(x1, x2):
    x1Avg = numpy.average(x1)
    x2Avg = numpy.average(x2)
    numTerm = numpy.multiply(x1-x1Avg, x2-x2Avg)
    num = numTerm.sum()
    resX1Sq = numpy.multiply(x1-x1Avg, x1-x1Avg)
    resX2Sq = numpy.multiply(x2-x2Avg, x2-x2Avg)
    den = numpy.sqrt(numpy.multiply(resX1Sq.sum(), resX2Sq.sum()))
    CC = num/den
    return CC

### Test spiral
# Ideal data
flag = 0
if flag == 1:
    S = settings.S
    m = settings.m
    t = numpy.linspace(0,8*numpy.pi/2,num=S)
    
    x = numpy.zeros((m,S))
    
    x[0,:] = t*numpy.cos(t)         
    x[1,:] = t*numpy.sin(t)#+0.5*numpy.random.rand(S)
    
    matplotlib.pyplot.scatter(x[0,:], x[1,:], s=1, c=t)
    matplotlib.pyplot.savefig('%s/underlying_data.png'%settings.results_path)
    matplotlib.pyplot.close()    
    joblib.dump(x, '%s/underlying_data_%d.jbl'%(settings.results_path, S))


    
# Input data
flag = 0
if flag == 1:
    S = settings.S
    m = settings.m
    t = numpy.linspace(0,8*numpy.pi/2,num=S)
    jitter_std_dev = 0.10*8*numpy.pi/2
    t = t + numpy.random.normal(scale=jitter_std_dev, size=S)
    
    x = numpy.zeros((m,S))
    mask = numpy.zeros((m,S))
    
    thr = 0.8
    
    x_0 = t*numpy.cos(t)#+0.5*numpy.random.rand(S)
    sparsities = numpy.random.rand(S)    
    sparsities[sparsities<thr]  = 0
    sparsities[sparsities>=thr] = 1
    partialities = numpy.random.rand(S)  
    x_0 = x_0*partialities
    x_0 = x_0*sparsities
    mask[0,:] = sparsities
    x[0,:] = x_0
        
    
    x_1 = t*numpy.sin(t)#+0.5*numpy.random.rand(S)
    sparsities = numpy.random.rand(S)
    sparsities[sparsities<thr]  = 0
    sparsities[sparsities>=thr] = 1
    partialities = numpy.random.rand(S)  
    x_1 = x_1*partialities
    x_1 = x_1*sparsities
    mask[1,:] = sparsities
    x[1,:] = x_1
    
    matplotlib.pyplot.scatter(x[0,:], x[1,:], s=1, c=t)
    matplotlib.pyplot.savefig('%s/input_data.png'%settings.results_path)
    matplotlib.pyplot.close()
    
    joblib.dump(x, '%s/x.jbl'%settings.results_path)
    joblib.dump(mask, '%s/mask.jbl'%settings.results_path)
    
#############################    
###     Plain SVD of x    ###
#############################

flag = 0
if flag == 1: 
    import nlsa.SVD
    
    x = joblib.load('%s/x.jbl'%settings.results_path)
    U, S, VH = nlsa.SVD.SVD_f(x)
    U, S, VH = nlsa.SVD.sorting(U, S, VH)
    
    print 'Done'
    print 'U: ', U.shape
    print 'S: ', S.shape
    print 'VH: ', VH.shape

    joblib.dump(U, '%s/U.jbl'%settings.results_path)
    joblib.dump(S, '%s/S.jbl'%settings.results_path)
    joblib.dump(VH, '%s/VT_final.jbl'%settings.results_path)
    
flag = 0
if flag == 1:  
    import nlsa.plot_SVs
    nlsa.plot_SVs.main(settings)    
    import nlsa.plot_chronos
    nlsa.plot_chronos.main(settings)
    
flag = 0
if flag == 1:
    # reconstruct
    modes = [0, 1]
    U = joblib.load('%s/U.jbl'%settings.results_path)
    S = joblib.load('%s/S.jbl'%settings.results_path)
    VH = joblib.load('%s/VT_final.jbl'%settings.results_path)
    print U.shape, S.shape, VH.shape
    x_r_tot = 0
    for mode in modes:
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]
        print u.shape, sv.shape, vT.shape
        x_r = sv*numpy.outer(u, vT)
        print x_r.shape
        matplotlib.pyplot.scatter(x_r[0,:], x_r[1,:], s=1, c=range(x_r.shape[1]))
        matplotlib.pyplot.savefig('%s/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
        matplotlib.pyplot.close()  
        x_r_tot += x_r
    matplotlib.pyplot.scatter(x_r_tot[0,:], x_r_tot[1,:], s=1, c=range(x_r.shape[1]))
    matplotlib.pyplot.savefig('%s/x_r_tot.png'%(settings.results_path), dpi=96*3)
    matplotlib.pyplot.close()  
    
    x_underlying = joblib.load('../../synthetic_data_spiral/underlying_data_%d.jbl'%settings.S)
    x_r_tot = x_r_tot.flatten()
    x_underlying = x_underlying.flatten()
    
    CC = Correlate(x_underlying, x_r_tot)
    print(CC)
    
   
   

#############################    
###         NLSA          ###
#############################


# NO LP FILTERING
flag = 0
if flag == 1:    
    import nlsa.calculate_distances_utilities
    distance_mode = 'onlymeasured_normalised'
    if distance_mode == 'allterms':
        nlsa.calculate_distances_utilities.calculate_d_sq_dense(settings)
    elif distance_mode == 'onlymeasured':
        nlsa.calculate_distances_utilities.calculate_d_sq_sparse(settings)
    elif distance_mode == 'onlymeasured_normalised':
        nlsa.calculate_distances_utilities.calculate_d_sq_SFX_element_n(settings)
        nlsa.calculate_distances_utilities.calculate_d_sq_sparse(settings)
    else:
        print 'Undefined distance mode.'
    

    
flag = 0
if flag == 1:
    import nlsa.plot_distance_distributions
    nlsa.plot_distance_distributions.plot_d_0j(settings)
    
flag = 0
if flag == 1:
    end_worker = settings.n_workers - 1
    os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel.sh %s'
              %(end_worker, settings.__name__)) 
    os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_n_Dsq_elements.sh %s'
              %(end_worker, settings.__name__)) 
   
flag = 0
if flag == 1:    
    import nlsa.util_merge_D_sq
    nlsa.util_merge_D_sq.f(settings)   
    nlsa.util_merge_D_sq.f_N_D_sq_elements(settings)   
    import nlsa.calculate_distances_utilities
    nlsa.calculate_distances_utilities.normalise(settings)
    
flag = 0
if flag == 1:
    import nlsa.plot_distance_distributions
    nlsa.plot_distance_distributions.plot_D_0j(settings)        
    
# Select b euclidean nns or b time nns.
flag = 0
if flag == 1:    
    #import nlsa.calculate_distances_utilities
    #nlsa.calculate_distances_utilities.sort_D_sq(settings)
    import nlsa.get_D_N
    nlsa.get_D_N.main_euclidean_nn(settings)
    #nlsa.get_D_N.main_time_nn_1(settings)
    #nlsa.get_D_N.main_time_nn_2(settings)
    
flag = 0
if flag == 1:
    import nlsa.get_epsilon
    nlsa.get_epsilon.main(settings)
        
flag = 0
if flag == 1:
    import nlsa.transition_matrix
    nlsa.transition_matrix.main(settings)
    
flag = 0
if flag == 1:
    import nlsa.probability_matrix
    nlsa.probability_matrix.main(settings)
    
flag = 0
if flag == 1:
    import nlsa.eigendecompose
    nlsa.eigendecompose.main(settings)
    
 
flag = 0
if flag == 1:
    evecs = joblib.load('%s/evecs_sorted.jbl'%settings.results_path)
    test = numpy.matmul(evecs.T, evecs)
    diff = abs(test - numpy.eye(settings.l))
    print numpy.amax(diff)
    
flag = 0
if flag == 1:  
    import nlsa.plot_P_evecs
    nlsa.plot_P_evecs.main(settings)

flag = 0
if flag == 1:
    end_worker = settings.n_workers_A - 1
    os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A.sh %s'
              %(end_worker, settings.__name__)) 
    
flag = 0
if flag == 1:
    import nlsa.util_merge_A
    nlsa.util_merge_A.main(settings)

flag = 0
if flag == 1: 
    import nlsa.SVD
    nlsa.SVD.main(settings)
   
flag = 0
if flag == 1:  
    import nlsa.plot_SVs
    nlsa.plot_SVs.main(settings)    
    import nlsa.plot_chronos
    nlsa.plot_chronos.main(settings)
   
flag = 0
if flag == 1:
    end_worker = settings.n_workers_reconstruction - 1
    os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s'
              %(end_worker, settings.__name__))    

flag = 0
if flag == 1:
    import nlsa.util_merge_x_r    
    for mode in settings.modes_to_reconstruct:
        nlsa.util_merge_x_r.f(settings, mode) 
        
flag = 0
if flag == 1:
    x_r_tot = 0
    for mode in [0]:
        x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
        print x_r[0,:].shape
        matplotlib.pyplot.scatter(x_r[0,:], x_r[1,:], s=1, c=numpy.asarray(range(settings.S))[settings.q:settings.S-settings.q+1])
        matplotlib.pyplot.savefig('%s/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
        matplotlib.pyplot.close()  
        x_r_tot += x_r
    matplotlib.pyplot.scatter(x_r_tot[0,:], x_r_tot[1,:], s=1, c=numpy.asarray(range(settings.S))[settings.q:settings.S-settings.q+1])#, c=range(settings.S))
    matplotlib.pyplot.savefig('%s/x_r_tot.png'%(settings.results_path), dpi=96*3)
    matplotlib.pyplot.close() 
    
    
    x_underlying = joblib.load('../../synthetic_data_spiral/underlying_data_%d.jbl'%settings.S)[:,settings.q:settings.S-settings.q+1]
    x_r_tot = x_r_tot.flatten()
    x_underlying = x_underlying.flatten()
    
    CC = Correlate(x_underlying, x_r_tot)
    print(CC)
        
#############################    
###           SSA         ###
#############################

flag = 0
if flag == 1:
    import nlsa.calculate_distances_utilities
    nlsa.calculate_distances_utilities.concatenate_backward(settings)

flag = 0
if flag == 1: 
    import nlsa.SVD
    
    x = joblib.load('%s/X_backward_q_%d.jbl'%(settings.results_path, settings.q))
    U, S, VH = nlsa.SVD.SVD_f(x)
    U, S, VH = nlsa.SVD.sorting(U, S, VH)
    
    print 'Done'
    print 'U: ', U.shape
    print 'S: ', S.shape
    print 'VH: ', VH.shape

    joblib.dump(U, '%s/U.jbl'%settings.results_path)
    joblib.dump(S, '%s/S.jbl'%settings.results_path)
    joblib.dump(VH, '%s/VT_final.jbl'%settings.results_path)
    
flag = 0
if flag == 1:  
    import nlsa.plot_SVs
    nlsa.plot_SVs.main(settings)    
    import nlsa.plot_chronos
    nlsa.plot_chronos.main(settings)
    
flag = 0
if flag == 1:
    end_worker = settings.n_workers_reconstruction - 1
    os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s'
              %(end_worker, settings.__name__))    

flag = 0
if flag == 1:
    import nlsa.util_merge_x_r    
    for mode in settings.modes_to_reconstruct:
        nlsa.util_merge_x_r.f(settings, mode) 
        
flag = 0
if flag == 1:
    x_r_tot = 0
    for mode in [0]:#settings.modes_to_reconstruct:
        x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
        print x_r[0,:].shape
        matplotlib.pyplot.scatter(x_r[0,:], x_r[1,:], s=1, c=numpy.asarray(range(settings.S))[settings.q-1:settings.S-settings.q+1])
        matplotlib.pyplot.savefig('%s/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
        matplotlib.pyplot.close()  
        x_r_tot += x_r
    matplotlib.pyplot.scatter(x_r_tot[0,:], x_r_tot[1,:], s=1, c=numpy.asarray(range(settings.S))[settings.q-1:settings.S-settings.q+1])#, c=range(settings.S))
    matplotlib.pyplot.savefig('%s/x_r_tot.png'%(settings.results_path), dpi=96*3)
    matplotlib.pyplot.close()  
    
    x_underlying = joblib.load('../../synthetic_data_spiral/underlying_data_%d.jbl'%settings.S)[:,settings.q-1:settings.S-settings.q+1]
    x_r_tot = x_r_tot.flatten()
    x_underlying = x_underlying.flatten()
    
    CC = Correlate(x_underlying, x_r_tot)
    print(CC)
    
    
# Test optimised XTX calculation
flag = 0
if flag == 1:
    x = joblib.load('%s/x.jbl'%settings.results_path)
    print x.shape
    xTx = numpy.matmul(x.T, x)
    joblib.dump(xTx, '%s/xTx.jbl'%settings.results_path)
    print xTx.shape
    
    s = settings.S - settings.q    
    XTX = numpy.zeros((s, s))
    xTx = joblib.load('%s/xTx.jbl'%settings.results_path)
    print xTx.shape, XTX.shape
    for i in range(1, settings.q+1):
        print i
        XTX += xTx[i:i+s, i:i+s]
    joblib.dump(XTX, '%s/XTX.jbl'%settings.results_path)
    
    old_X = joblib.load('../../synthetic_data_spiral/test1/ssa/X_backward_q_%d.jbl'%settings.q)
    old_XTX = numpy.matmul(old_X.T, old_X)
    print XTX.shape
    diff = XTX - old_XTX[1:,1:]
    
    print numpy.amax(abs(diff))