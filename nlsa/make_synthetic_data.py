# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot
import random
import joblib
import os
import time 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

import settings_synthetic_data as settings
import correlate
import local_linearity

def plot_components(settings):
    m = settings.m
    S = settings.S
    q = settings.q
    results_path = settings.results_path
    
    x_1  = numpy.zeros((m,S))    
    x_2  = numpy.zeros((m,S))    
    
    ts = numpy.asarray(range(S), dtype=numpy.double)
    T = S-q
    omega = 2*numpy.pi/T
    
    tc = float(S)/2
    e1 = 1-numpy.exp(-ts/tc)
    e2 = 1-e1
    
    for i in range(m): 
        A_i = numpy.cos(0.6*(2*numpy.pi/m)*i) 
        B_i = numpy.sin(3*(2*numpy.pi/m)*i+numpy.pi/5) 
        C_i = numpy.sin(0.8*(2*numpy.pi/m)*i+numpy.pi/7) 
        D_i = numpy.cos(2.1*(2*numpy.pi/m)*i) 
        E_i = numpy.cos(1.2*(2*numpy.pi/m)*i+numpy.pi/10) 
        F_i = numpy.sin(1.8*(2*numpy.pi/m)*i+numpy.pi/11) 
        
        x_1[i,:] = (e1*(
                          A_i + 
                          B_i*numpy.cos(3*omega*ts) + 
                          C_i*numpy.sin(10*omega*ts)
                          ) )
        x_2[i,:] = (e2*(
                          D_i+
                          E_i*numpy.sin(7*omega*ts) +
                          F_i*numpy.sin(11*omega*ts+numpy.pi/10) 
                          ))
        
    matplotlib.pyplot.imshow(x_1, cmap='jet')
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/x_1.png'%(results_path), dpi=96*3)
    matplotlib.pyplot.close()  
    
    matplotlib.pyplot.imshow(x_2, cmap='jet')
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/x_2.png'%(results_path), dpi=96*3)
    matplotlib.pyplot.close()
    
    
    matplotlib.pyplot.imshow(x_1+x_2, cmap='jet')
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/sum.png'%(results_path), dpi=96*3)
    matplotlib.pyplot.close()

flag = 0
if flag == 1:    
    plot_components(settings)  
    
    
def make_x(settings):
    m = settings.m
    S = settings.S
    q = settings.q
    results_path = settings.results_path
    
    x = numpy.zeros((m,S))    
    mask = numpy.zeros((m,S))
    
    ts = numpy.asarray(range(S), dtype=numpy.double)
    T = S-q
    omega = 2*numpy.pi/T
    
    min_period = T/11
    jitter_factor = 0.3
    jitter_std_dev = jitter_factor*min_period
    ts = ts + numpy.random.normal(scale=jitter_std_dev, size=S)
    
    tc = float(S)/2
    e1 = 1-numpy.exp(-ts/tc)
    e2 = 1-e1
    
    for i in range(m):        
                     
        A_i = numpy.cos(0.6*(2*numpy.pi/m)*i) 
        B_i = numpy.sin(3*(2*numpy.pi/m)*i+numpy.pi/5) 
        C_i = numpy.sin(0.8*(2*numpy.pi/m)*i+numpy.pi/7) 
        D_i = numpy.cos(2.1*(2*numpy.pi/m)*i) 
        E_i = numpy.cos(1.2*(2*numpy.pi/m)*i+numpy.pi/10) 
        F_i = numpy.sin(1.8*(2*numpy.pi/m)*i+numpy.pi/11) 
        x_i = (e1*(
              A_i + 
              B_i*numpy.cos(3*omega*ts) + 
              C_i*numpy.sin(10*omega*ts)
              ) + 
              e2*(
              D_i+
              E_i*numpy.sin(7*omega*ts) +
              F_i*numpy.sin(11*omega*ts+numpy.pi/10) 
              ))
              
        
        partialities = numpy.random.rand(S)             
        x_i = x_i*partialities
        
        sparsities = numpy.random.rand(S)
        thr = 0.982
        sparsities[sparsities<thr]  = 0
        sparsities[sparsities>=thr] = 1
        x_i = x_i*sparsities
        
        mask[i,:] = sparsities
        x[i,:] = x_i
        
        
    matplotlib.pyplot.imshow(x, cmap='jet')
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/x_jitter_factor_%.2f.png'%(results_path, jitter_factor), dpi=96*3)
    matplotlib.pyplot.close()    
    
    joblib.dump(x, '%s/x.jbl'%results_path)  
    joblib.dump(mask, '%s/mask.jbl'%results_path)

def make_lp_filter_functions(settings):
    import make_lp_filter
    
    results_path = settings.results_path
    F = make_lp_filter.get_F(settings)
    Q, R = make_lp_filter.on_qr(settings, F)
    
    d = make_lp_filter.check_on(Q)
    print numpy.amax(abs(d))
    
    joblib.dump(Q, '%s/F_on_qr.jbl'%results_path)


flag = 0
if flag == 1:    
    make_x(settings)

flag = 0
if flag == 1:
    x = joblib.load('%s/x.jbl'%settings.results_path)  
    im = matplotlib.pyplot.imshow(x, cmap='jet')
    
    ax = matplotlib.pyplot.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
   
    cb = matplotlib.pyplot.colorbar(im, cax=cax)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cb.locator = tick_locator
    cb.update_ticks()
    matplotlib.pyplot.savefig('%s/x_jet.svg'%(settings.results_path), dpi=96*3)
    matplotlib.pyplot.close() 

################
###   NLSA   ###
################

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
    d_sq = joblib.load('%s/d_sq.jbl'%(settings.results_path))
    print numpy.amax(d_sq), numpy.amin(d_sq)
    print numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq))

flag = 0
if flag == 1:
    x = joblib.load('%s/x.jbl'%(settings.results_path))
    d_ij = (x[:,5000]-x[:,5001])**2
    d_ij = numpy.sqrt(d_ij.sum())
    print d_ij
    
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
    #os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction_onecopy.sh %s'
    #          %(end_worker, settings.__name__))    

flag = 0
if flag == 1:
    import nlsa.util_merge_x_r    
    for mode in settings.modes_to_reconstruct:
        nlsa.util_merge_x_r.f(settings, mode) 
        
flag = 0
if flag == 1:
    
    benchmark = joblib.load('../../synthetic_data_4/test5/x.jbl')
    print benchmark.shape
    benchmark = benchmark[:, settings.q:settings.q+(settings.S-settings.q-settings.ncopies+1)]
    print benchmark.shape
    benchmark = benchmark.flatten()
    
    CCs = []
    
    x_r_tot = 0
    for mode in settings.modes_to_reconstruct:
        print mode
        x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
        matplotlib.pyplot.imshow(x_r, cmap='jet')
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.savefig('%s/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
        matplotlib.pyplot.close()  
        x_r_tot += x_r
    
        matplotlib.pyplot.imshow(x_r_tot, cmap='jet')
        matplotlib.pyplot.colorbar()
        
        print x_r_tot.shape
        
        x_r_tot_flat = x_r_tot.flatten()
        
        CC = correlate.Correlate(benchmark, x_r_tot_flat)
        print CC
        
        CCs.append(CC)
        
        matplotlib.pyplot.title('%.4f'%CC)
        matplotlib.pyplot.savefig('%s/x_r_tot_%d_modes.png'%(settings.results_path, mode+1), dpi=96*3)
        matplotlib.pyplot.close() 
        
    joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)

    
#################################
# Cmp
flag = 0
if flag == 1:
    Dsq_1 = joblib.load('../../synthetic_data_4/test5/nlsa/q_4000/D_sq_parallel.jbl')
    print Dsq_1.shape
    Dsq_2 = joblib.load('../../synthetic_data_4/test6/nlsa/q_4000/distance_calculation_allterms/D_sq_parallel.jbl')
    print Dsq_2.shape
    Dsq_3 = joblib.load('../../synthetic_data_4/test6/nlsa/q_4000/distance_calculation_onlymeasured_normalised/D_sq_normalised.jbl')
    print Dsq_2.shape
    
    
    print numpy.amax(Dsq_1), numpy.amin(Dsq_1)
    print 'Diag:', numpy.amax(numpy.diag(Dsq_1)), numpy.amin(numpy.diag(Dsq_1))
    numpy.fill_diagonal(Dsq_1, 0)
    print numpy.amax(Dsq_1), numpy.amin(Dsq_1)
    Dsq_1[Dsq_1<0]=0
    print numpy.amax(Dsq_1), numpy.amin(Dsq_1)
    
    print numpy.amax(Dsq_2), numpy.amin(Dsq_2)
    print 'Diag:', numpy.amax(numpy.diag(Dsq_2)), numpy.amin(numpy.diag(Dsq_2))
    numpy.fill_diagonal(Dsq_2, 0)
    print numpy.amax(Dsq_2), numpy.amin(Dsq_2)
    Dsq_2[Dsq_2<0]=0
    print numpy.amax(Dsq_2), numpy.amin(Dsq_2)
    
    Dsq_3[Dsq_3<0]=0
    
    D_1 = numpy.sqrt(Dsq_1)
    D_2 = numpy.sqrt(Dsq_2)
    D_3 = numpy.sqrt(Dsq_3)  
    
    print D_1.shape
    print D_2.shape
    
    D_1 = D_1.flatten()
    D_2 = D_2.flatten()
    D_3 = D_3.flatten()
    
    print D_1.shape
    print D_2.shape
    CC = correlate.Correlate(D_1,D_2)
    print CC
    CC = correlate.Correlate(D_1,D_3)
    print CC

    # fig, axs = matplotlib.pyplot.subplots(6, 1, sharex=True, sharey=False)
    # fig.suptitle('CC: %.4f'%CC)
    # # marker symbol
    # axs[0].plot(range(Dsq_1.shape[1]), Dsq_1[5000,:],  'o-', c='b', markersize=3, markeredgewidth=0.0)
    # #axs[0, 0].set_title("marker='>'")
    # axs[1].plot(range(Dsq_1.shape[1]), Dsq_2[5000,:],  'o-', c='b', markersize=3, markeredgewidth=0.0)
    # #axs[0, 1].set_title("marker='>'")
    # axs[2].plot(range(Dsq_1.shape[1]), Dsq_1[15000,:], 'o-', c='b', markersize=3, markeredgewidth=0.0)
    # #axs[0, 2].set_title("marker='>'")
    # axs[3].plot(range(Dsq_1.shape[1]), Dsq_2[15000,:], 'o-', c='b', markersize=3, markeredgewidth=0.0)
    # #axs[1, 0].set_title("marker='>'")
    # axs[4].plot(range(Dsq_1.shape[1]), Dsq_1[25000,:], 'o-', c='b', markersize=3, markeredgewidth=0.0)
    # #axs[1, 1].set_title("marker='>'")
    # axs[5].plot(range(Dsq_1.shape[1]), Dsq_2[25000,:], 'o-', c='b', markersize=3, markeredgewidth=0.0)
    # #axs[1, 2].set_title("marker='>'")
    # matplotlib.pyplot.savefig('../../synthetic_data_4/Distances_benchmark_to_sparsepartial.png', dpi=4*96)


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
    
    benchmark = joblib.load('../../synthetic_data_4/test5/x.jbl')
    print benchmark.shape
    benchmark = benchmark.flatten()
    
    CCs = []
    
    # reconstruct
    modes = range(20)#[0,1,2,3]
    U = joblib.load('%s/U.jbl'%settings.results_path)
    S = joblib.load('%s/S.jbl'%settings.results_path)
    VH = joblib.load('%s/VT_final.jbl'%settings.results_path)
    print U.shape, S.shape, VH.shape
    x_r_tot = 0
    for mode in modes:
        print mode
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]
        #print u.shape, sv.shape, vT.shape
        x_r = sv*numpy.outer(u, vT)
        #print x_r.shape
        
        # matplotlib.pyplot.imshow(x_r, cmap='jet')
        # matplotlib.pyplot.colorbar()
        # matplotlib.pyplot.savefig('%s/modes_sum/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
        # matplotlib.pyplot.close()  
        x_r_tot += x_r
    
        # matplotlib.pyplot.imshow(x_r_tot, cmap='jet')
        # matplotlib.pyplot.colorbar()
        
        #print x_r_tot.shape
        
        x_r_tot_flat = x_r_tot.flatten()
        
        CC = correlate.Correlate(benchmark, x_r_tot_flat)
        print CC
        CCs.append(CC)
        
        # matplotlib.pyplot.title('%.4f'%CC)
        # matplotlib.pyplot.savefig('%s/modes_sum/x_r_tot_%d_modes.png'%(settings.results_path, mode+1), dpi=96*3)
        # matplotlib.pyplot.close() 
        
    joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)
    
#########################
### Fourier filtering ###
#########################

flag = 0
if flag == 1:
    make_lp_filter_functions(settings)
    
flag = 0
if flag == 1:
    end_worker = settings.n_workers_A - 1
    os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A_fourier.sh %s'
              %(end_worker, settings.__name__)) 
   
flag = 0
if flag == 1:
    import nlsa.util_merge_A
    nlsa.util_merge_A.main(settings)

flag = 0
if flag == 1: 
    import nlsa.SVD
    print '\n****** RUNNING SVD ******'
    
    results_path = settings.results_path
    datatype = settings.datatype

    A = joblib.load('%s/A_parallel.jbl'%results_path)
    print 'Loaded'
    U, S, VH = nlsa.SVD.SVD_f_manual(A)
    U, S, VH = nlsa.SVD.sorting(U, S, VH)
    
    print 'Done'
    print 'U: ', U.shape
    print 'S: ', S.shape
    print 'VH: ', VH.shape

    joblib.dump(U, '%s/U.jbl'%results_path)
    joblib.dump(S, '%s/S.jbl'%results_path)
    joblib.dump(VH, '%s/VH.jbl'%results_path)
    
    evecs = joblib.load('%s/F_on_qr.jbl'%(results_path))
    Phi = evecs[:,0:2*settings.f_max_considered+1]
    
    VT_final = nlsa.SVD.project_chronos(VH, Phi)    
    print 'VT_final: ', VT_final.shape
    joblib.dump(VT_final, '%s/VT_final.jbl'%results_path)
   
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
    
    benchmark = joblib.load('../../synthetic_data_4/test5/x.jbl')
    print benchmark.shape
    benchmark = benchmark[:, settings.q:settings.q+(settings.S-settings.q-settings.ncopies+1)]
    print benchmark.shape
    benchmark = benchmark.flatten()
    
    CCs = []
    
    x_r_tot = 0
    for mode in range(20):#settings.modes_to_reconstruct:
        print mode
        x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
        # matplotlib.pyplot.imshow(x_r, cmap='jet')
        # matplotlib.pyplot.colorbar()
        # matplotlib.pyplot.savefig('%s/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
        # matplotlib.pyplot.close()  
        x_r_tot += x_r
    
        # matplotlib.pyplot.imshow(x_r_tot, cmap='jet')
        # matplotlib.pyplot.colorbar()
        
        print x_r_tot.shape
               
        x_r_tot_flat = x_r_tot.flatten()
        
        CC = correlate.Correlate(benchmark, x_r_tot_flat)
        print CC
        
        # matplotlib.pyplot.title('%.4f'%CC)
        # matplotlib.pyplot.savefig('%s/x_r_tot.png'%(settings.results_path), dpi=96*3)
        # matplotlib.pyplot.close() 
        
        CCs.append(CC)
    joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)


# Test U to u
flag = 0
if flag == 1:
    U = joblib.load('%s/U.jbl'%settings.results_path)
    print 'U:', U.shape
    ntopos=20
    U = U[:,0:ntopos]
    print 'U:', U.shape
    u = numpy.zeros((settings.m, ntopos))
    for i in range(ntopos):
        print i
        U_i = U[:,i]
        print 'U_i:', U_i.shape
        print U_i[0,], U_i[1], U_i[settings.m]
        
        u_i = U_i.reshape((settings.m, settings.q), order='F')
        print u_i[0,0], u_i[1,0], u_i[0,1]
        u_i = numpy.average(u_i, axis=1)
        print u_i.shape
        u[:,i] = u_i
    joblib.dump(u, '%s/u.jbl'%settings.results_path)

flag = 0
if flag == 1:
    
    # benchmark = joblib.load('../../synthetic_data_4/test5/x.jbl')
    # print benchmark.shape
    # benchmark = benchmark.flatten()
    
    #CCs = []
    
    # reconstruct
    modes = range(20)#[0,1,2,3]
    U = joblib.load('%s/u.jbl'%settings.results_path)
    S = joblib.load('%s/S.jbl'%settings.results_path)
    VH = joblib.load('%s/VT_final.jbl'%settings.results_path)
    print U.shape, S.shape, VH.shape
    x_r_tot = 0
    for mode in modes:
        print mode
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]
        #print u.shape, sv.shape, vT.shape
        x_r = sv*numpy.outer(u, vT)
        print x_r.shape
        
        matplotlib.pyplot.imshow(x_r, cmap='jet')
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.savefig('%s/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
        matplotlib.pyplot.close()  
        x_r_tot += x_r
    
        matplotlib.pyplot.imshow(x_r_tot, cmap='jet')
        matplotlib.pyplot.colorbar()
        
        print x_r_tot.shape
        
        #x_r_tot_flat = x_r_tot.flatten()
        
        # CC = Correlate(benchmark, x_r_tot_flat)
        # print CC
        # CCs.append(CC)
        
        #matplotlib.pyplot.title('%.4f'%CC)
        matplotlib.pyplot.savefig('%s/x_r_tot_%d_modes.png'%(settings.results_path, mode+1), dpi=96*3)
        matplotlib.pyplot.close() 
        
    #joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)   
###################
######  SSA  ######  
###################

# Calc xTx
flag = 0
if flag == 1:
    x = joblib.load('%s/x.jbl'%settings.results_path)
    print 'x: ', x.shape
    xTx = numpy.matmul(x.T, x)
    joblib.dump(xTx, '%s/xTx.jbl'%settings.results_path)
    print 'xTx: ', xTx.shape

# Calc XTX
flag = 0
if flag == 1:
    s = settings.S - settings.q    
    XTX = numpy.zeros((s, s))
    xTx = joblib.load('%s/xTx.jbl'%settings.results_path)
    print 'xTx: ', xTx.shape, 'XTX: ', XTX.shape
    start = time.time()
    for i in range(1, settings.q+1): # Time ~q seconds
        print i
        XTX += xTx[i:i+s, i:i+s]
    print 'Time: ', time.time()-start
    joblib.dump(XTX, '%s/XTX.jbl'%settings.results_path)

# SVD XTX
flag = 0
if flag == 1:
    XTX = joblib.load('%s/XTX.jbl'%settings.results_path)
    evals_XTX, evecs_XTX = numpy.linalg.eigh(XTX)
    print 'Done'
    for i in evals_XTX:
        print i
    evals_XTX[numpy.argwhere(evals_XTX<0)]=0  
    SVs = numpy.sqrt(evals_XTX)
    VT = evecs_XTX.T
    print 'Sorting'
    sort_idxs = numpy.argsort(SVs)[::-1]
    SVs_sorted = SVs[sort_idxs]
    VT_sorted = VT[sort_idxs,:]
    joblib.dump(SVs_sorted, '%s/S.jbl'%settings.results_path)
    joblib.dump(VT_sorted, '%s/VT_final.jbl'%settings.results_path)

flag = 0
if flag == 1:    
    SVs = joblib.load('%s/S.jbl'%settings.results_path)
    VT  = joblib.load('%s/VT_final.jbl'%settings.results_path)
    U_temp = numpy.matmul(VT.T, numpy.diag(1.0/SVs))
    print 'VS-1: ', U_temp.shape
    U_temp = U_temp[:,0:20]
    x = joblib.load('%s/x.jbl'%settings.results_path)
    m = x.shape[0]
    q = settings.q
    s = x.shape[1]-q
    U = numpy.zeros((m*q, 20))
    start = time.time()
    for j in range(0,q):
        #j=0
        #U[0:m,:] = numpy.matmul(x[:,q:q+s], U_temp)
        U[j*m:(j+1)*m,:] = numpy.matmul(x[:,q-j:q+s-j], U_temp)
    #U = numpy.matmul(A, U_temp)     
    print 'Time: ', time.time()-start
    joblib.dump(U, '%s/U.jbl'%settings.results_path)
    
    

flag = 0
if flag == 1:  
    import nlsa.plot_SVs
    nlsa.plot_SVs.main(settings)    
    VT_final = joblib.load('%s/VT_final.jbl'%(settings.results_path))
    
    nmodes = VT_final.shape[0]
    s = VT_final.shape[1]
    
    print 'nmodes: ', nmodes
    print 's: ', s
    
    out_folder = '%s/chronos'%(settings.results_path)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    for i in range(20):
        print i
        chrono = VT_final[i,:]
        matplotlib.pyplot.figure(figsize=(30,10))
        matplotlib.pyplot.plot(range(s), chrono, 'o-', markersize=8)
        #matplotlib.pyplot.plot(range(120), chrono[0:120], 'o-', markersize=8)
        ax = matplotlib.pyplot.gca()
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        matplotlib.pyplot.savefig('%s/chrono_%d.png'%(out_folder, i), dpi=2*96)
        matplotlib.pyplot.close()
        
flag = 0
if flag == 1:
    end_worker = settings.n_workers_reconstruction - 1
    os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s'
              %(end_worker, settings.__name__))    
    
flag = 0
if flag == 1:
    import nlsa.util_merge_x_r    
    for mode in range(20):
        nlsa.util_merge_x_r.f(settings, mode) 
  
flag = 0
if flag == 1:
    
    benchmark = joblib.load('../../synthetic_data_4/test5/x.jbl')
    print benchmark.shape
    benchmark = benchmark[:, settings.q:settings.q+(settings.S-settings.q-settings.ncopies+1)]
    print benchmark.shape
    benchmark = benchmark.flatten()
    
    CCs = []
    
    x_r_tot = 0
    for mode in settings.modes_to_reconstruct:
        print 'Mode:', mode
        x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
        matplotlib.pyplot.imshow(x_r, cmap='jet')
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.savefig('%s/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
        matplotlib.pyplot.close()  
        x_r_tot += x_r
    
        matplotlib.pyplot.imshow(x_r_tot, cmap='jet')
        matplotlib.pyplot.colorbar()
        
        print x_r_tot.shape
               
        x_r_tot_flat = x_r_tot.flatten()
        
        CC = correlate.Correlate(benchmark, x_r_tot_flat)
        print 'CC: ', CC
        
        matplotlib.pyplot.title('%.4f'%CC)
        matplotlib.pyplot.savefig('%s/x_r_tot_%d_modes.png'%(settings.results_path, mode+1), dpi=96*3)
        matplotlib.pyplot.close() 
        
        CCs.append(CC)
    joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)

flag = 0
if flag == 1:
    CCs = joblib.load('%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)
    matplotlib.pyplot.scatter(range(1, len(CCs)+1), CCs, c='b')
    matplotlib.pyplot.xticks(range(1,len(CCs)+1,2))
    matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_nmodes_q_%d.png'%(settings.results_path, settings.q))
    matplotlib.pyplot.close()
  
flag = 0
if flag == 1:
        
    local_linearity_lst = []   
    x_r_tot = 0
    for mode in settings.modes_to_reconstruct:
        print 'mode: ', mode
        x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
        x_r_tot += x_r  
        L = local_linearity.local_linearity_measure(x_r_tot)
        
        local_linearity_lst.append(L)
    joblib.dump(local_linearity_lst, '%s/local_linearity_vs_nmodes.jbl'%settings.results_path)   
    
flag = 0
if flag == 1:
    
    lls = joblib.load('%s/local_linearity_vs_nmodes.jbl'%settings.results_path)
    matplotlib.pyplot.scatter(range(1, len(lls)+1), numpy.log(lls), c='b')
    matplotlib.pyplot.xticks(range(1,len(lls)+1,2))
    matplotlib.pyplot.savefig('%s/local_linearity_vs_nmodes_q_%d.png'%(settings.results_path, settings.q))
    matplotlib.pyplot.close()   

    
# BINNING
flag = 1
if flag == 1:
    x = joblib.load('%s/x.jbl'%settings.results_path)
    print 'x: ', x.shape
    
    benchmark = joblib.load('../../synthetic_data_5/test1/x.jbl')
    print benchmark.shape
    
    bin_sizes = []
    CCs = []
    
    # bin_size = 1
    # CC = correlate.Correlate(benchmark.flatten(), x.flatten())
    # bin_sizes.append(bin_size)
    # CCs.append(CC)
    # print CC
    # print benchmark.shape, x.shape
    
    for n in [600]:#]range(100, 2100, 100):#range(200,2200,200):
        bin_size = 2*n + 1
        x_binned = numpy.zeros((settings.m, settings.S-bin_size+1))
        print x_binned.shape
        for i in range(x_binned.shape[1]):
            #print i
            x_avg = numpy.average(x[:,i:i+bin_size], axis=1)
            x_binned[:,i] = x_avg
            
        CC = correlate.Correlate(benchmark[:, n:-n].flatten(), x_binned.flatten())
        print n, CC
        bin_sizes.append(bin_size)
        CCs.append(CC)
                
        x_large = numpy.zeros((settings.m, settings.S))
        x_large[:] = numpy.nan
        x_large[:, n:-n] = x_binned
        cmap = matplotlib.cm.jet
        cmap.set_bad('white')
        im = matplotlib.pyplot.imshow(x_large, cmap=cmap)
        matplotlib.pyplot.title('Bin size: %d CC: %.4f'%(bin_size, CC))
        ax = matplotlib.pyplot.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)   
        cb = matplotlib.pyplot.colorbar(im, cax=cax)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cb.locator = tick_locator
        cb.update_ticks()
        
        matplotlib.pyplot.savefig('%s/x_r_binsize_%d_jet_nan.png'%(settings.results_path, bin_size), dpi=96*3)
        matplotlib.pyplot.close()  
        
    #joblib.dump(bin_sizes, '%s/binsizes.jbl'%settings.results_path)
    #joblib.dump(CCs, '%s/reconstruction_CC_vs_binsize.jbl'%settings.results_path)
        
flag = 0
if flag == 1:
    bin_sizes = joblib.load('%s/binsizes.jbl'%settings.results_path)
    CCs = joblib.load('%s/reconstruction_CC_vs_binsize.jbl'%settings.results_path)
    matplotlib.pyplot.plot(bin_sizes, CCs, 'o-', c='b')
    matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
    #matplotlib.pyplot.xticks(range(1,len(CCs)+1,2))
    matplotlib.pyplot.xlabel('bin size', fontsize=14)
    matplotlib.pyplot.ylabel('CC', fontsize=14)
    matplotlib.pyplot.xlim(left=bin_sizes[0]-100, right=bin_sizes[-1]+100)
    matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_binsize.pdf'%(settings.results_path), dpi=4*96)
    matplotlib.pyplot.close()    

# # NLSA WITH LOW-PASS FILTERING    
# flag = 0
# if flag == 1:
#     make_lp_filter_functions(settings)
    
# flag = 0
# if flag == 1:
#     end_worker = settings.n_workers_aj - 1
#     os.system('sbatch -p day --array=0-%d ../scripts_parallel_submission/run_parallel_aj.sh %s'
#                   %(end_worker, settings.__name__)) 
    
# flag = 0
# if flag == 1:
#     end_worker = settings.n_workers_aj - 1
#     os.system('sbatch -p hour --array=0-%d ../scripts_parallel_submission/run_parallel_ATA_lp_filter.sh %s'
#               %(end_worker, settings.__name__))   
    
# flag = 0
# if flag == 1:
#     import nlsa.calculate_lp_filter_ATA_merge
#     nlsa.calculate_lp_filter_ATA_merge.main(settings) 
    
# flag = 0
# if flag == 1:
#     end_worker = settings.n_workers_lp_filter_Dsq - 1
#     os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_Dsq_lp_filter.sh %s'
#               %(end_worker, settings.__name__))    
    
# flag = 0
# if flag == 1:  
#     import nlsa.util_merge_D_sq
#     nlsa.util_merge_D_sq.f_lp_filter(settings)  
         
# flag = 0
# if flag == 1:
#     import nlsa.plot_distance_distributions
#     # nlsa.plot_distance_distributions.plot_distributions(settings)
#     #nlsa.plot_distance_distributions.plot_D_matrix(settings)
#     nlsa.plot_distance_distributions.plot_D_0j_lp_filter(settings)
#     #nlsa.plot_distance_distributions.plot_D_0j_recap_lin(settings)
#     #nlsa.plot_distance_distributions.plot_D_submatrix(settings)

# flag = 0
# if flag == 1:    
#     import nlsa.calculate_distances_utilities
#     nlsa.calculate_distances_utilities.sort_D_sq(settings)
    
# flag = 0
# if flag == 1:
#     import nlsa.get_epsilon
#     nlsa.get_epsilon.main(settings)
        
# flag = 0
# if flag == 1:
#     import nlsa.transition_matrix
#     nlsa.transition_matrix.main(settings)
    
# flag = 0
# if flag == 1:
#     import nlsa.probability_matrix
#     nlsa.probability_matrix.main(settings)
    
# flag = 0
# if flag == 1:
#     import nlsa.eigendecompose
#     nlsa.eigendecompose.main(settings)
    
# flag = 0
# if flag == 1:
#     import nlsa.evecs_normalisation
#     nlsa.evecs_normalisation.main(settings)
    
# flag = 0
# if flag == 1:  
#     import nlsa.plot_P_evecs
#     nlsa.plot_P_evecs.main(settings)

# flag = 0
# if flag == 1:
#     end_worker = settings.n_workers_A - 1
#     os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A_lp.sh %s'
#               %(end_worker, settings.__name__)) 
    
# flag = 0
# if flag == 1:
#     import nlsa.util_merge_A
#     nlsa.util_merge_A.main(settings)

# flag = 0
# if flag == 1: 
#     import nlsa.SVD
#     nlsa.SVD.main(settings)
   
# flag = 0
# if flag == 1:  
#     import nlsa.plot_SVs
#     nlsa.plot_SVs.main(settings)    
#     import nlsa.plot_chronos
#     nlsa.plot_chronos.main(settings)
   
# flag = 0
# if flag == 1:
#     end_worker = settings.n_workers_reconstruction - 1
#     os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s'
#               %(end_worker, settings.__name__))    

# flag = 0
# if flag == 1:
#     import nlsa.util_merge_x_r    
#     for mode in range(2):
#         nlsa.util_merge_x_r.f(settings, mode) 
        
# flag = 0
# if flag == 1:
#     x_r_tot = 0
#     for mode in range(2):
#         x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
#         matplotlib.pyplot.imshow(x_r, cmap='jet')
#         matplotlib.pyplot.colorbar()
#         #matplotlib.pyplot.title(r"x_i = A_i*cos(3*w*t) + B_i*sin(10*w*t)")
#         matplotlib.pyplot.savefig('%s/x_r_mode_%d.png'%(settings.results_path, mode), dpi=96*3)
#         matplotlib.pyplot.close()  
#         x_r_tot += x_r
#     matplotlib.pyplot.imshow(x_r_tot, cmap='jet')
#     matplotlib.pyplot.colorbar()
#     matplotlib.pyplot.savefig('%s/x_r_tot.png'%(settings.results_path), dpi=96*3)
#     matplotlib.pyplot.close()  