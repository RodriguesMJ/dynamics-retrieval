# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot
import random
import joblib
import os
import time 
import settings_synthetic_data as settings

# -*- coding: utf-8 -*-
import numpy

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
    
    # min_period = T/10
    # jitter_std_dev = 0.33*min_period
    # ts = ts + numpy.random.normal(scale=jitter_std_dev, size=S)
    
    tc = float(S)/2
    e1 = 1-numpy.exp(-ts/tc)
    e2 = 1-e1
    
    for i in range(m):        
        
        #errors = numpy.random.normal(size=S)
        
        # A_i = numpy.cos(0.6*(2*numpy.pi/m)*i) 
        # B_i = numpy.sin(3*(2*numpy.pi/m)*i+numpy.pi/5) 
        # C_i = numpy.sin(0.8*(2*numpy.pi/m)*i+numpy.pi/7)         
        # x_i = (A_i*numpy.cos(3*omega*ts) + 
        #       B_i*numpy.sin(10*omega*ts) + 
        #       C_i*numpy.sin(4*omega*ts) )
              
        
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
        
        # # C = 5
        # # x_i = C + A_i*(numpy.cos(3*omega*ts)+0.6*numpy.sin(10*omega*ts))
        # # x_i = x_i*partialities
        # # x_i_gauss = numpy.random.normal(loc=x_i, scale=numpy.sqrt(x_i))
        # # x[i,:] = x_i_gauss*sparsities
        # # mask[i,:] = sparsities
        
    matplotlib.pyplot.imshow(x, cmap='jet')
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/x.png'%(results_path), dpi=96*3)
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
    #nlsa.util_merge_D_sq.f(settings)   
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
        
        CC = Correlate(benchmark, x_r_tot_flat)
        print CC
        
        CCs.append(CC)
        
        # matplotlib.pyplot.title('%.4f'%CC)
        # matplotlib.pyplot.savefig('%s/x_r_tot.png'%(settings.results_path), dpi=96*3)
        # matplotlib.pyplot.close() 
        
    joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)

    
#################################
# Cmp
flag = 0
if flag == 1:
    Dsq_1 = joblib.load('../../synthetic_data_4/test1/nlsa/D_sq_parallel.jbl')
    print Dsq_1.shape
    Dsq_2 = joblib.load('%s/D_sq_normalised.jbl'%settings.results_path)
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
        
    D_1 = numpy.sqrt(Dsq_1)
    D_2 = numpy.sqrt(Dsq_2)
        
    print D_1.shape
    print D_2.shape
    
    D_1 = D_1.flatten()
    D_2 = D_2.flatten()
    
    print D_1.shape
    print D_2.shape
    CC = Correlate(D_1,D_2)
    print CC


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
        
        CC = Correlate(benchmark, x_r_tot_flat)
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
        
        CC = Correlate(benchmark, x_r_tot_flat)
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
    print x.shape
    xTx = numpy.matmul(x.T, x)
    joblib.dump(xTx, '%s/xTx.jbl'%settings.results_path)
    print xTx.shape

# Calc XTX
flag = 0
if flag == 1:
    s = settings.S - settings.q    
    XTX = numpy.zeros((s, s))
    xTx = joblib.load('%s/xTx.jbl'%settings.results_path)
    print xTx.shape, XTX.shape
    for i in range(1, settings.q+1):
        print i
        XTX += xTx[i:i+s, i:i+s]
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
    S = numpy.sqrt(evals_XTX)
    VT = evecs_XTX.T
    #U_temp = numpy.matmul(evecs_XtX, numpy.diag(1.0/S))
    #U = numpy.matmul(A, U_temp)     
    #joblib.dump(U, '%s/U.jbl'%settings.results_path)
    joblib.dump(S, '%s/S.jbl'%settings.results_path)
    joblib.dump(VT, '%s/VT.jbl'%settings.results_path)


flag = 0
if flag == 1:  
    import nlsa.plot_SVs
    nlsa.plot_SVs.main(settings)    
    VT_final = joblib.load('%s/VT.jbl'%(settings.results_path))
    
    nmodes = VT_final.shape[0]
    s = VT_final.shape[1]
    
    print 'nmodes: ', nmodes
    print 's: ', s
    
    out_folder = '%s/chronos'%(settings.results_path)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    for i in range(nmodes-6,nmodes):
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