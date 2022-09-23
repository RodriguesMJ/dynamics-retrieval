# -*- coding: utf-8 -*-
import numpy
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot
import math
import joblib
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.pylab

import nlsa.correlate
import nlsa.local_linearity


def make_settings(sett, fn, q, fmax, path, datafile, nmodes):
    fopen = open(fn, 'w')
    fopen.write('# -*- coding: utf-8 -*-\n')
    fopen.write('import numpy\n')
    fopen.write('import math\n')
    fopen.write('S = %d\n'%sett.S)
    fopen.write('m = %d\n'%sett.m)
    fopen.write('q = %d\n'%q)    
    fopen.write('f_max = %d\n'%fmax)
    fopen.write('f_max_considered = f_max\n')
    fopen.write('results_path = "%s"\n'%path)
    fopen.write('paral_step_A = 1000\n')    
    fopen.write('datatype = numpy.float64\n')
    fopen.write('data_file = "%s"\n'%datafile)
    fopen.write('n_workers_A = int(math.ceil(float(q)/paral_step_A))\n')
    fopen.write('p = 0\n')
    fopen.write('modes_to_reconstruct = range(0, %d)\n'%nmodes)
    fopen.close()



def make_lp_filter_functions(settings):
    import nlsa.make_lp_filter
    print 'q: ', settings.q
    
    results_path = settings.results_path
    F = nlsa.make_lp_filter.get_F_sv_t_range(settings)
    Q, R = nlsa.make_lp_filter.on_qr(settings, F)
    #Q = nlsa.make_lp_filter.on_modified(settings, F)
    d = nlsa.make_lp_filter.check_on(Q)
    print 'Normalisation: ', numpy.amax(abs(d))
    
    joblib.dump(Q, '%s/F_on.jbl'%results_path)



def do_SVD(settings):
    import nlsa.SVD
    results_path = settings.results_path

    A = joblib.load('%s/A_parallel.jbl'%results_path)
    print 'Loaded'
    #U, S, VH = nlsa.SVD.SVD_f_manual(A)
    #U, S, VH = nlsa.SVD.sorting(U, S, VH)
    U, S, VH = nlsa.SVD.SVD_f(A)
    for i in S[0:20]:
        print i
    
    
    print 'Done'
    print 'U: ', U.shape
    print 'S: ', S.shape
    print 'VH: ', VH.shape

    joblib.dump(U[:,0:20], '%s/U.jbl'%results_path)
    joblib.dump(S, '%s/S.jbl'%results_path)
    joblib.dump(VH, '%s/VH.jbl'%results_path)
    
    evecs = joblib.load('%s/F_on.jbl'%(results_path))
    Phi = evecs[:,0:2*settings.f_max_considered+1]
    
    VT_final = nlsa.SVD.project_chronos(VH, Phi)    
    print 'VT_final: ', VT_final.shape
    joblib.dump(VT_final, '%s/VT_final.jbl'%results_path)



def get_L(settings):
    results_path = settings.results_path
    p = settings.p
    t_r = joblib.load('%s/t_r_p_%d.jbl'%(results_path, p))
    
    Ls = []   
    x_r_tot = 0
    for mode in settings.modes_to_reconstruct:
        print 'Mode: ', mode
        x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(results_path, p, mode))
        x_r_tot += x_r              
        L = nlsa.local_linearity.local_linearity_measure_jitter(x_r_tot, t_r)
        Ls.append(L)
        
    joblib.dump(Ls, '%s/p_%d_local_linearity_vs_nmodes.jbl'%(results_path, p))   
    
    matplotlib.pyplot.scatter(range(1, len(Ls)+1), numpy.log10(Ls), c='b')
    matplotlib.pyplot.xticks(range(1,len(Ls)+1,2))
    matplotlib.pyplot.savefig('%s/p_%d_log10_L_vs_nmodes_q_%d_fmax_%d.png'%(results_path, p, settings.q, settings.f_max_considered))
    matplotlib.pyplot.close()  



####################
### Make dataset ###
####################

### SPARSE ###

# Data conversion with convert.py
flag = 0
if flag == 1:
    import settings_bR_dark as settings
    import nlsa.convert
    nlsa.convert.main(settings)

flag = 0
if flag == 1:
    import settings_bR_light as settings
    import nlsa.select_frames
    nlsa.select_frames.main(settings)
    
flag = 0
if flag == 1:
    import settings_bR_dark as settings
    import nlsa.boost
    nlsa.boost.main(settings)
 
# Calculate intensity deviations from the mean
flag = 0
if flag == 1:
    import settings_bR_light as settings
    from scipy import sparse
    T = joblib.load('%s/T_bst_sparse_%s.jbl'%(settings.results_path, settings.label))
    M = joblib.load('%s/M_sel_sparse_%s.jbl'%(settings.results_path, settings.label))
    print T.shape, M.shape
    print T[100,30:60]
    ns = numpy.sum(M, axis=1)
    avgs = numpy.sum(T, axis=1)/ns
    print avgs[100,0]
    joblib.dump(avgs, '%s/T_avgs_%s.jbl'%(settings.results_path, settings.label))
    T = T-avgs
    print T[100,58], T[100,57]
    print T.shape
    for i in range(M.shape[1]):
        if i%500 == 0:
            print i
        T[:,i] = numpy.multiply(T[:,i], (M[:,i].todense()))
    print T[100,58], T[100,57]    
    dT_sparse = sparse.csr_matrix(T)
    joblib.dump(dT_sparse, '%s/dT_sparse_%s.jbl'%(settings.results_path, settings.label))

##### MERGE TEST #######
flag = 0
if flag == 1:
    import settings_bR_light as settings
    data_path = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/results_LPSA'
    
    fpath = '%s/binning'%(data_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        
    miller_h = joblib.load('%s/bR_light/miller_h_light.jbl'%(data_path))
    miller_k = joblib.load('%s/bR_light/miller_k_light.jbl'%(data_path))
    miller_l = joblib.load('%s/bR_light/miller_l_light.jbl'%(data_path))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    print out.shape
    
    T  = joblib.load('%s/bR_light/T_sel_sparse_light.jbl'%(data_path))
    M  = joblib.load('%s/bR_light/M_sel_sparse_light.jbl'%(data_path))
    ts = joblib.load('%s/bR_light/ts_sel_light.jbl'%(data_path))
    print T.shape, M.shape, ts.shape
    
    N = 30000
    
    T_early = T[:,0:N]
    M_early = M[:,0:N]
    print 'Early times', ts[0,0], ts[N,0]
    print T_early.shape
    
    T_late = T[:,T.shape[1]-N:]
    M_late = M[:,M.shape[1]-N:]
    print 'Late times', ts[T.shape[1]-N,0], ts[-1,0]
    print T_late.shape
    
    I_early = numpy.sum(T_early, axis=1)
    n_early = numpy.sum(M_early, axis=1)
    I_early = I_early / n_early
    
    I_early[numpy.isnan(I_early)] = 0
    I_early[I_early<0] = 0 
    sigI_early = numpy.sqrt(I_early)
        
    out[:, 3] = I_early.flatten()
    out[:, 4] = sigI_early.flatten()
    
    f_out = '%s/I_early_avg.txt'%(fpath)              
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%17.2f%17.2f')
    
    I_late = numpy.sum(T_late, axis=1)
    n_late = numpy.sum(M_late, axis=1)
    I_late = I_late / n_late
    
    I_late[numpy.isnan(I_late)] = 0
    I_late[I_late<0] = 0 
    sigI_late = numpy.sqrt(I_late)
        
    out[:, 3] = I_late.flatten()
    out[:, 4] = sigI_late.flatten()
    
    f_out = '%s/I_late_avg.txt'%(fpath)              
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%17.2f%17.2f')
    
    miller_h = joblib.load('%s/bR_dark/miller_h_dark.jbl'%(data_path))
    miller_k = joblib.load('%s/bR_dark/miller_k_dark.jbl'%(data_path))
    miller_l = joblib.load('%s/bR_dark/miller_l_dark.jbl'%(data_path))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    print out.shape
    
    T_dark  = joblib.load('%s/bR_dark/T_sparse_dark.jbl'%(data_path))
    M_dark  = joblib.load('%s/bR_dark/M_sparse_dark.jbl'%(data_path))
    print T_dark.shape
    
    I_dark = numpy.sum(T_dark, axis=1)
    n_dark = numpy.sum(M_dark, axis=1)
    I_dark = I_dark / n_dark
    
    I_dark[numpy.isnan(I_dark)] = 0
    I_dark[I_dark<0] = 0 
    sigI_dark = numpy.sqrt(I_dark)
        
    out[:, 3] = I_dark.flatten()
    out[:, 4] = sigI_dark.flatten()
    
    f_out = '%s/I_dark_avg.txt'%(fpath)              
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%17.2f%17.2f')
    
# CHECK INPUT
flag = 0
if flag == 1:
    import joblib
    import settings_bR_light as settings
    hs = joblib.load('%s/miller_h_light.jbl'%settings.results_path)
    print hs.shape
    ks = joblib.load('%s/miller_k_light.jbl'%settings.results_path)
    print ks.shape
    ls = joblib.load('%s/miller_l_light.jbl'%settings.results_path)
    print ls.shape
    bs = joblib.load('%s/boost_factors_light.jbl'%settings.results_path)
    print bs.shape
    T = joblib.load('%s/T_bst_sparse_light.jbl'%settings.results_path)
    print T.shape
    M = joblib.load('%s/M_sel_sparse_light.jbl'%settings.results_path)
    print M.shape
    ts = joblib.load('%s/ts_sel_light.jbl'%settings.results_path)
    print ts.shape
    
    
    
    
    
#################################
### LPSA PARA SEARCH : q-scan ###
#################################

qs = [501, 1001, 2501, 5001, 7501, 10001, 12501, 17501]

flag = 0
if flag == 1:
    import settings_bR_light as settings
    for q in qs:
        
        # MAKE OUTPUT FOLDER
        q_path = '%s/f_max_%d_q_%d'%(settings.results_path, settings.f_max_q_scan, q)
        if not os.path.exists(q_path):
            os.mkdir(q_path)
        
        # MAKE SETTINGS FILE
        data_file = '%s/dT_sparse_%s.jbl'%(q_path, settings.label)
        fn = '/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/workflows/settings_q_%d.py'%q
        make_settings(settings, fn, q, settings.f_max_q_scan, q_path, data_file, 20)

flag = 0
if flag == 1:    
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q, settings.results_path
        make_lp_filter_functions(settings)

flag = 0
if flag == 1:
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        end_worker = settings.n_workers_A - 1
        os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A_fourier.sh %s'
                  %(end_worker, settings.__name__))     
flag = 0
if flag == 1:
    import nlsa.util_merge_A
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        nlsa.util_merge_A.main(settings)   


flag = 0
if flag == 1:
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        end_worker = (2*settings.f_max + 1) - 1
        os.system('sbatch -p day --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_aj.sh %s'
                  %(end_worker, settings.__name__)) 

flag = 0
if flag == 1:
    import nlsa.util_merge_A
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q   
        print 'jmax: ', settings.f_max
        
        results_path = settings.results_path
        datatype = settings.datatype
        m = settings.m
        q = settings.q
        
        A = numpy.zeros(shape=(m*q, 2*settings.f_max+1), dtype=datatype)
        for i in range(2*settings.f_max+1):
            print i
            fn = '%s/aj/a_%d.jbl'%(results_path, i)
            aj = joblib.load(fn)
            A[:,i] = aj
        
        print 'Done.'
         
        print 'A: ', A.shape, A.dtype
        print 'Saving.'
        joblib.dump(A, '%s/A_parallel.jbl'%results_path)
        print 'Done.'
        
     
flag = 0
if flag == 1: 
    
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q   
        print 'jmax: ', settings.f_max
        
        # do_SVD(settings)
        
        end_worker = (2*settings.f_max + 1) - 1
        os.system('sbatch -p day --array=0-%d ../scripts_parallel_submission/run_parallel_ATA.sh %s'
                  %(end_worker, settings.__name__)) 
        
flag = 0
if flag == 1:  
    import nlsa.calculate_ATA_merge
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q   
        print 'jmax: ', settings.f_max
        nlsa.calculate_ATA_merge.main(settings)
        
flag = 0
if flag == 1: 
    print '\n****** RUNNING SVD ******'
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        do_SVD(settings)

flag = 0
if flag == 1: 
            
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        ATA = joblib.load('%s/ATA.jbl'%settings.results_path)[0:2*settings.f_max+1, 0:2*settings.f_max+1]
        print 'A.T A: ', ATA.shape
        
        print 'Eigendecompose'
        evals_ATA, evecs_ATA = numpy.linalg.eigh(ATA)
        print 'Done'
        evals_ATA[numpy.argwhere(evals_ATA<0)]=0  
        S = numpy.sqrt(evals_ATA)
        VH = evecs_ATA.T
        print evecs_ATA[0,0:5]
        print VH[0:5,0]
        
        joblib.dump(S,  '%s/S_unsorted.jbl'%settings.results_path)
        joblib.dump(VH, '%s/VH_unsorted.jbl'%settings.results_path)

        # A = joblib.load('%s/A_parallel.jbl'%settings.results_path)[:, 0:2*settings.f_max+1]        
        # print 'A:', A.shape
        # U_temp = numpy.matmul(evecs_ATA, numpy.diag(1.0/S))
        # U = numpy.matmul(A, U_temp)             
        # joblib.dump(U, '%s/U_unsorted.jbl'%settings.results_path)

def get_Ai(settings, i, step):
    A = joblib.load('%s/A_parallel.jbl'%settings.results_path)[:, 0:2*settings.f_max+1]        
    print 'A:', A.shape
    start = i*step
    end = min([(i+1)*step, settings.m*settings.q])
    print start, end
    joblib.dump(A[start:end, :], '%s/A_%d.jbl'%(settings.results_path, i))

n_chuncks = 2

flag = 0
if flag == 1: 
            
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        
        step = int(math.ceil(float(settings.m * settings.q)/n_chuncks))
        for i in range(0, n_chuncks):
            get_Ai(settings, i, step)
            
flag = 0
if flag == 1: 
            
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        
        S = joblib.load('%s/S_unsorted.jbl'%settings.results_path)
        VH = joblib.load('%s/VH_unsorted.jbl'%settings.results_path)
        evecs_ATA = VH.T
        U_temp = numpy.matmul(evecs_ATA, numpy.diag(1.0/S))
        
        for i in range(0, n_chuncks):           
            A_i = joblib.load('%s/A_%d.jbl'%(settings.results_path,i))
            print 'A_i:', A_i.shape
            U_i = numpy.matmul(A_i, U_temp)  
            joblib.dump(U_i, '%s/U_%d.jbl'%(settings.results_path, i))
            
flag = 0
if flag == 1:
    import nlsa.SVD
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q 
        
        S  = joblib.load('%s/S_unsorted.jbl'%settings.results_path)
        VH = joblib.load('%s/VH_unsorted.jbl'%settings.results_path)
        
        #U  = joblib.load('%s/U_unsorted.jbl'%settings.results_path)
        #U, S, VH = nlsa.SVD.sorting(U, S, VH)
        
        sort_idxs = numpy.argsort(S)[::-1]
        S = S[sort_idxs]
        VH = VH[sort_idxs,:]
        
        for i in S[0:20]:
            print i   
        
        print 'S: ', S.shape
        print 'VH: ', VH.shape

        #joblib.dump(U[:,0:20], '%s/U.jbl'%settings.results_path)
        joblib.dump(S, '%s/S.jbl'%settings.results_path)
        joblib.dump(VH, '%s/VH.jbl'%settings.results_path)
                
        for i in range(0, n_chuncks):
            U_i = joblib.load('%s/U_%d.jbl'%(settings.results_path, i))
            U_i_sorted = U_i[:,sort_idxs[0:20]]
            joblib.dump(U_i_sorted, '%s/U_%d_sorted.jbl'%(settings.results_path, i))
        
        evecs = joblib.load('%s/F_on.jbl'%(settings.results_path))
        Phi = evecs[:,0:2*settings.f_max_considered+1]
        
        VT_final = nlsa.SVD.project_chronos(VH, Phi)    
        print 'VT_final: ', VT_final.shape
        joblib.dump(VT_final, '%s/VT_final.jbl'%settings.results_path)

    
flag = 0
if flag == 1:
    for q in [17501]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q       
        print 'jmax: ', settings.f_max
        
        results_path = settings.results_path
        datatype = settings.datatype
        
        U = numpy.zeros(shape=(settings.m*settings.q, 20), dtype=datatype)
        step = int(math.ceil(float(settings.m * settings.q)/n_chuncks))
        for i in range(0, n_chuncks):
            start = i*step
            end = min([(i+1)*step, settings.m*settings.q])
            U_i = joblib.load('%s/U_%d_sorted.jbl'%(results_path, i))
            U[start:end,:] = U_i
            
        print 'Done.'
        print 'Saving.'
        joblib.dump(U, '%s/U.jbl'%results_path)
        print 'Done.'


flag = 0
if flag == 1:  
    import nlsa.plot_SVs
    import nlsa.plot_chronos    
    for q in [17501]:#qs:        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        nlsa.plot_SVs.main(settings)       
        nlsa.plot_chronos.main(settings)

qs = [17501]
# p=0 reconstruction 
flag = 1
if flag == 1:
    import nlsa.reconstruct_p
    for q in qs:        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        nlsa.reconstruct_p.f(settings)
        nlsa.reconstruct_p.f_ts(settings)      

# Calculate L of p=0 reconstructed signal (ie L of central block of reconstructed supervectors)   
flag = 1
if flag == 1:
    for q in qs:        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        get_L(settings)
        
flag = 1
if flag == 1:
    for q in qs:        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q   
        print 'jmax: ', settings.f_max    
        results_path = settings.results_path
        x_r_tot = 0
        for mode in range(0,5):#range(0, 11):
            print 'Mode: ', mode
            x_r = joblib.load('%s/movie_p_0_mode_%d.jbl'%(results_path, mode))
            x_r_tot += x_r  
            if mode == 4: #(mode == 5 or mode == 10):
                joblib.dump(x_r_tot, '%s/movie_p_0_sum_%d_modes.jbl'%(results_path, mode+1))
        
flag = 0
if flag == 1:    
    #os.system('python export_Is.py')       
    for q in qs:        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q     
        
        results_path = settings.results_path
        p = settings.p
        t_r = joblib.load('%s/t_r_p_%d.jbl'%(results_path, p))
        print 'p: ', settings.p
        
        nmodes = 6
        
        x_r_tot = 0
        for mode in range(nmodes):
            print 'Mode: ', mode
            x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(results_path, p, mode))
            x_r_tot += x_r              
        
        # DEBOOST
        # n_obs = joblib.load('%s/boost_factors_light.jbl'%(results_path))
        # n_obs_rec = 1.0/n_obs
        # print n_obs[0:5]
        # print n_obs_rec[0:5]
        # print n_obs_rec.shape
        # x_r_debst = x_r_tot / n_obs_rec
    
        joblib.dump(x_r_tot, '%s/x_r_p_%d_sum_%d_modes_boosted.jbl'%(results_path, p, nmodes))   
        print x_r_tot.shape
        
        miller_h = joblib.load('%s/miller_h_light.jbl'%(results_path))
        miller_k = joblib.load('%s/miller_k_light.jbl'%(results_path))
        miller_l = joblib.load('%s/miller_l_light.jbl'%(results_path))
        
        out = numpy.zeros((miller_h.shape[0], 5))
        out[:, 0] = miller_h.flatten()
        out[:, 1] = miller_k.flatten()
        out[:, 2] = miller_l.flatten()
        
        fpath = '%s/extracted_Is_bst_p_%d_%d_modes'%(results_path, p, nmodes)
        if not os.path.exists(fpath):
            os.mkdir(fpath)
            
        for i in range(0, x_r_tot.shape[1], 100):   
        
            f_out = '%s/bR_light_p_%d_%d_modes_timestep_%0.6d.txt'%(fpath, 
                                                                    p, 
                                                                    nmodes, 
                                                                    i)    
                    
            I = x_r_tot[:, i]
        
            I[numpy.isnan(I)] = 0
            I[I<0] = 0 
            #I = mult_factor * I 
            sigIs = numpy.sqrt(I)
            
            out[:, 3] = I
            out[:, 4] = sigIs
            
            numpy.savetxt(f_out, out, fmt='%6d%6d%6d%20.2f%16.2f')


flag = 0
if flag == 1:
    
    import settings_bR_dark as settings
    miller_h = joblib.load('%s/miller_h_dark.jbl'%(settings.results_path))
    miller_k = joblib.load('%s/miller_k_dark.jbl'%(settings.results_path))
    miller_l = joblib.load('%s/miller_l_dark.jbl'%(settings.results_path))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    print out.shape
    
    T_dark  = joblib.load('%s/T_bst_sparse_dark.jbl'%(settings.results_path))
    M_dark  = joblib.load('%s/M_sparse_dark.jbl'%(settings.results_path))
    print T_dark.shape
    
    I_dark = numpy.sum(T_dark, axis=1)
    n_dark = numpy.sum(M_dark, axis=1)
    I_dark = I_dark / n_dark
    
    I_dark[numpy.isnan(I_dark)] = 0
    I_dark[I_dark<0] = 0 
    sigI_dark = numpy.sqrt(I_dark)
        
    out[:, 3] = I_dark.flatten()
    out[:, 4] = sigI_dark.flatten()
    
    f_out = '%s/I_dark_bst_avg.txt'%(settings.results_path)              
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%17.2f%17.2f')
        
            
        

# Calculate step-wise CC in the sequence of reconstructed signal with progressively larger q.   
qs = [501, 1001, 2501, 5001, 7501, 10001, 12501, 15001, 17501]
flag = 0
if flag == 1:
    
    def get_x_r_large(q, nmodes):
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        results_path = settings.results_path
        #x_r_tot = 0
        p = 0
        # for mode in range(nmodes):
        #     print 'Mode: ', mode
        #     x_r = joblib.load('%s/reconstruction_p_%d/movie_p_0_mode_%d.jbl'%(results_path, p, mode))
        #     x_r_tot += x_r    
        x_r_tot = joblib.load('%s/reconstruction_p_%d/movie_p_0_sum_%d_modes.jbl'%(results_path, p, nmodes))
        # Since p=0
        x_r_tot_large = numpy.zeros((settings.m, settings.S))
        x_r_tot_large[:,(q-1)/2:(q-1)/2+(settings.S-q+1)] = x_r_tot
        return x_r_tot_large 
    
    import settings_bR_light as settings   
    import nlsa.correlate as correlate
    n_modes = 6
    
    f_max = settings.f_max_q_scan
    
    start_large = get_x_r_large(qs[0], n_modes)
    
    q_largest = qs[-1]
    CCs = []
    for q in qs[1:]:
        x_r_tot_large = get_x_r_large(q, n_modes)
        start_flat   =   start_large[:,(q_largest-1)/2:(q_largest-1)/2+(settings.S-q_largest+1)].flatten()
        x_r_tot_flat = x_r_tot_large[:,(q_largest-1)/2:(q_largest-1)/2+(settings.S-q_largest+1)].flatten()
        print start_flat.shape, x_r_tot_flat.shape
        CC = correlate.Correlate(start_flat, x_r_tot_flat)
        CCs.append(CC)
        print CC
        start_large = x_r_tot_large
    joblib.dump(CCs, '%s/LPSA_f_max_%d_q_scan_p_0_%d_modes_successive_CCs.jbl'%(settings.results_path, f_max, n_modes)) 

    CCs = joblib.load('%s/LPSA_f_max_%d_q_scan_p_0_%d_modes_successive_CCs.jbl'%(settings.results_path, f_max, n_modes))    
    n_curves = len(CCs)
    colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
    matplotlib.pyplot.xticks(range(1,len(CCs)+1,1))   
    for i, CC in enumerate(CCs):
        matplotlib.pyplot.scatter(i+1, CCs[i], c=colors[i], label='q=%d,q=%d'%(qs[i], qs[i+1]))
    matplotlib.pyplot.legend(frameon=True, scatterpoints=1, loc='upper right', fontsize=6) 
    matplotlib.pyplot.savefig('%s/LPSA_f_max_%d_q_scan_p_0_%d_modes_successive_CCs.png'%(settings.results_path, f_max, n_modes), dpi=96*4)
    matplotlib.pyplot.close()


####################################
### LPSA PARA SEARCH : jmax-scan ###
####################################

f_max_s = range(51,61)
    
flag = 0
if flag == 1:
    import settings_bR_light as settings
    for f_max in f_max_s:
           
        # MAKE OUTPUT FOLDER
        f_max_path = '%s/f_max_%d_q_%d'%(settings.results_path, f_max, settings.q_f_max_scan)
        if not os.path.exists(f_max_path):
            os.mkdir(f_max_path)
            
        # MAKE SETTINGS FILE
        data_file = '%s/dT_sparse_%s.jbl'%(f_max_path, settings.label)
        fn = '/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/workflows/settings_f_max_%d.py'%f_max
        make_settings(settings, fn, settings.q_f_max_scan, f_max, f_max_path, data_file, min(20, 2*f_max+1))
        

flag = 0
if flag == 1:    
    for f_max in [56, 57, 58, 59]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)
        print 'jmax: ', settings.f_max, settings.results_path        
        make_lp_filter_functions(settings)


flag = 0
if flag == 1:
    for f_max in [60]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        # end_worker = settings.n_workers_A - 1
        # os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A_fourier.sh %s'
        #           %(end_worker, settings.__name__)) 
        end_worker = (2*settings.f_max + 1) - 1
        os.system('sbatch -p day --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_aj.sh %s'
                  %(end_worker, settings.__name__)) 

flag = 0
if flag == 1:
    import nlsa.util_merge_A
    for f_max in [60]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        
        results_path = settings.results_path
        datatype = settings.datatype
        m = settings.m
        q = settings.q
        
        A = numpy.zeros(shape=(m*q, 2*f_max+1), dtype=datatype)
        for i in range(2*f_max+1):
            print i
            fn = '%s/aj/a_%d.jbl'%(results_path, i)
            aj = joblib.load(fn)
            A[:,i] = aj
        
        print 'Done.'
         
        print 'A: ', A.shape, A.dtype
        print 'Saving.'
        joblib.dump(A, '%s/A_parallel.jbl'%results_path)
        print 'Done.'
        
     
flag = 0
if flag == 1: 
    print '\n****** RUNNING SVD ******'
    for f_max in [60]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        
        # do_SVD(settings)
        
        end_worker = (2*settings.f_max + 1) - 1
        os.system('sbatch -p day --array=0-%d ../scripts_parallel_submission/run_parallel_ATA.sh %s'
                  %(end_worker, settings.__name__)) 
        
flag = 0
if flag == 1:  
    import nlsa.calculate_ATA_merge
    for f_max in [60]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        nlsa.calculate_ATA_merge.main(settings)
        
flag = 0
if flag == 1: 
            
    for f_max in [59]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)  
        
        ATA = joblib.load('%s/ATA.jbl'%settings.results_path)[0:2*f_max+1, 0:2*f_max+1]
        print 'A.T A: ', ATA.shape
        
        print 'Eigendecompose'
        evals_ATA, evecs_ATA = numpy.linalg.eigh(ATA)
        print 'Done'
        evals_ATA[numpy.argwhere(evals_ATA<0)]=0  
        S = numpy.sqrt(evals_ATA)
        VH = evecs_ATA.T
        print evecs_ATA[0,0:5]
        print VH[0:5,0]
        
        joblib.dump(S,  '%s/S_unsorted.jbl'%settings.results_path)
        joblib.dump(VH, '%s/VH_unsorted.jbl'%settings.results_path)

        # A = joblib.load('%s/A_parallel.jbl'%settings.results_path)[:, 0:2*f_max+1]        
        # print 'A:', A.shape
        # U_temp = numpy.matmul(evecs_ATA, numpy.diag(1.0/S))
        # U = numpy.matmul(A, U_temp)             
        # joblib.dump(U, '%s/U_unsorted.jbl'%settings.results_path)

def get_Ai(settings, i, step):
    A = joblib.load('%s/A_parallel.jbl'%settings.results_path)[:, 0:2*settings.f_max+1]        
    print 'A:', A.shape
    start = i*step
    end = min([(i+1)*step, settings.m*settings.q])
    print start, end
    joblib.dump(A[start:end, :], '%s/A_%d.jbl'%(settings.results_path, i))

n_chuncks = 2

flag = 0
if flag == 1: 
            
    for f_max in [59]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)  
        
        step = int(math.ceil(float(settings.m * settings.q)/n_chuncks))
        for i in range(0, n_chuncks):
            get_Ai(settings, i, step)

                    
# flag = 0
# if flag == 1: 
            
#     for f_max in [51]:
#         modulename = 'settings_f_max_%d'%f_max
#         settings = __import__(modulename)  
    
#         A = joblib.load('%s/A_parallel.jbl'%settings.results_path)[:, 0:2*f_max+1]        
#         print 'A:', A.shape
        
#         step = int(math.ceil(float(settings.m * settings.q)/2))
#         for i in [0]:
#             start = i*step
#             end = min([(i+1)*step, settings.m*settings.q])
#             print start, end
#             joblib.dump(A[start:end, :], '%s/A_%d.jbl'%(settings.results_path, i))
            
# flag = 0
# if flag == 1: 
            
#     for f_max in [51]:
#         modulename = 'settings_f_max_%d'%f_max
#         settings = __import__(modulename)  
    
#         A = joblib.load('%s/A_parallel.jbl'%settings.results_path)[:, 0:2*f_max+1]        
#         print 'A:', A.shape
        
#         step = int(math.ceil(float(settings.m * settings.q)/2))
#         for i in [1]:
#             start = i*step
#             end = min([(i+1)*step, settings.m*settings.q])
#             print start, end
#             joblib.dump(A[start:end, :], '%s/A_%d.jbl'%(settings.results_path, i))
 
flag = 0
if flag == 1: 
            
    for f_max in [59]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)  
        
        S = joblib.load('%s/S_unsorted.jbl'%settings.results_path)
        VH = joblib.load('%s/VH_unsorted.jbl'%settings.results_path)
        evecs_ATA = VH.T
        U_temp = numpy.matmul(evecs_ATA, numpy.diag(1.0/S))
        
        for i in [1]:#range(0, n_chuncks):           
            A_i = joblib.load('%s/A_%d.jbl'%(settings.results_path,i))
            print 'A_i:', A_i.shape
            U_i = numpy.matmul(A_i, U_temp)  
            joblib.dump(U_i, '%s/U_%d.jbl'%(settings.results_path, i))
            
flag = 0
if flag == 1:
    for f_max in [55]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        
        results_path = settings.results_path
        datatype = settings.datatype
        
        U = numpy.zeros(shape=(settings.m*settings.q, 2*f_max+1), dtype=datatype)
        step = int(math.ceil(float(settings.m * settings.q)/n_chuncks))
        for i in range(0, n_chuncks):
            start = i*step
            end = min([(i+1)*step, settings.m*settings.q])
            U_i = joblib.load('%s/U_%d.jbl'%(results_path, i))
            U[start:end,:] = U_i
            
        print 'Done.'
        print 'Saving.'
        joblib.dump(U, '%s/U_unsorted.jbl'%results_path)
        print 'Done.'

f_max_s = [60]
                
flag = 0
if flag == 1:
    import nlsa.SVD
    for f_max in f_max_s:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        
        S  = joblib.load('%s/S_unsorted.jbl'%settings.results_path)
        VH = joblib.load('%s/VH_unsorted.jbl'%settings.results_path)
        
        #U  = joblib.load('%s/U_unsorted.jbl'%settings.results_path)
        #U, S, VH = nlsa.SVD.sorting(U, S, VH)
        
        sort_idxs = numpy.argsort(S)[::-1]
        S = S[sort_idxs]
        VH = VH[sort_idxs,:]
        
        for i in S[0:20]:
            print i   
        
        print 'S: ', S.shape
        print 'VH: ', VH.shape

        #joblib.dump(U[:,0:20], '%s/U.jbl'%settings.results_path)
        joblib.dump(S, '%s/S.jbl'%settings.results_path)
        joblib.dump(VH, '%s/VH.jbl'%settings.results_path)
                
        for i in range(0, n_chuncks):
            U_i = joblib.load('%s/U_%d.jbl'%(settings.results_path, i))
            U_i_sorted = U_i[:,sort_idxs[0:20]]
            joblib.dump(U_i_sorted, '%s/U_%d_sorted.jbl'%(settings.results_path, i))
        
        evecs = joblib.load('%s/F_on.jbl'%(settings.results_path))
        Phi = evecs[:,0:2*settings.f_max_considered+1]
        
        VT_final = nlsa.SVD.project_chronos(VH, Phi)    
        print 'VT_final: ', VT_final.shape
        joblib.dump(VT_final, '%s/VT_final.jbl'%settings.results_path)
    

flag = 0
if flag == 1:
    for f_max in [60]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        
        results_path = settings.results_path
        datatype = settings.datatype
        
        U = numpy.zeros(shape=(settings.m*settings.q, 20), dtype=datatype)
        step = int(math.ceil(float(settings.m * settings.q)/n_chuncks))
        for i in range(0, n_chuncks):
            start = i*step
            end = min([(i+1)*step, settings.m*settings.q])
            U_i = joblib.load('%s/U_%d_sorted.jbl'%(results_path, i))
            U[start:end,:] = U_i
            
        print 'Done.'
        print 'Saving.'
        joblib.dump(U, '%s/U.jbl'%results_path)
        print 'Done.'

            
flag = 0
if flag == 1:  
    import nlsa.plot_SVs
    import nlsa.plot_chronos
    for f_max in [60]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        nlsa.plot_SVs.main(settings)       
        nlsa.plot_chronos.main(settings)
      
# p=0 reconstruction 
flag = 0
if flag == 1:
    import nlsa.reconstruct_p
    for f_max in f_max_s:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        nlsa.reconstruct_p.f(settings)
        nlsa.reconstruct_p.f_ts(settings)              

# Calculate L of p=0 reconstructed signal (ie L of central block of reconstructed supervectors)   
flag = 0
if flag == 1:
    for f_max in f_max_s:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max     
        get_L(settings)
        
flag = 0
if flag == 1:
    for f_max in f_max_s:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max    
        results_path = settings.results_path
        x_r_tot = 0
        for mode in range(0, 11):
            print 'Mode: ', mode
            x_r = joblib.load('%s/movie_p_0_mode_%d.jbl'%(results_path, mode))
            x_r_tot += x_r  
            if (mode == 5 or mode == 10):
                joblib.dump(x_r_tot, '%s/movie_p_0_sum_%d_modes.jbl'%(results_path, mode+1))
                        
# CCs of central block with increasing fmax  
flag = 0
if flag == 1:
    import nlsa.correlate
    n_modes_lst = [6,11]
    for n_modes in n_modes_lst:
        f_max_s = range(10, 61) #range(int(math.floor(float(n_modes)/2)), 51)
    
        def get_x_r(fmax, nmodes):
            modulename = 'settings_f_max_%d'%fmax
            settings = __import__(modulename)        
            print 'jmax: ', settings.f_max
            results_path = settings.results_path
            # x_r_tot = 0
            # for mode in range(0, min(nmodes, 2*fmax+1)):
            #     print 'Mode: ', mode
            #     x_r = joblib.load('%s/reconstruction_p_0/movie_p_0_mode_%d.jbl'%(results_path, mode))
            #     x_r_tot += x_r  
            x_r_tot = joblib.load('%s/reconstruction_p_0/movie_p_0_sum_%d_modes.jbl'%(results_path, nmodes))
            return x_r_tot
        
        import settings_bR_light as settings  
    
        start = get_x_r(f_max_s[0], n_modes)        
        CCs = []
        for f_max in f_max_s[1:]:
            x_r_tot = get_x_r(f_max, n_modes)
            start_flat   =   start.flatten()
            x_r_tot_flat = x_r_tot.flatten()
            CC = nlsa.correlate.Correlate(start_flat, x_r_tot_flat)
            CCs.append(CC)
            print CC
            start = x_r_tot        
        joblib.dump(CCs, '%s/LPSA_f_max_scan_step_one_q_15001_p_0_%d_modes_successive_CCs.jbl'%(settings.results_path, n_modes))



flag = 0
if flag == 1:
    import settings_bR_light as settings
    n_modes_lst = [6,11]
    for n_modes in n_modes_lst:
        f_max_s = range(10,61) #range(int(math.floor(float(n_modes)/2)), 51)
   
        CCs = joblib.load('%s/LPSA_f_max_scan_step_one_q_15001_p_0_%d_modes_successive_CCs.jbl'%(settings.results_path, n_modes))
        n_curves = len(CCs)
        colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
        #matplotlib.pyplot.xticks(range(1,len(CCs)+1,1))
        for i, CC in enumerate(CCs):
            matplotlib.pyplot.scatter(i+f_max_s[0], CCs[i], c='b', label='$j_{\mathrm{max}}=$%d, $j_{\mathrm{max}}=$%d'%(f_max_s[i], f_max_s[i+1]))
        #matplotlib.pyplot.legend(frameon=True, scatterpoints=1, loc='lower right', fontsize=10) 
        matplotlib.pyplot.axhline(y=1.0,c='k')
        matplotlib.pyplot.gca().set_xlim([9, 61])
        matplotlib.pyplot.gca().set_ylim([0.750, 1.002])
        matplotlib.pyplot.savefig('%s/LPSA_f_max_scan_step_one_q_15001_p_0_%d_modes_successive_CCs.png'%(settings.results_path, n_modes))
        matplotlib.pyplot.close()

""" 
###############################
### Standard reconstruction ###
###############################
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    end_worker = settings.n_workers_reconstruction - 1
    os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s'
              %(end_worker, settings.__name__))    
 
# TIME ASSIGNMENTwith p=(q-1)/2
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    import nlsa.reconstruct_p
    nlsa.reconstruct_p.f_ts(settings)
      
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    import nlsa.util_merge_x_r    
    for mode in settings.modes_to_reconstruct:
        nlsa.util_merge_x_r.f(settings, mode) 


#######################################################
### p-dependendent reconstruction (avg 2p+1 copies) ###
#######################################################
    
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    import nlsa.reconstruct_p
    nlsa.reconstruct_p.f(settings)   
    nlsa.reconstruct_p.f_ts(settings)

# SVD of result
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    n_modes_to_use = 4
    x_r_tot = 0
    for mode in range(n_modes_to_use):
        print 'Mode:', mode
        #x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(settings.results_path, settings.p, mode))
        x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
        x_r_tot += x_r
    joblib.dump(x_r_tot, '%s/x_r_tot_p_%d_%d_modes.jbl'%(settings.results_path, settings.p, n_modes_to_use))
        
flag = 0
if flag == 1: 
    import settings_synthetic_data_jitter as settings
    import nlsa.SVD
    n_modes_to_use = 4
    x = joblib.load('%s/x_r_tot_p_%d_%d_modes.jbl'%(settings.results_path, settings.p, n_modes_to_use))
    U, S, VH = nlsa.SVD.SVD_f(x)
    print 'Sorting'
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
    import settings_synthetic_data_jitter as settings
    import nlsa.plot_SVs
    nlsa.plot_SVs.main(settings)    
    import nlsa.plot_chronos
    nlsa.plot_chronos.main(settings)    
    
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    t_r = joblib.load('%s/t_r_p_%d.jbl'%(settings.results_path, settings.p))
    print t_r[0], t_r[-1]
    print t_r.shape
    start_idx = (settings.S - t_r.shape[0])/2
    end_idx = start_idx + t_r.shape[0]
    print start_idx, end_idx
    
    benchmark = eval_model(settings, t_r)
    print 'Benchmark: ', benchmark.shape
    plot_signal(benchmark, '%s/benchmark_at_t_r.png'%(settings.results_path))
    benchmark = benchmark.flatten()
    
    U = joblib.load('%s/U.jbl'%settings.results_path)
    S = joblib.load('%s/S.jbl'%settings.results_path)
    VH = joblib.load('%s/VT_final.jbl'%settings.results_path)
    
    x_r_tot = 0
    CCs = []
    for mode in range(4):
        print 'Mode: ', mode
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]
      
        x_r = sv*numpy.outer(u, vT)
        print 'x_r: ', x_r.shape
        plot_signal(x_r, '%s/x_r_mode_%d.png'%(settings.results_path, mode))
        
        x_r_tot += x_r      
        x_r_tot_flat = x_r_tot.flatten()
        
        CC = correlate.Correlate(benchmark, x_r_tot_flat)
        print 'CC: ', CC
        CCs.append(CC)
        
        plot_signal(x_r_tot, '%s/x_r_tot_%d_modes.png'%(settings.results_path, mode+1), '%.4f'%CC)
        
        x_r_large = numpy.zeros((settings.m, settings.S))
        x_r_large[:] = numpy.nan
        x_r_large[:, start_idx:end_idx] = x_r_tot
        cmap = matplotlib.cm.jet
        cmap.set_bad('white')
        im = matplotlib.pyplot.imshow(x_r_large, cmap=cmap)
        ax = matplotlib.pyplot.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)   
        cb = matplotlib.pyplot.colorbar(im, cax=cax)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cb.locator = tick_locator
        cb.update_ticks()  
        matplotlib.pyplot.savefig('%s/x_r_tot_%d_modes_jet_nan.png'%(settings.results_path, mode+1), dpi=96*3)
        matplotlib.pyplot.close() 
           
    # joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)

    # matplotlib.pyplot.scatter(range(1, len(CCs)+1), CCs, c='b')
    # matplotlib.pyplot.xticks(range(1,len(CCs)+1,2))
    # matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_nmodes.png'%(settings.results_path))
    # matplotlib.pyplot.close()
    

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings  
    
    U = joblib.load('%s/U.jbl'%settings.results_path)
    S = joblib.load('%s/S.jbl'%settings.results_path)
    VH = joblib.load('%s/VT_final.jbl'%settings.results_path)
    t_r = joblib.load('%s/t_r_p_%d.jbl'%(settings.results_path, settings.p))
    
    x_r_tot = 0
    Ls = []       
    for mode in range(20):
        print 'Mode: ', mode
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]
        x_r = sv*numpy.outer(u, vT)
        print 'x_r: ', x_r.shape
    
        x_r_tot += x_r  
        L = local_linearity.local_linearity_measure_jitter(x_r_tot, t_r)
        
        Ls.append(L)
    joblib.dump(Ls, '%s/local_linearity_vs_nmodes.jbl'%settings.results_path)   
    
    matplotlib.pyplot.scatter(range(1, len(Ls)+1), numpy.log10(Ls), c='b')
    matplotlib.pyplot.xticks(range(1,len(Ls)+1,2))
    matplotlib.pyplot.savefig('%s/local_linearity_vs_nmodes.png'%(settings.results_path))
    matplotlib.pyplot.close()   


###################
###   BINNING   ### 
###################

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    x = joblib.load('%s/x.jbl'%settings.results_path)
    print 'x: ', x.shape
    
    ts_meas = joblib.load('%s/ts_meas.jbl'%(settings.results_path))
    print 'ts_meas: ', ts_meas.shape, ts_meas.dtype
    
    bin_sizes = []
    CCs = []
    
    bin_size = 1
    CC = correlate.Correlate(eval_model(settings, ts_meas).flatten(), x.flatten())
    bin_sizes.append(bin_size)
    CCs.append(CC)
    print 'Starting CC: ', CC
    
    for n in range(100,2100,100):
        bin_size = 2*n + 1
        x_binned = numpy.zeros((settings.m, settings.S-bin_size+1))
        ts_binned = []
        for i in range(x_binned.shape[1]):
            x_avg = numpy.average(x[:,i:i+bin_size], axis=1)
            x_binned[:,i] = x_avg
            
            t_binned = numpy.average(ts_meas[i:i+bin_size])           
            ts_binned.append(t_binned)
            
        ts_binned = numpy.asarray(ts_binned)
        
        CC = correlate.Correlate(eval_model(settings, ts_binned).flatten(), x_binned.flatten())
        print n, CC
        bin_sizes.append(bin_size)
        CCs.append(CC)
        
        plot_signal(x_binned, '%s/x_r_binsize_%d_jet_nan.png'%(settings.results_path, bin_size), 'Bin size: %d CC: %.4f'%(bin_size, CC)) 
        
    joblib.dump(bin_sizes, '%s/binsizes.jbl'%settings.results_path)
    joblib.dump(CCs, '%s/reconstruction_CC_vs_binsize.jbl'%settings.results_path)
    
    matplotlib.pyplot.plot(bin_sizes, CCs, 'o-', c='b')
    matplotlib.pyplot.axhline(y=1, xmin=0, xmax=1, c='k', linewidth=1)
    matplotlib.pyplot.xlabel('bin size', fontsize=14)
    matplotlib.pyplot.ylabel('CC', fontsize=14)
    matplotlib.pyplot.xlim(left=bin_sizes[0]-100, right=bin_sizes[-1]+100)
    matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_binsize.pdf'%(settings.results_path), dpi=4*96)
    matplotlib.pyplot.close()    


################
###   NLSA   ###
################

flag = 0
if flag == 1:    
    import nlsa.calculate_distances_utilities
    import settings_synthetic_data_jitter as settings
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
    import settings_synthetic_data_jitter as settings
    d_sq = joblib.load('%s/d_sq.jbl'%(settings.results_path))
    print numpy.amax(d_sq), numpy.amin(d_sq)
    print numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq))

flag = 0
if flag == 1:
    qs = [1, 51, 101, 501, 1001, 2001]  
    for q in qs:        
        b = 3000
        log10eps = 1.0
        l = 50
        p = (q-1)/2
        
        root_f = "../../synthetic_data_jitter/test6"
        results_path = '%s/NLSA/q_%d'%(root_f, q)
        if not os.path.exists(results_path):
            os.mkdir(results_path)
            
        fopen = open('./settings_q_%d.py'%q, 'w')
        fopen.write('# -*- coding: utf-8 -*-\n')
        fopen.write('import numpy\n')
        fopen.write('import math\n')
        fopen.write('S = 30000\n')
        fopen.write('m = 7000\n')
        fopen.write('q = %d\n'%q) 
        fopen.write('T_model = 26000\n')#S-q+1
        fopen.write('tc = float(S)/2\n')
        fopen.write('paral_step = 100\n')
        fopen.write('n_workers = int(math.ceil(float(q)/paral_step))\n')
        fopen.write('b = %d\n'%b)  
        fopen.write('log10eps = %0.1f\n'%log10eps)
        fopen.write('sigma_sq = 2*10**log10eps\n')
        fopen.write('l = %d\n'%l)
        fopen.write('nmodes = l\n') 
        fopen.write('toproject = range(nmodes)\n') 
        fopen.write('results_path = "%s"\n'%results_path)
        fopen.write('paral_step_A = 50\n')    
        fopen.write('datatype = numpy.float64\n')
        fopen.write('data_file = "%s/x.jbl"\n'%results_path)
        fopen.write('n_workers_A = int(math.ceil(float(q)/paral_step_A))\n')
        fopen.write('p = %d\n'%p)
        fopen.write('modes_to_reconstruct = range(min(l, 20))\n')
        fopen.close() 
    
flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings
    qs = [1, 51, 101, 501, 1001, 2001]  
    for q in qs:  
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)   
        end_worker = settings.n_workers - 1
        os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel.sh %s'
                  %(end_worker, settings.__name__)) 
        os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_n_Dsq_elements.sh %s'
                  %(end_worker, settings.__name__)) 
        
flag = 0
if flag == 1: 
    #import settings_synthetic_data_jitter as settings
    import nlsa.util_merge_D_sq
    import nlsa.calculate_distances_utilities
    qs = [1, 51, 101, 501, 1001, 2001]  
    for q in qs:  
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)   
        nlsa.util_merge_D_sq.f(settings)   
        nlsa.util_merge_D_sq.f_N_D_sq_elements(settings)          
        nlsa.calculate_distances_utilities.normalise(settings)
    
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    import nlsa.plot_distance_distributions
    qs = [1, 51, 101, 501, 1001, 2001]  
    for q in qs:  
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename) 
        nlsa.plot_distance_distributions.plot_D_0j(settings)        

flag = 0
if flag == 1:
    bs = [10, 100, 500, 1000, 2000, 3000]  
    for b in bs:
        q = 1001
        root_f = "../../synthetic_data_jitter/test6"
        results_path = '%s/NLSA/q_%d/b_%d'%(root_f, q, b)
        if not os.path.exists(results_path):
            os.mkdir(results_path)
            
        fopen = open('./settings_b_%d.py'%b, 'w')
        fopen.write('# -*- coding: utf-8 -*-\n')
        fopen.write('import numpy\n')
        fopen.write('import math\n')
        fopen.write('S = 30000\n')
        fopen.write('m = 7000\n')
        fopen.write('q = %d\n'%q) 
        fopen.write('T_model = 26000\n')#S-q+1
        fopen.write('tc = float(S)/2\n')
        fopen.write('b = %d\n'%b)  
        fopen.write('log10eps = 1.0\n')
        fopen.write('sigma_sq = 2*10**log10eps\n')
        fopen.write('l = 50\n')
        fopen.write('nmodes = l\n') 
        fopen.write('toproject = range(nmodes)\n') 
        fopen.write('results_path = "%s"\n'%results_path)
        fopen.write('paral_step_A = 50\n')    
        fopen.write('datatype = numpy.float64\n')
        fopen.write('data_file = "%s/x.jbl"\n'%results_path)
        fopen.write('n_workers_A = int(math.ceil(float(q)/paral_step_A))\n')
        fopen.write('p = (q-1)/2\n')
        fopen.write('modes_to_reconstruct = range(20)\n')       
        fopen.write('ncopies = q\n')
        fopen.write('paral_step_reconstruction = 4000\n')
        fopen.write('n_workers_reconstruction = int(math.ceil(float(S-q+1-ncopies+1)/paral_step_reconstruction))\n')
        fopen.close() 
        
flag = 0
if flag == 1:
    # Select b euclidean nns or b time nns.
    #import settings_synthetic_data_jitter as settings
    import nlsa.get_D_N
    bs = [10, 100, 500, 1000, 2000]  
    for b in bs:
        modulename = 'settings_b_%d'%b
        settings = __import__(modulename) 
        nlsa.get_D_N.main_euclidean_nn(settings)
        #nlsa.get_D_N.main_time_nn_1(settings)
        #nlsa.get_D_N.main_time_nn_2(settings)
    
flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings
    import nlsa.get_epsilon
    bs = [10, 100, 500, 1000, 2000]  
    for b in bs: 
        modulename = 'settings_b_%d'%b
        settings = __import__(modulename) 
        nlsa.get_epsilon.main(settings)


flag = 0
if flag == 1:
    log10eps_lst = [-2.0, -1.0, 0.0, 3.0, 8.0]  
    for log10eps in log10eps_lst:
        q = 1001
        b = 100
        root_f = "../../synthetic_data_jitter/test6"
        results_path = '%s/NLSA/q_%d/b_%d/log10eps_%0.1f'%(root_f, q, b, log10eps)
        if not os.path.exists(results_path):
            os.mkdir(results_path)
            
        fopen = open('./settings_log10eps_%0.1f.py'%log10eps, 'w')
        fopen.write('# -*- coding: utf-8 -*-\n')
        fopen.write('import numpy\n')
        fopen.write('import math\n')
        fopen.write('S = 30000\n')
        fopen.write('m = 7000\n')
        fopen.write('q = %d\n'%q) 
        fopen.write('T_model = 26000\n')#S-q+1
        fopen.write('tc = float(S)/2\n')
        fopen.write('b = %d\n'%b)  
        fopen.write('log10eps = %0.1f\n'%log10eps)
        fopen.write('sigma_sq = 2*10**log10eps\n')
        fopen.write('l = 50\n')
        fopen.write('nmodes = l\n') 
        fopen.write('toproject = range(nmodes)\n') 
        fopen.write('results_path = "%s"\n'%results_path)
        fopen.write('paral_step_A = 50\n')    
        fopen.write('datatype = numpy.float64\n')
        fopen.write('data_file = "%s/x.jbl"\n'%results_path)
        fopen.write('n_workers_A = int(math.ceil(float(q)/paral_step_A))\n')
        fopen.write('p = (q-1)/2\n')
        fopen.write('modes_to_reconstruct = range(20)\n')       
        fopen.write('ncopies = q\n')
        fopen.write('paral_step_reconstruction = 4000\n')
        fopen.write('n_workers_reconstruction = int(math.ceil(float(S-q+1-ncopies+1)/paral_step_reconstruction))\n')
        fopen.close() 
      
        
        
flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings
    import nlsa.transition_matrix
    # bs = [10, 100, 500, 1000, 2000]  
    # for b in bs:
    #     modulename = 'settings_b_%d'%b
    modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    for modulename in modulenames: 
        settings = __import__(modulename) 
        nlsa.transition_matrix.main(settings)

flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings
    import nlsa.probability_matrix 
    # bs = [10, 100, 500, 1000, 2000]  
    # for b in bs:
    #     modulename = 'settings_b_%d'%b
    modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    for modulename in modulenames:
        settings = __import__(modulename)      
        nlsa.probability_matrix.main(settings)

flag = 0
if flag == 1:
    ls = [5, 10, 30]  
    for l in ls:
        q = 1001
        b = 100
        log10eps = 1.0
        root_f = "../../synthetic_data_jitter/test6"
        results_path = '%s/NLSA/q_%d/b_%d/log10eps_%0.1f/l_%d'%(root_f, q, b, log10eps, l)
        if not os.path.exists(results_path):
            os.mkdir(results_path)
            
        fopen = open('./settings_l_%d.py'%l, 'w')
        fopen.write('# -*- coding: utf-8 -*-\n')
        fopen.write('import numpy\n')
        fopen.write('import math\n')
        fopen.write('S = 30000\n')
        fopen.write('m = 7000\n')
        fopen.write('q = %d\n'%q) 
        fopen.write('T_model = 26000\n')#S-q+1
        fopen.write('tc = float(S)/2\n')
        fopen.write('b = %d\n'%b)  
        fopen.write('log10eps = %0.1f\n'%log10eps)
        fopen.write('sigma_sq = 2*10**log10eps\n')
        fopen.write('l = %d\n'%l)
        fopen.write('nmodes = l\n') 
        fopen.write('toproject = range(nmodes)\n') 
        fopen.write('results_path = "%s"\n'%results_path)
        fopen.write('paral_step_A = 50\n')    
        fopen.write('datatype = numpy.float64\n')
        fopen.write('data_file = "%s/x.jbl"\n'%results_path)
        fopen.write('n_workers_A = int(math.ceil(float(q)/paral_step_A))\n')
        fopen.write('p = (q-1)/2\n')
        fopen.write('modes_to_reconstruct = range(min(l,20))\n')       
        fopen.write('ncopies = q\n')
        fopen.write('paral_step_reconstruction = 7000\n')
        fopen.write('n_workers_reconstruction = int(math.ceil(float(S-q+1-ncopies+1)/paral_step_reconstruction))\n')
        fopen.close() 

flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings
    import nlsa.eigendecompose
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames: 
    #     settings = __import__(modulename) 
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename) 
        nlsa.eigendecompose.main(settings)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    evecs = joblib.load('%s/evecs_sorted.jbl'%settings.results_path)
    test = numpy.matmul(evecs.T, evecs)
    diff = abs(test - numpy.eye(settings.l))
    print numpy.amax(diff)
    
flag = 0
if flag == 1:  
    #import settings_synthetic_data_jitter as settings
    import nlsa.plot_P_evecs
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames: 
    #     settings = __import__(modulename) 
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename)      
        nlsa.plot_P_evecs.main(settings)

flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames: 
    #     settings = __import__(modulename) 
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename)     
        end_worker = settings.n_workers_A - 1
        os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A.sh %s'
                  %(end_worker, settings.__name__)) 
    
flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings
    import nlsa.util_merge_A
    
    #modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename)    
        nlsa.util_merge_A.main(settings)

flag = 0
if flag == 1: 
    #import settings_synthetic_data_jitter as settings
    import nlsa.SVD
    #modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename) 
        nlsa.SVD.main(settings)
   
flag = 0
if flag == 1: 
    import settings_synthetic_data_jitter as settings
    #modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # ls = [5, 10, 30] 
    # for l in ls:
    #     modulename = 'settings_l_%d'%l
    #     settings = __import__(modulename)   
    import nlsa.plot_SVs
    nlsa.plot_SVs.main(settings)    
    import nlsa.plot_chronos
    nlsa.plot_chronos.main(settings)
        
flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename) 
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames: 
    #     settings = __import__(modulename) 
        # MAKE SUPERVECTOR TIMESTAMPS
        S = settings.S
        q = settings.q  
        s = S-q+1
        
        ts_meas = joblib.load('%s/ts_meas.jbl'%settings.results_path)  
        ts_svs = []
        for i in range(s):
            t_sv = numpy.average(ts_meas[i:i+q])
            ts_svs.append(t_sv)        
        ts_svs = numpy.asarray(ts_svs)
        joblib.dump(ts_svs, '%s/ts_svs.jbl'%settings.results_path)
    
# p-dependent reconstruction 
flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings
    import nlsa.reconstruct_p     
    # modulenames = ['settings_log10eps_m1p5', 'settings_log10eps_8p0']
    # for modulename in modulenames:
    #     settings = __import__(modulename)   
    qs = [2001]#, 501, 1001, 2001]  
    for q in qs:  
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)    
        nlsa.reconstruct_p.f(settings)  
        nlsa.reconstruct_p.f_ts(settings)  


# STANDARD RECONSTRUCTION    
flag = 0
if flag == 1:
    #import settings_q_2001 as settings
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename) 
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames: 
    #     settings = __import__(modulename) 
        end_worker = settings.n_workers_reconstruction - 1
        os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s'
                  %(end_worker, settings.__name__))    
         
# TIME ASSIGNMENT with p=(q-1)/2
flag = 0
if flag == 1:
    #import settings_q_2001 as settings
    import nlsa.reconstruct_p
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename) 
    # modulenames = ['settings_log10eps_m2p0', 'settings_log10eps_m1p0', 'settings_log10eps_0p0', 'settings_log10eps_3p0', 'settings_log10eps_8p0']
    # for modulename in modulenames: 
    #     settings = __import__(modulename) 
        nlsa.reconstruct_p.f_ts(settings)
    
flag = 0
if flag == 1:
    import nlsa.util_merge_x_r  
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename)   
           
        for mode in settings.modes_to_reconstruct:
            nlsa.util_merge_x_r.f(settings, mode) 

# CC TO BENCHMARK        
flag = 0
if flag == 1:
    #import settings_synthetic_data_jitter as settings  
    import nlsa.reconstruct_p
    ls = [5, 10, 30] 
    for l in ls:
        modulename = 'settings_l_%d'%l
        settings = __import__(modulename) 
        t_r = joblib.load('%s/t_r_p_%d.jbl'%(settings.results_path, settings.p))
        
        benchmark = eval_model(settings, t_r)
        print 'Benchmark: ', benchmark.shape
        plot_signal(benchmark, '%s/benchmark_at_t_r.png'%(settings.results_path))
        benchmark = benchmark.flatten()
    
        x_r_tot = 0
        CCs = []
        for mode in range(min(l, 20)):
            print 'Mode: ', mode
            
            x_r = joblib.load('%s/movie_mode_%d_parallel.jbl'%(settings.results_path, mode))
            #x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(settings.results_path, settings.p, mode))
            print 'x_r: ', x_r.shape
            plot_signal(x_r, '%s/x_r_mode_%d.png'%(settings.results_path, mode))
            
            x_r_tot += x_r      
            x_r_tot_flat = x_r_tot.flatten()
            
            CC = correlate.Correlate(benchmark, x_r_tot_flat)
            print 'CC: ', CC
            CCs.append(CC)
            
            plot_signal(x_r_tot, '%s/x_r_tot_%d_modes.png'%(settings.results_path, mode+1), '%.4f'%CC)
            
        joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)
    
        matplotlib.pyplot.scatter(range(1, len(CCs)+1), CCs, c='b')
        matplotlib.pyplot.xticks(range(1,len(CCs)+1,2))
        matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_nmodes.png'%(settings.results_path))
        matplotlib.pyplot.close()
"""