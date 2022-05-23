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

import correlate
import local_linearity
 
def model(settings, i, ts):    
    m = settings.m    
    T = settings.T
    omega = 2*numpy.pi/T    
    tc = settings.tc
    
    e1 = 1-numpy.exp(-ts/tc)
    e2 = 1-e1
    
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
    return x_i
    
def eval_model(settings, times):
    m = settings.m    
    x = numpy.zeros((m,times.shape[0]))      
    for i in range(m):        
        x[i,:] = model(settings,i,times)
    return x

def plot_signal(image, filename, fig_title=''):
    im = matplotlib.pyplot.imshow(image, cmap='jet')
    matplotlib.pyplot.title(fig_title)
    ax = matplotlib.pyplot.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)   
    cb = matplotlib.pyplot.colorbar(im, cax=cax)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cb.locator = tick_locator
    cb.update_ticks()
    matplotlib.pyplot.savefig(filename, dpi=96*3)
    matplotlib.pyplot.close()  
    
def make_x(settings):
    m = settings.m
    S = settings.S
    
    results_path = settings.results_path
    
    x = numpy.zeros((m,S))    
    mask = numpy.zeros((m,S))
    
    ts_meas = numpy.sort(S*numpy.random.rand(S))
    
    T = settings.T
    min_period = T/11
    jitter_factor = settings.jitter_factor
    jitter_std_dev = jitter_factor*min_period
    ts_true = ts_meas + numpy.random.normal(loc=0.0, scale=jitter_std_dev, size=S)
    
    for i in range(m):        
        x_i = model(settings, i, ts_true)             
        
        partialities = numpy.random.rand(S)             
        x_i = x_i*partialities
        
        sparsities = numpy.random.rand(S)
        thr = 0.982
        sparsities[sparsities<thr]  = 0
        sparsities[sparsities>=thr] = 1
        x_i = x_i*sparsities
        
        mask[i,:] = sparsities
        x[i,:] = x_i
        
    fn = '%s/x_jitter_factor_%0.2f.png'%(results_path, jitter_factor)
    plot_signal(x, fn)
    fn = '%s/x_underlying_ts_meas.png'%(results_path)
    plot_signal(eval_model(settings, ts_meas), fn)
    
    joblib.dump(x, '%s/x.jbl'%results_path)  
    joblib.dump(mask, '%s/mask.jbl'%results_path)
    joblib.dump(ts_true, '%s/ts_true.jbl'%results_path)  
    joblib.dump(ts_meas, '%s/ts_meas.jbl'%results_path)  
    

def make_lp_filter_functions(settings):
    import make_lp_filter
    print 'q: ', settings.q
    
    results_path = settings.results_path
    F = make_lp_filter.get_F_sv_t_range(settings)
    Q, R = make_lp_filter.on_qr(settings, F)
    
    d = make_lp_filter.check_on(Q)
    print 'Normalisation: ', numpy.amax(abs(d))
    
    joblib.dump(Q, '%s/F_on_qr.jbl'%results_path)



##############################
### Make synthetic dataset ###
##############################

flag = 0
if flag == 1:    
    import settings_synthetic_data_jitter as settings
    make_x(settings)


    
#################################
### LPSA PARA SEARCH : q-scan ###
#################################
root_path = '../../synthetic_data_jitter/test6/LPSA_para_search'
m = 7000
S = 30000
paral_step_A = 500
f_max = 100
f_max_considered = f_max

qs = [1, 51, 101, 501, 1001, 2001, 3001, 4001, 5001]

flag = 0
if flag == 1:
    for q in qs:
        q_path = '%s/f_max_%d_q_%d'%(root_path, f_max, q)
        if not os.path.exists(q_path):
            os.mkdir(q_path)
        
flag = 0
if flag == 1:
    for q in qs:
        q_path = '%s/f_max_%d_q_%d'%(root_path, f_max, q)
        fopen = open('/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/nlsa/settings_q_%d.py'%q, 'w')
        fopen.write('# -*- coding: utf-8 -*-\n')
        fopen.write('import numpy\n')
        fopen.write('S = %d\n'%S)
        fopen.write('m = %d\n'%m)
        fopen.write('q = %d\n'%q)    
        fopen.write('f_max = %d\n'%f_max)
        fopen.write('f_max_considered = f_max\n')
        fopen.write('results_path = "%s"\n'%q_path)
        fopen.write('paral_step_A = %d\n'%paral_step_A)
        fopen.write('datatype = numpy.float64\n')
        data_file = '%s/x.jbl'%q_path
        fopen.write('data_file = "%s"\n'%data_file)
        n_workers_A = int(math.ceil(float(q)/paral_step_A))
        fopen.write('n_workers_A = %d\n'%n_workers_A)
        fopen.write('p = 0\n')
        fopen.write('modes_to_reconstruct = range(20)\n')
        fopen.close()

flag = 0
if flag == 1:    
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)
        print 'q: ', settings.q
        print settings.results_path
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
    
    import nlsa.SVD
    print '\n****** RUNNING SVD ******'
    for q in qs:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        
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
    import nlsa.plot_chronos    
    for q in qs:        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        nlsa.plot_SVs.main(settings)       
        nlsa.plot_chronos.main(settings)

# p=0 reconstruction 
flag = 0
if flag == 1:
    import nlsa.reconstruct_p
    for q in qs:        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        nlsa.reconstruct_p.f(settings)
        nlsa.reconstruct_p.f_ts(settings)      

# Calculate L of p=0 reconstructed signal 
# (eqv: L of central block of reconstreucted supervecctors)   
flag = 0
if flag == 1:
    for q in qs:        
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        p = settings.p
        results_path = settings.results_path
        t_r = joblib.load('%s/t_r_p_%d.jbl'%(results_path, p))
        
        lls = []   
        x_r_tot = 0
        for mode in settings.modes_to_reconstruct:
            print 'Mode: ', mode
            x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(results_path, p, mode))
            x_r_tot += x_r              
            L = local_linearity.local_linearity_measure_jitter(x_r_tot, t_r)
            lls.append(L)
        joblib.dump(lls, '%s/p_%d_local_linearity_vs_nmodes.jbl'%(results_path, p))   
        
        matplotlib.pyplot.scatter(range(1, len(lls)+1), numpy.log10(lls), c='b')
        matplotlib.pyplot.xticks(range(1,len(lls)+1,2))
        matplotlib.pyplot.savefig('%s/p_%d_log10_L_vs_nmodes_q_%d_fmax_%d.png'%(results_path, p, settings.q, settings.f_max_considered))
        matplotlib.pyplot.close()  
        
flag = 0
if flag == 1:
    nmodes = 4
    q = qs[0]
    modulename = 'settings_q_%d'%q
    settings = __import__(modulename)        
    print 'q: ', settings.q
    p = 0
    results_path = settings.results_path
    x_r_tot = 0
    for mode in range(nmodes):
        print 'Mode: ', mode
        x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(results_path, p, mode))
        x_r_tot += x_r     
    start = x_r_tot
    start_large = numpy.zeros((settings.m, settings.S))
    # for p=0
    start_large[:,(q-1)/2:(q-1)/2+(settings.S-q+1)] = start
    
    q_largest = qs[-1]
    CCs = []
    for q in qs[1:]:
        modulename = 'settings_q_%d'%q
        settings = __import__(modulename)        
        print 'q: ', settings.q
        results_path = settings.results_path
        x_r_tot = 0
        for mode in range(nmodes):
            print 'Mode: ', mode
            x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(results_path, p, mode))
            x_r_tot += x_r  
            
        x_r_tot_large = numpy.zeros((settings.m, settings.S))
        # for p=0
        x_r_tot_large[:,(q-1)/2:(q-1)/2+(settings.S-q+1)] = x_r_tot
        
        start_flat   =   start_large[:,(q_largest-1)/2:(q_largest-1)/2+(settings.S-q_largest+1)].flatten()
        x_r_tot_flat = x_r_tot_large[:,(q_largest-1)/2:(q_largest-1)/2+(settings.S-q_largest+1)].flatten()
        CC = correlate.Correlate(start_flat, x_r_tot_flat)
        CCs.append(CC)
        print CC
        
        start_large = x_r_tot_large
        
    joblib.dump(CCs, '%s/LPSA_f_max_100_q_scan_p_%d_%d_modes_successive_CCs.jbl'%(root_path, p, nmodes))
        
    n_curves = len(CCs)
    colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
    matplotlib.pyplot.xticks(range(1,len(CCs)+1,1))
    
    for i, CC in enumerate(CCs):
        matplotlib.pyplot.scatter(i+1, CCs[i], c=colors[i], label='q=%d,q=%d'%(qs[i], qs[i+1]))
    matplotlib.pyplot.legend(frameon=True, scatterpoints=1, loc='upper right', fontsize=10) 
    matplotlib.pyplot.savefig('%s/LPSA_f_max_100_q_scan_p_%d_%d_modes_successive_CCs.png'%(root_path, p, nmodes))
    matplotlib.pyplot.close()


####################################
### LPSA PARA SEARCH : jmax-scan ###
####################################
root_path = '../../synthetic_data_jitter/test6/LPSA_para_search'

m = 7000
S = 30000
paral_step_A = 1000
q = 4001
f_max_s = [1, 5, 10, 50, 150, 300]    # 100 already done
    
flag = 0
if flag == 1:
    for f_max in f_max_s:
        f_max_path = '%s/f_max_%d_q_%d'%(root_path, f_max, q)
        if not os.path.exists(f_max_path):
            os.mkdir(f_max_path)
               
flag = 0
if flag == 1:
    for f_max in [100]:#f_max_s:
        f_max_path = '%s/f_max_%d_q_%d'%(root_path, f_max, q)
        fopen = open('/das/work/units/LBR-Xray/p17491/Cecilia_Casadei/NLSA/code/nlsa/settings_f_max_%d.py'%f_max, 'w')
        fopen.write('# -*- coding: utf-8 -*-\n')
        fopen.write('import numpy\n')
        fopen.write('S = %d\n'%S)
        fopen.write('m = %d\n'%m)
        fopen.write('q = %d\n'%q)    
        fopen.write('f_max = %d\n'%f_max)
        fopen.write('f_max_considered = f_max\n')
        fopen.write('results_path = "%s"\n'%f_max_path)
        fopen.write('paral_step_A = %d\n'%paral_step_A)
        fopen.write('datatype = numpy.float64\n')
        data_file = '%s/x.jbl'%f_max_path
        fopen.write('data_file = "%s"\n'%data_file)
        n_workers_A = int(math.ceil(float(q)/paral_step_A))
        fopen.write('n_workers_A = %d\n'%n_workers_A)
        fopen.write('p = 0\n')
        n_modes = min(20, 2*f_max+1)
        fopen.write('modes_to_reconstruct = range(0, %d)\n'%n_modes)
        fopen.close()

flag = 0
if flag == 1:    
    for f_max in f_max_s:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)
        
        print 'jmax: ', settings.f_max
        print settings.results_path
        
        make_lp_filter_functions(settings)


flag = 0
if flag == 1:
    for f_max in [100]:#f_max_s:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        
        end_worker = settings.n_workers_A - 1
        os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A_fourier.sh %s'
                  %(end_worker, settings.__name__)) 

flag = 0
if flag == 1:
    import nlsa.util_merge_A
    for f_max in [100]:#f_max_s:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        
        nlsa.util_merge_A.main(settings)
        
flag = 0
if flag == 1: 
    import nlsa.SVD
    print '\n****** RUNNING SVD ******'
    for f_max in f_max_s:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        
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
    import nlsa.plot_chronos
    for f_max in f_max_s:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'jmax: ', settings.f_max
        nlsa.plot_SVs.main(settings)       
        nlsa.plot_chronos.main(settings)
        
# p=0 reconstruction 
flag = 0
if flag == 1:
    import nlsa.reconstruct_p
    for f_max in [100]:#f_max_s[:-1]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print settings.f_max
        nlsa.reconstruct_p.f(settings)
        nlsa.reconstruct_p.f_ts(settings)              

# Calculate L of p=0 reconstructed signal 
# (eqv: L of central block of reconstreucted supervecctors)   
flag = 0
if flag == 1:
    for f_max in f_max_s[:-1]:
        modulename = 'settings_f_max_%d'%f_max
        settings = __import__(modulename)        
        print 'f_max: ', settings.f_max     
        print 'q: ', settings.q
        p = settings.p
        results_path = settings.results_path
        t_r = joblib.load('%s/t_r_p_%d.jbl'%(results_path, p))
        
        lls = []   
        x_r_tot = 0
        for mode in settings.modes_to_reconstruct:
            print 'Mode: ', mode
            x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(results_path, p, mode))
            x_r_tot += x_r              
            L = local_linearity.local_linearity_measure_jitter(x_r_tot, t_r)
            lls.append(L)
        joblib.dump(lls, '%s/p_%d_local_linearity_vs_nmodes.jbl'%(results_path, p))   
        
        matplotlib.pyplot.scatter(range(1, len(lls)+1), numpy.log10(lls), c='b')
        matplotlib.pyplot.xticks(range(1,len(lls)+1,2))
        matplotlib.pyplot.savefig('%s/p_%d_log10_L_vs_nmodes_q_%d_fmax_%d.png'%(results_path, p, settings.q, settings.f_max_considered))
        matplotlib.pyplot.close()  

# CCs of central blockwith increasing fmax
f_max_s = [5, 10, 50, 100, 150]        
flag = 0
if flag == 1:
    nmodes = 4
    CCs = []
    fmax = f_max_s[0]
    modulename = 'settings_f_max_%d'%fmax
    settings = __import__(modulename)        
    print 'fmax: ', settings.f_max
    p = 0
    results_path = settings.results_path
    x_r_tot = 0
    for mode in range(nmodes):
        print 'Mode: ', mode
        x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(results_path, p, mode))
        x_r_tot += x_r     
    start = x_r_tot
    
    for fmax in f_max_s[1:]:
        modulename = 'settings_f_max_%d'%fmax
        settings = __import__(modulename)        
        print 'fmax: ', settings.f_max
        results_path = settings.results_path
        x_r_tot = 0
        for mode in range(nmodes):
            print 'Mode: ', mode
            x_r = joblib.load('%s/movie_p_%d_mode_%d.jbl'%(results_path, p, mode))
            x_r_tot += x_r  
        
        start_flat   =   start.flatten()
        x_r_tot_flat = x_r_tot.flatten()
        CC = correlate.Correlate(start_flat, x_r_tot_flat)
        CCs.append(CC)
        print CC
        
        start = x_r_tot
        
    joblib.dump(CCs, '%s/LPSA_f_max_scan_q_4001_scan_p_%d_%d_modes_successive_CCs.jbl'%(root_path, p, nmodes))

    n_curves = len(CCs)
    colors = matplotlib.pylab.cm.Blues(numpy.linspace(0.15,1,n_curves))   
    matplotlib.pyplot.xticks(range(1,len(CCs)+1,1))
    
    for i, CC in enumerate(CCs):
        matplotlib.pyplot.scatter(i+1, CCs[i], c=colors[i], label='$j_{\mathrm{max}}=$%d, $j_{\mathrm{max}}=$%d'%(f_max_s[i], f_max_s[i+1]))
    matplotlib.pyplot.legend(frameon=True, scatterpoints=1, loc='lower right', fontsize=10) 
    matplotlib.pyplot.savefig('%s/LPSA_f_max_scan_q_4001_scan_p_%d_%d_modes_successive_CCs.png'%(root_path, p, nmodes))
    matplotlib.pyplot.close()


###############################
### Standard reconstruction ###
###############################
flag = 0
if flag == 1:  
    import nlsa.plot_SVs
    import nlsa.plot_chronos
    import settings_synthetic_data_jitter as settings
    nlsa.plot_SVs.main(settings)       
    nlsa.plot_chronos.main(settings)
        
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
    benchmark = eval_model(settings, t_r)
    print 'Benchmark: ', benchmark.shape
    plot_signal(benchmark, '%s/benchmark_at_t_r.png'%(settings.results_path))
    
    benchmark = benchmark.flatten()
    
    CCs = []
    
    U = joblib.load('%s/U.jbl'%settings.results_path)
    S = joblib.load('%s/S.jbl'%settings.results_path)
    VH = joblib.load('%s/VT_final.jbl'%settings.results_path)
    print U.shape, S.shape, VH.shape
    x_r_tot = 0
    for mode in range(20):
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
        
    joblib.dump(CCs, '%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)

flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    CCs = joblib.load('%s/reconstruction_CC_vs_nmodes.jbl'%settings.results_path)
    matplotlib.pyplot.scatter(range(1, len(CCs)+1), CCs, c='b')
    matplotlib.pyplot.xticks(range(1,len(CCs)+1,2))
    matplotlib.pyplot.savefig('%s/reconstruction_CC_vs_nmodes.png'%(settings.results_path))
    matplotlib.pyplot.close()


flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings      
    U = joblib.load('%s/U.jbl'%settings.results_path)
    S = joblib.load('%s/S.jbl'%settings.results_path)
    VH = joblib.load('%s/VT_final.jbl'%settings.results_path)
    print U.shape, S.shape, VH.shape
    
    t_r = joblib.load('%s/t_r_p_%d.jbl'%(settings.results_path, settings.p))
    
    local_linearity_lst = []   
    x_r_tot = 0
    for mode in range(20):
        print 'Mode: ', mode
        u = U[:, mode]
        sv = S[mode]
        vT = VH[mode, :]
        x_r = sv*numpy.outer(u, vT)
        print 'x_r: ', x_r.shape
    
        x_r_tot += x_r  
        L = local_linearity.local_linearity_measure_jitter(x_r_tot, t_r)
        
        local_linearity_lst.append(L)
    joblib.dump(local_linearity_lst, '%s/local_linearity_vs_nmodes.jbl'%settings.results_path)   
    
flag = 0
if flag == 1:
    import settings_synthetic_data_jitter as settings
    lls = joblib.load('%s/local_linearity_vs_nmodes.jbl'%settings.results_path)
    matplotlib.pyplot.scatter(range(1, len(lls)+1), numpy.log10(lls), c='b')
    matplotlib.pyplot.xticks(range(1,len(lls)+1,2))
    matplotlib.pyplot.savefig('%s/local_linearity_vs_nmodes.png'%(settings.results_path))
    matplotlib.pyplot.close()   



# BINNING 
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
