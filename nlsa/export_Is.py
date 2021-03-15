# -*- coding: utf-8 -*-
import os
import joblib
import numpy

import settings_rho_light as settings

def get_avg_Is_negbin():
    results_path = settings.results_path
    mult_factor = 1000000
    label = settings.label
    
    fpath = '%s/reconstructed_intensities_mode_0_neg_avg/'%(results_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        
    miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
    miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
    miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    x_r = joblib.load('%s/movie_mode_0_parallel_extended_fwd_bwd.jbl'%(results_path))
    
    # Select negative delay bin
    x_r = x_r[:,15000:25000]
    I = numpy.mean(x_r, axis=1)
    
    print I.shape
    if I.shape[0] != miller_h.shape[0]:
        print 'Problem'
    
    I[numpy.isnan(I)] = 0
    I[I<0] = 0 
    I = mult_factor * I 
    sigIs = numpy.sqrt(I)
        
    out[:, 3] = I
    out[:, 4] = sigIs
    
    f_out = '%s/rho_%s_mode_0_neg_avg.txt'%(fpath, label)              
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%17.2f%17.2f')

def get_avg_Is():
    results_path = settings.results_path
    mult_factor = 100000
    label = settings.label
    
    fpath = '%s/reconstructed_intensities_mode_0_avg/'%(results_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        
    miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
    miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
    miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    x_r = joblib.load('%s/movie_mode_0_parallel_extended_fwd_bwd.jbl'%(results_path))
    I = numpy.mean(x_r, axis=1)
    
    print I.shape
    if I.shape[0] != miller_h.shape[0]:
        print 'Problem'
    
    I[numpy.isnan(I)] = 0
    I[I<0] = 0 
    I = mult_factor * I 
    sigIs = numpy.sqrt(I)
        
    out[:, 3] = I
    out[:, 4] = sigIs
    
    f_out = '%s/rho_%s_mode_0_avg.txt'%(fpath, label)              
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%17.2f%17.2f')
        
    

def get_Is(mode):
    
    print 'Modes 0 +', mode
    results_path = settings.results_path
    mult_factor = 1000000
    label = settings.label
    
    output_step = 100
    mode_0 = 0
    mode_i = mode
    
    fpath = '%s/reconstructed_intensities_mode_%d_%d_extended_fwd_bwd/'%(results_path, 
                                                                         mode_0, 
                                                                         mode_i)
    #fpath = '%s/reconstructed_intensities_mode_%d_extended_fwd_bwd/'%(results_path, 
    #                                                                   mode_i)
    #fpath = '%s/reconstructed_intensities_mode_all_extended_fwd_bwd/'%(results_path)
    
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        
    miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
    miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
    miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()

    x_r_0 = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
                                                                          mode_0))
    #for mode_i in range(1, 6):
    x_r_i = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
                                                                          mode_i))
    #x_r = x_r + x_r_i
    
    x_r = x_r_0 + x_r_i
    #x_r = x_r_i
    print x_r.shape
    
    if x_r.shape[0] != miller_h.shape[0]:
        print 'Problem'
        
    for i in range(0, x_r.shape[1], output_step):
    #for i in range(75000, 90000, output_step):    
        
        f_out = '%s/rho_%s_mode_%d_%d_timestep_%0.6d.txt'%(fpath, 
                                                           label, 
                                                           mode_0, 
                                                           mode_i, 
                                                           i)    
        #f_out = '%s/bR_%s_mode_%d_timestep_%0.6d.txt'%(fpath, label, mode_i, i)   
        #f_out = '%s/bR_%s_mode_all_timestep_%0.6d.txt'%(fpath, label, i)                                                     
    
        I = x_r[:, i]
    
        I[numpy.isnan(I)] = 0
        I[I<0] = 0 
        I = mult_factor * I 
        sigIs = numpy.sqrt(I)
        
        out[:, 3] = I
        out[:, 4] = sigIs
        
        numpy.savetxt(f_out, out, fmt='%6d%6d%6d%16.2f%16.2f')




def get_Is_bins():
    label = settings.label 
    mult_factor = 1000000
    results_path = settings.results_path
    q = settings.q
    ncopies = settings.ncopies   
        
    fpath = '%s/reconstructed_intensities_bins/'%(results_path)     
    if not os.path.exists(fpath):
            os.mkdir(fpath)
            
    miller_h = joblib.load('../data_bR/converted_data_%s/miller_h_%s.jbl'%(label, label))
    miller_k = joblib.load('../data_bR/converted_data_%s/miller_k_%s.jbl'%(label, label))
    miller_l = joblib.load('../data_bR/converted_data_%s/miller_l_%s.jbl'%(label, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    t_uniform = joblib.load('../data_bR/converted_data_%s/t_uniform_%s.jbl'%(label, label))
    S = t_uniform.shape[0]
    t_final = t_uniform[q:S-ncopies+1,:].flatten()
    print t_final.shape[0], 'reconstructed frames'
    
    mode_0 = 0
    #mode_i = 5
    x_r_0 = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode_0))
    
    x_r = x_r_0
    for mode_i in range(1, 6):   
        print mode_i
        x_r_i = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode_i))
        x_r = x_r + x_r_i
        
    #x_r = x_r_0 + x_r_i
    print x_r.shape

    ts = [0, 49, 94, 141, 193, 245, 314, 406, 457, 490, 518, 547, 574, 606, 646, 695, 759, 847, 946, 1025, 1201]
    for i in range(0, len(ts)-3):
        t1 = float(ts[i])
        t2 = float(ts[i+3])
        print t1, t2
        
        idxs = numpy.argwhere((t1 <= t_final) & (t_final < t2)).flatten()
        x_r_bin = x_r[:,idxs]
        n_bin = x_r_bin.shape[1]
        f_out = '%s/bR_%s_mode_all_bin_%d_%dfs_%dfs_n_%d.txt'%(fpath, label, i, t1, t2, n_bin)
        #f_out = '%s/bR_%s_mode_%d_%d_bin_%d_%dfs_%dfs_n_%d.txt'%(fpath, label, mode_0, mode_i, i, t1, t2, n_bin)
        
        avg_bin = x_r_bin.sum(axis=1)
        I = avg_bin/n_bin
        I[numpy.isnan(I)] = 0
        I[I<0] = 0 
        I = mult_factor * I 
        sigIs = numpy.sqrt(I)
            
        out[:, 3] = I
        out[:, 4] = sigIs
            
        numpy.savetxt(f_out, out, fmt='%6d%6d%6d%12.2f%12.2f')

def get_info():
    results_path = settings.results_path
    
    S = joblib.load('%s/S.jbl'%results_path)
    
    mode_0 = 0        
    print 'Mode: ', mode_0
    x_r_0 = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, 
                                                         mode_0))
    N = x_r_0.shape[0]*x_r_0.shape[1]
    idxs = numpy.argwhere(x_r_0<0)
    print 'Negative values in x_r_0: ', float(len(idxs))/N
       
#    modes = [1, 2, 3, 4, 5]
#    for mode_i in modes:
#        print '******* Mode: ', mode_i
#        
#        x_r_i = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, 
#                                                                  mode_i))
#        print 'Before reweight'
#        idxs = numpy.argwhere(x_r_i<0)
#        print 'Negative values in x_r_i: ', float(len(idxs))/N
#     
#        x_r = x_r_0 + x_r_i
#        idxs = numpy.argwhere(x_r<0)
#        print 'Negative values in x_r: ', float(len(idxs))/N
#        
#        
#        x_r_i = x_r_i * (S[0]/S[mode_i])    
#        print 'Reweight factor: ', S[0]/S[mode_i]
#        
#        
#        print 'After reweight'
#        
#        x_r = x_r_0 + x_r_i
#        idxs = numpy.argwhere(x_r<0)
#        print 'Negative values in x_r: ', float(len(idxs))/N
#        
    
def get_Is_reweight(mode):
    
    print 'mode ', mode
    results_path = settings.results_path
    mult_factor = 1000000
    label = settings.label

    S = joblib.load('%s/S.jbl'%results_path)
    print S[0], S[mode]
    
    output_step = 100
    mode_0 = 0
    mode_i = mode

    fpath = '%s/reconstructed_intensities_mode_%d_%d_equalweight/'%(results_path, 
                                                                    mode_0,
                                                                    mode_i)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        
    miller_h = joblib.load('../data_bR/converted_data_%s/miller_h_%s.jbl'%(label, 
                                                                           label))
    miller_k = joblib.load('../data_bR/converted_data_%s/miller_k_%s.jbl'%(label, 
                                                                           label))
    miller_l = joblib.load('../data_bR/converted_data_%s/miller_l_%s.jbl'%(label,
                                                                           label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    x_r_0 = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode_0))
    x_r_i= joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode_i))
     
    x_r_i = x_r_i * (S[0]/S[mode_i])    
    print 'Reweight factor: ', S[0]/S[mode_i]
    x_r = x_r_0 + x_r_i
    print x_r.shape

    if x_r.shape[0] != miller_h.shape[0]:
        print 'Problem'
            
    for i in range(0, x_r.shape[1], output_step):            
        print i
        f_out = '%s/bR_%s_mode_%d_%d_timestep_%d_equalweight.txt'%(fpath, 
                                                                   label, 
                                                                   mode_0, 
                                                                   mode_i, 
                                                                   i)                                                        
        I = x_r[:, i]
    
        I[numpy.isnan(I)] = 0
        I[I<0] = 0 
        I = mult_factor * I 
        sigIs = numpy.sqrt(I)
        
        out[:, 3] = I
        out[:, 4] = sigIs
        
        numpy.savetxt(f_out, out, fmt='%6d%6d%6d%12.2f%12.2f')    
        
def check_Is(mode):
    
    print 'Modes 0 +', mode
    results_path = settings.results_path
    label = settings.label
    
    mode_0 = 0
    mode_i = mode  
        
    miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
    miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
    miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()

    x_r_0 = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
                                                                          mode_0))    
    x_r_i = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
                                                                          mode_i))    
    x_r = x_r_0 + x_r_i  
    print x_r.shape
    for idx in range(10, 1000, 100):
        print out[idx, 0], out[idx, 1], out[idx, 2], x_r[idx, 0]
    
def export_binned_data():
    #results_path = settings.results_path
    datatype = settings.datatype
    label = settings.label
    
    mult_factor = 100
       
    # EXTRACT DATA
    T_sparse = joblib.load('../data_rho/data_converted_%s/T_sparse_%s.jbl'%(label,
                                                                            label))
    print 'T_sparse: ', T_sparse.shape
    print 'T_sparse nonzero: ', T_sparse.count_nonzero()
    x = T_sparse[:,:].todense()
    print 'x: ', x.shape, x.dtype
    x = numpy.asarray(x, dtype=datatype)
    print 'x: ', x.shape, x.dtype
    
    M_sparse = joblib.load('../data_rho/data_converted_%s/M_sparse_%s.jbl'%(label,
                                                                            label))
    print 'M_sparse: ', M_sparse.shape
    print 'M_sparse nonzero: ', M_sparse.count_nonzero()
    mask = M_sparse[:,:].todense()
    print 'mask: ', mask.shape, mask.dtype
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    print 'mask: ', mask.shape, mask.dtype
#    test = mask.sum(axis = 1)
#    idxs_test = numpy.argwhere(test<1)
#    print 'test', idxs_test.shape
    
#    t_uniform = joblib.load('../data_rho/data_converted_%s/t_uniform_light.jbl'%label)
#    t_uniform = t_uniform.flatten()
#    print t_uniform.shape
#    
#    idxs = numpy.argwhere(t_uniform < -100).flatten()
#    print idxs.shape, ' in negative bin'
#    
#    T_neg = x[:, idxs]
#    M_neg = mask[:, idxs]
#    print 'Negative delays: ', T_neg.shape, M_neg.shape
#    
#    idxs = numpy.argwhere(t_uniform > 200).flatten()
#    print idxs.shape, ' in positive bin'
#    
#    T_pos = x[:, idxs]
#    M_pos = mask[:, idxs]
#    print 'Positive delays: ', T_pos.shape, M_pos.shape
#    
#    I_neg = T_neg.sum(axis=1)/M_neg.sum(axis=1)
#    I_pos = T_pos.sum(axis=1)/M_pos.sum(axis=1)
#    print I_neg.shape
#    print I_pos.shape

    I = x.sum(axis=1)/mask.sum(axis=1)
    
    miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
    miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
    miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
        
#    I_neg[numpy.isnan(I_neg)] = 0
#    I_neg[I_neg<0] = 0 
#    I_neg = mult_factor * I_neg 
#    sigIneg = numpy.sqrt(I_neg)
#    
#    out[:, 3] = I_neg
#    out[:, 4] = sigIneg
#    
#    f_out = './negative_bin_mult_%d.txt'%mult_factor
#    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%12.2f%12.2f')    
    
#    I_pos[numpy.isnan(I_pos)] = 0
#    I_pos[I_pos<0] = 0 
#    I_pos = mult_factor * I_pos
#    sigIpos = numpy.sqrt(I_pos)
#    
#    out[:, 3] = I_pos
#    out[:, 4] = sigIpos
#    
#    f_out = './positive_bin_mult_%d.txt'%mult_factor
#    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%12.2f%12.2f')   

    I[numpy.isnan(I)] = 0
    I[I<0] = 0 
    I = mult_factor * I
    sigI = numpy.sqrt(I)
    
    out[:, 3] = I
    out[:, 4] = sigI
    
    f_out = './dark_bin_mult_%d.txt'%mult_factor
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  


def get_avg_plus_dI(mode):
    
    print 'Avg + mode ', mode
    results_path = settings.results_path
    mult_factor = 1000000
    label = settings.label    
    output_step = 100
     
    fpath = '%s/reconstructed_intensities_avg_plus_mode_%d/'%(results_path, 
                                                              mode)    
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        
    miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
    miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
    miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten() 

    folder = '../data_rho/data_converted_%s'%label
    x_avg = joblib.load('%s/T_avg_%s.jbl'%(folder, label))    
    x_r = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
                                                                        mode))
    print 'x_avg: ', x_avg.shape
    print 'x_r: ', x_r.shape
    
    if x_r.shape[0] != miller_h.shape[0]:
        print 'Problem'
        
    for i in range(0, x_r.shape[1], output_step):
        f_out = '%s/rho_%s_avg_plus_mode_%d_timestep_%0.6d.txt'%(fpath, 
                                                                 label, 
                                                                 mode, 
                                                                 i)    
        I = x_r[:, i] + x_avg
    
        I[numpy.isnan(I)] = 0
        I[I<0] = 0 
        I = mult_factor * I 
        sigIs = numpy.sqrt(I)
        
        out[:, 3] = I
        out[:, 4] = sigIs
        
        numpy.savetxt(f_out, out, fmt='%6d%6d%6d%16.2f%16.2f')

def get_avg_dark():
    
    print 'Avg dark'
    results_path = settings.results_path
    mult_factor = 1000000
    label = settings.label    
           
    miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
    miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
    miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    folder = '../data_rho/data_converted_%s'%label
    I = joblib.load('%s/T_avg_%s.jbl'%(folder, label))
        
    print 'x_avg: ', I.shape
    
    f_out = '%s/rho_%s_avg.txt'%(results_path, label) 
        
    I[numpy.isnan(I)] = 0
    I[I<0] = 0 
    I = mult_factor * I 
    sigIs = numpy.sqrt(I)
    
    out[:, 3] = I
    out[:, 4] = sigIs
    
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%16.2f%16.2f')
        
if __name__ == "__main__":
    #check_Is(1)
#    get_info()
    #get_avg_dark()
    #for mode in range(0, 3):
     #   get_avg_plus_dI(mode)
    for mode in range(1,3):
        get_Is(mode)
        
#    for mode in range(1, 4):
#        get_Is(mode)
    #export_binned_data()