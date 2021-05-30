# -*- coding: utf-8 -*-
import os
import joblib
import numpy

def get_Is(settings, mode):
    
    data_path = settings.data_path
    results_path = settings.results_path
    label = settings.label
    
    mult_factor = 1000000    
    output_step = 100
    
    mode_0 = 0
    mode_i = mode
    
    print 'Modes', mode_0, ' +', mode
    
    fpath = '%s/reconstructed_intensities_mode_%d_%d/'%(results_path, 
                                                        mode_0, 
                                                        mode_i)
    
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        
    miller_h = joblib.load('%s/miller_h_%s.jbl'%(data_path, label))
    miller_k = joblib.load('%s/miller_k_%s.jbl'%(data_path, label))
    miller_l = joblib.load('%s/miller_l_%s.jbl'%(data_path, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()

    x_r_0 = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
                                                                          mode_0))
    
    x_r_i = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
                                                                          mode_i))
    
    x_r = x_r_0 + x_r_i
    
    print x_r.shape, x_r.dtype
    
    if x_r.shape[0] != miller_h.shape[0]:
        print 'Problem'
        
    for i in range(0, x_r.shape[1], output_step):   
        
        f_out = '%s/rho_%s_mode_%d_%d_timestep_%0.6d.txt'%(fpath, 
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
        
        numpy.savetxt(f_out, out, fmt='%6d%6d%6d%20.2f%16.2f')



def get_avg_Is(settings):
    
    data_path = settings.data_path
    results_path = settings.results_path
    label = settings.label
       
    mult_factor = 100000
    
    fpath = '%s/reconstructed_intensities_mode_0_avg/'%(results_path)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        
    miller_h = joblib.load('%s/miller_h_%s.jbl'%(data_path, label))
    miller_k = joblib.load('%s/miller_k_%s.jbl'%(data_path, label))
    miller_l = joblib.load('%s/miller_l_%s.jbl'%(data_path, label))
    
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
        

def get_merged_data(x, mask, datatype):
    N = mask.sum(axis=1)
    print 'N: ', N.shape
    
    I_avg = (x.sum(axis=1))/N
    I_avg = numpy.asarray(I_avg, dtype=datatype)
    print 'I_avg:', I_avg.shape, I_avg.dtype
    
    I_avg_rep = I_avg[:, numpy.newaxis]
    I_avg_rep = numpy.repeat(I_avg_rep, x.shape[1], axis = 1)
    print 'x:', x.shape, x.dtype
    print 'I_avg_rep:', I_avg_rep.shape, I_avg_rep.dtype
    
    dI_sq = (x-I_avg_rep)*(x-I_avg_rep)
    print 'dI_sq:', dI_sq.shape
    
    s_dI_sq = dI_sq.sum(axis=1)
    
    # Crystfel definition
    sigI = (numpy.sqrt(s_dI_sq))/N
    sigI = numpy.asarray(sigI, dtype=datatype)
    print 'sigI:', sigI.shape, sigI.dtype
    
    return I_avg, sigI


def get_bin(t_uniform, t_left, t_right, x, mask, datatype):
    idxs_r = numpy.argwhere(t_uniform < t_right)
    idxs_l = numpy.argwhere(t_uniform > t_left)
    idxs = numpy.intersect1d(idxs_r, idxs_l)
    print idxs.shape, ' in bin', t_left, 'to', t_right, 'fs'   
    T = x[:, idxs]
    M = mask[:, idxs]
    print 'Bin matrices: ', T.shape, M.shape    
    I_avg, sigI = get_merged_data(T, M, datatype)
    nnans = numpy.count_nonzero(numpy.isnan(I_avg))
    print 'n of nans in I_avg: ', nnans
    nnans = numpy.count_nonzero(numpy.isnan(sigI))
    print 'n of nans in sigI: ', nnans
    I_avg[numpy.isnan(I_avg)] = 0
    sigI[numpy.isnan(sigI)] = 0
    return I_avg, sigI
    
    
def export_merged_data_light(settings):
    data_path = settings.data_path
    data_file = settings.data_file
    results_path = settings.results_path
    datatype = settings.datatype
    label = settings.label
    
    mult_factor = 100
    
    out_path = '%s/merged'%results_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        
    # EXTRACT DATA
    T_sparse = joblib.load(data_file)
    print 'T_sparse: ', T_sparse.shape
    print 'T_sparse nonzero: ', T_sparse.count_nonzero()
    x = T_sparse[:,:].todense()
    print 'x: ', x.shape, x.dtype
    x = numpy.asarray(x, dtype=datatype)
    print 'x: ', x.shape, x.dtype
    nnans = numpy.count_nonzero(numpy.isnan(x))
    print 'n of nans in x: ', nnans
    
    x = mult_factor*x
    
    M_sparse = joblib.load('%s/M_sparse_%s.jbl'%(data_path, label))
    print 'M_sparse: ', M_sparse.shape
    print 'M_sparse nonzero: ', M_sparse.count_nonzero()
    mask = M_sparse[:,:].todense()
    print 'mask: ', mask.shape, mask.dtype
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    print 'mask: ', mask.shape, mask.dtype
    
    t_uniform = joblib.load('%s/t_uniform_%s.jbl'%(data_path, label))
    t_uniform = t_uniform.flatten()
    print 't_uniform: ', t_uniform.shape
    
    miller_h = joblib.load('%s/miller_h_%s.jbl'%(data_path, label))
    miller_k = joblib.load('%s/miller_k_%s.jbl'%(data_path, label))
    miller_l = joblib.load('%s/miller_l_%s.jbl'%(data_path, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    # NEGATIVE DELAYS
    # print '\nNEGATIVE DELAYS'
    # t_left = -330
    # t_right = -100
    # I_avg, sigI = get_bin(t_uniform, t_left, t_right, x, mask, datatype)    
    # out[:, 3] = I_avg
    # out[:, 4] = sigI        
    # f_out = '%s/%s_merged_%d_to_%d_fs.txt'%(out_path, label, t_left, t_right)
    # numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  
    
    # POSITIVE DELAYS
    print '\nPOSITIVE DELAYS'
    t_left = 180
    t_right = 380
    I_avg, sigI = get_bin(t_uniform, t_left, t_right, x, mask, datatype)
    out[:, 3] = I_avg
    out[:, 4] = sigI        
    f_out = '%s/%s_merged_%d_to_%d_fs.txt'%(out_path, label, t_left, t_right)
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  
    
    # POSITIVE DELAYS (ULTRASHORT)
    # print '\nPOSITIVE DELAYS (ULTRASHORT)'
    # t_left = 0
    # t_right = 200
    # I_avg, sigI = get_bin(t_uniform, t_left, t_right, x, mask, datatype)
    # out[:, 3] = I_avg
    # out[:, 4] = sigI        
    # f_out = '%s/%s_merged_%d_to_%d_fs.txt'%(out_path, label, t_left, t_right)
    # numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  


   
def export_merged_data_dark(settings):    
    data_path = settings.data_path
    data_file = settings.data_file
    results_path = settings.results_path
    datatype = settings.datatype
    label = settings.label
    
    mult_factor = 100
    
    out_path = '%s/merged'%results_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
       
    # EXTRACT DATA
    T_sparse = joblib.load(data_file)
    print 'T_sparse: ', T_sparse.shape
    print 'T_sparse nonzero: ', T_sparse.count_nonzero()
    x = T_sparse[:,:].todense()
    print 'x: ', x.shape, x.dtype
    x = numpy.asarray(x, dtype=datatype)
    print 'x: ', x.shape, x.dtype
    nnans = numpy.count_nonzero(numpy.isnan(x))
    print 'n of nans in x: ', nnans
    
    x = mult_factor*x
    
    M_sparse = joblib.load('%s/M_sparse_%s.jbl'%(data_path, label))
    print 'M_sparse: ', M_sparse.shape
    print 'M_sparse nonzero: ', M_sparse.count_nonzero()
    mask = M_sparse[:,:].todense()
    print 'mask: ', mask.shape, mask.dtype
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    print 'mask: ', mask.shape, mask.dtype
    
    I_avg, sigI = get_merged_data(x, mask, datatype)
    
    miller_h = joblib.load('%s/miller_h_%s.jbl'%(data_path, label))
    miller_k = joblib.load('%s/miller_k_%s.jbl'%(data_path, label))
    miller_l = joblib.load('%s/miller_l_%s.jbl'%(data_path, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    out[:, 3] = I_avg
    out[:, 4] = sigI
    
    f_out = '%s/%s_merged.txt'%(out_path, label)
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  


# def get_avg_Is_negbin():
#     results_path = settings.results_path
#     mult_factor = 1000000
#     label = settings.label
    
#     fpath = '%s/reconstructed_intensities_mode_0_neg_avg/'%(results_path)
#     if not os.path.exists(fpath):
#         os.mkdir(fpath)
        
#     miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
#     miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
#     miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
#     out = numpy.zeros((miller_h.shape[0], 5))
#     out[:, 0] = miller_h.flatten()
#     out[:, 1] = miller_k.flatten()
#     out[:, 2] = miller_l.flatten()
    
#     x_r = joblib.load('%s/movie_mode_0_parallel_extended_fwd_bwd.jbl'%(results_path))
    
#     # Select negative delay bin
#     x_r = x_r[:,15000:25000]
#     I = numpy.mean(x_r, axis=1)
    
#     print I.shape
#     if I.shape[0] != miller_h.shape[0]:
#         print 'Problem'
    
#     I[numpy.isnan(I)] = 0
#     I[I<0] = 0 
#     I = mult_factor * I 
#     sigIs = numpy.sqrt(I)
        
#     out[:, 3] = I
#     out[:, 4] = sigIs
    
#     f_out = '%s/rho_%s_mode_0_neg_avg.txt'%(fpath, label)              
#     numpy.savetxt(f_out, out, fmt='%6d%6d%6d%17.2f%17.2f')




# def get_Is_bins():
#     label = settings.label 
#     mult_factor = 1000000
#     results_path = settings.results_path
#     q = settings.q
#     ncopies = settings.ncopies   
        
#     fpath = '%s/reconstructed_intensities_bins/'%(results_path)     
#     if not os.path.exists(fpath):
#             os.mkdir(fpath)
            
#     miller_h = joblib.load('../data_bR/converted_data_%s/miller_h_%s.jbl'%(label, label))
#     miller_k = joblib.load('../data_bR/converted_data_%s/miller_k_%s.jbl'%(label, label))
#     miller_l = joblib.load('../data_bR/converted_data_%s/miller_l_%s.jbl'%(label, label))
    
#     out = numpy.zeros((miller_h.shape[0], 5))
#     out[:, 0] = miller_h.flatten()
#     out[:, 1] = miller_k.flatten()
#     out[:, 2] = miller_l.flatten()
    
#     t_uniform = joblib.load('../data_bR/converted_data_%s/t_uniform_%s.jbl'%(label, label))
#     S = t_uniform.shape[0]
#     t_final = t_uniform[q:S-ncopies+1,:].flatten()
#     print t_final.shape[0], 'reconstructed frames'
    
#     mode_0 = 0
#     #mode_i = 5
#     x_r_0 = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode_0))
    
#     x_r = x_r_0
#     for mode_i in range(1, 6):   
#         print mode_i
#         x_r_i = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode_i))
#         x_r = x_r + x_r_i
        
#     #x_r = x_r_0 + x_r_i
#     print x_r.shape

#     ts = [0, 49, 94, 141, 193, 245, 314, 406, 457, 490, 518, 547, 574, 606, 646, 695, 759, 847, 946, 1025, 1201]
#     for i in range(0, len(ts)-3):
#         t1 = float(ts[i])
#         t2 = float(ts[i+3])
#         print t1, t2
        
#         idxs = numpy.argwhere((t1 <= t_final) & (t_final < t2)).flatten()
#         x_r_bin = x_r[:,idxs]
#         n_bin = x_r_bin.shape[1]
#         f_out = '%s/bR_%s_mode_all_bin_%d_%dfs_%dfs_n_%d.txt'%(fpath, label, i, t1, t2, n_bin)
#         #f_out = '%s/bR_%s_mode_%d_%d_bin_%d_%dfs_%dfs_n_%d.txt'%(fpath, label, mode_0, mode_i, i, t1, t2, n_bin)
        
#         avg_bin = x_r_bin.sum(axis=1)
#         I = avg_bin/n_bin
#         I[numpy.isnan(I)] = 0
#         I[I<0] = 0 
#         I = mult_factor * I 
#         sigIs = numpy.sqrt(I)
            
#         out[:, 3] = I
#         out[:, 4] = sigIs
            
#         numpy.savetxt(f_out, out, fmt='%6d%6d%6d%12.2f%12.2f')

# def get_info():
#     results_path = settings.results_path
    
#     S = joblib.load('%s/S.jbl'%results_path)
    
#     mode_0 = 0        
#     print 'Mode: ', mode_0
#     x_r_0 = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, 
#                                                          mode_0))
#     N = x_r_0.shape[0]*x_r_0.shape[1]
#     idxs = numpy.argwhere(x_r_0<0)
#     print 'Negative values in x_r_0: ', float(len(idxs))/N
       
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
    
# def get_Is_reweight(mode):
    
#     print 'mode ', mode
#     results_path = settings.results_path
#     mult_factor = 1000000
#     label = settings.label

#     S = joblib.load('%s/S.jbl'%results_path)
#     print S[0], S[mode]
    
#     output_step = 100
#     mode_0 = 0
#     mode_i = mode

#     fpath = '%s/reconstructed_intensities_mode_%d_%d_equalweight/'%(results_path, 
#                                                                     mode_0,
#                                                                     mode_i)
#     if not os.path.exists(fpath):
#         os.mkdir(fpath)
        
#     miller_h = joblib.load('../data_bR/converted_data_%s/miller_h_%s.jbl'%(label, 
#                                                                            label))
#     miller_k = joblib.load('../data_bR/converted_data_%s/miller_k_%s.jbl'%(label, 
#                                                                            label))
#     miller_l = joblib.load('../data_bR/converted_data_%s/miller_l_%s.jbl'%(label,
#                                                                            label))
    
#     out = numpy.zeros((miller_h.shape[0], 5))
#     out[:, 0] = miller_h.flatten()
#     out[:, 1] = miller_k.flatten()
#     out[:, 2] = miller_l.flatten()
    
#     x_r_0 = joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode_0))
#     x_r_i= joblib.load('%s/movie_mode_%d_parallel.jbl'%(results_path, mode_i))
     
#     x_r_i = x_r_i * (S[0]/S[mode_i])    
#     print 'Reweight factor: ', S[0]/S[mode_i]
#     x_r = x_r_0 + x_r_i
#     print x_r.shape

#     if x_r.shape[0] != miller_h.shape[0]:
#         print 'Problem'
            
#     for i in range(0, x_r.shape[1], output_step):            
#         print i
#         f_out = '%s/bR_%s_mode_%d_%d_timestep_%d_equalweight.txt'%(fpath, 
#                                                                    label, 
#                                                                    mode_0, 
#                                                                    mode_i, 
#                                                                    i)                                                        
#         I = x_r[:, i]
    
#         I[numpy.isnan(I)] = 0
#         I[I<0] = 0 
#         I = mult_factor * I 
#         sigIs = numpy.sqrt(I)
        
#         out[:, 3] = I
#         out[:, 4] = sigIs
        
#         numpy.savetxt(f_out, out, fmt='%6d%6d%6d%12.2f%12.2f')    
        
# def check_Is(mode):
    
#     print 'Modes 0 +', mode
#     results_path = settings.results_path
#     label = settings.label
    
#     mode_0 = 0
#     mode_i = mode  
        
#     miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
#     miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
#     miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
#     out = numpy.zeros((miller_h.shape[0], 5))
#     out[:, 0] = miller_h.flatten()
#     out[:, 1] = miller_k.flatten()
#     out[:, 2] = miller_l.flatten()

#     x_r_0 = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
#                                                                           mode_0))    
#     x_r_i = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
#                                                                           mode_i))    
#     x_r = x_r_0 + x_r_i  
#     print x_r.shape
#     for idx in range(10, 1000, 100):
#         print out[idx, 0], out[idx, 1], out[idx, 2], x_r[idx, 0]
    
# def export_binned_data():
#     #results_path = settings.results_path
#     datatype = settings.datatype
#     label = settings.label
    
#     mult_factor = 100
       
#     # EXTRACT DATA
#     T_sparse = joblib.load('../data_rho/data_converted_%s/T_sparse_%s.jbl'%(label,
#                                                                             label))
#     print 'T_sparse: ', T_sparse.shape
#     print 'T_sparse nonzero: ', T_sparse.count_nonzero()
#     x = T_sparse[:,:].todense()
#     print 'x: ', x.shape, x.dtype
#     x = numpy.asarray(x, dtype=datatype)
#     print 'x: ', x.shape, x.dtype
    
#     M_sparse = joblib.load('../data_rho/data_converted_%s/M_sparse_%s.jbl'%(label,
#                                                                             label))
#     print 'M_sparse: ', M_sparse.shape
#     print 'M_sparse nonzero: ', M_sparse.count_nonzero()
#     mask = M_sparse[:,:].todense()
#     print 'mask: ', mask.shape, mask.dtype
#     mask = numpy.asarray(mask, dtype=numpy.uint8)
#     print 'mask: ', mask.shape, mask.dtype
# #    test = mask.sum(axis = 1)
# #    idxs_test = numpy.argwhere(test<1)
# #    print 'test', idxs_test.shape
    
# #    t_uniform = joblib.load('../data_rho/data_converted_%s/t_uniform_light.jbl'%label)
# #    t_uniform = t_uniform.flatten()
# #    print t_uniform.shape
# #    
# #    idxs = numpy.argwhere(t_uniform < -100).flatten()
# #    print idxs.shape, ' in negative bin'
# #    
# #    T_neg = x[:, idxs]
# #    M_neg = mask[:, idxs]
# #    print 'Negative delays: ', T_neg.shape, M_neg.shape
# #    
# #    idxs = numpy.argwhere(t_uniform > 200).flatten()
# #    print idxs.shape, ' in positive bin'
# #    
# #    T_pos = x[:, idxs]
# #    M_pos = mask[:, idxs]
# #    print 'Positive delays: ', T_pos.shape, M_pos.shape
# #    
# #    I_neg = T_neg.sum(axis=1)/M_neg.sum(axis=1)
# #    I_pos = T_pos.sum(axis=1)/M_pos.sum(axis=1)
# #    print I_neg.shape
# #    print I_pos.shape

#     I = x.sum(axis=1)/mask.sum(axis=1)
    
#     miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
#     miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
#     miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
#     out = numpy.zeros((miller_h.shape[0], 5))
#     out[:, 0] = miller_h.flatten()
#     out[:, 1] = miller_k.flatten()
#     out[:, 2] = miller_l.flatten()
        
# #    I_neg[numpy.isnan(I_neg)] = 0
# #    I_neg[I_neg<0] = 0 
# #    I_neg = mult_factor * I_neg 
# #    sigIneg = numpy.sqrt(I_neg)
# #    
# #    out[:, 3] = I_neg
# #    out[:, 4] = sigIneg
# #    
# #    f_out = './negative_bin_mult_%d.txt'%mult_factor
# #    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%12.2f%12.2f')    
    
# #    I_pos[numpy.isnan(I_pos)] = 0
# #    I_pos[I_pos<0] = 0 
# #    I_pos = mult_factor * I_pos
# #    sigIpos = numpy.sqrt(I_pos)
# #    
# #    out[:, 3] = I_pos
# #    out[:, 4] = sigIpos
# #    
# #    f_out = './positive_bin_mult_%d.txt'%mult_factor
# #    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%12.2f%12.2f')   

#     I[numpy.isnan(I)] = 0
#     I[I<0] = 0 
#     I = mult_factor * I
#     sigI = numpy.sqrt(I)
    
#     out[:, 3] = I
#     out[:, 4] = sigI
    
#     f_out = './dark_bin_mult_%d.txt'%mult_factor
#     numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  


# def get_avg_plus_dI(mode):
    
#     print 'Avg + mode ', mode
#     results_path = settings.results_path
#     mult_factor = 1000000
#     label = settings.label    
#     output_step = 100
     
#     fpath = '%s/reconstructed_intensities_avg_plus_mode_%d/'%(results_path, 
#                                                               mode)    
#     if not os.path.exists(fpath):
#         os.mkdir(fpath)
        
#     miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
#     miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
#     miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
#     out = numpy.zeros((miller_h.shape[0], 5))
#     out[:, 0] = miller_h.flatten()
#     out[:, 1] = miller_k.flatten()
#     out[:, 2] = miller_l.flatten() 

#     folder = '../data_rho/data_converted_%s'%label
#     x_avg = joblib.load('%s/T_avg_%s.jbl'%(folder, label))    
#     x_r = joblib.load('%s/movie_mode_%d_parallel_extended_fwd_bwd.jbl'%(results_path, 
#                                                                         mode))
#     print 'x_avg: ', x_avg.shape
#     print 'x_r: ', x_r.shape
    
#     if x_r.shape[0] != miller_h.shape[0]:
#         print 'Problem'
        
#     for i in range(0, x_r.shape[1], output_step):
#         f_out = '%s/rho_%s_avg_plus_mode_%d_timestep_%0.6d.txt'%(fpath, 
#                                                                  label, 
#                                                                  mode, 
#                                                                  i)    
#         I = x_r[:, i] + x_avg
    
#         I[numpy.isnan(I)] = 0
#         I[I<0] = 0 
#         I = mult_factor * I 
#         sigIs = numpy.sqrt(I)
        
#         out[:, 3] = I
#         out[:, 4] = sigIs
        
#         numpy.savetxt(f_out, out, fmt='%6d%6d%6d%16.2f%16.2f')

# def get_avg_dark():
    
#     print 'Avg dark'
#     results_path = settings.results_path
#     mult_factor = 1000000
#     label = settings.label    
           
#     miller_h = joblib.load('../data_rho/data_converted_%s/miller_h_%s.jbl'%(label, label))
#     miller_k = joblib.load('../data_rho/data_converted_%s/miller_k_%s.jbl'%(label, label))
#     miller_l = joblib.load('../data_rho/data_converted_%s/miller_l_%s.jbl'%(label, label))
    
#     out = numpy.zeros((miller_h.shape[0], 5))
#     out[:, 0] = miller_h.flatten()
#     out[:, 1] = miller_k.flatten()
#     out[:, 2] = miller_l.flatten()
    
#     folder = '../data_rho/data_converted_%s'%label
#     I = joblib.load('%s/T_avg_%s.jbl'%(folder, label))
        
#     print 'x_avg: ', I.shape
    
#     f_out = '%s/rho_%s_avg.txt'%(results_path, label) 
        
#     I[numpy.isnan(I)] = 0
#     I[I<0] = 0 
#     I = mult_factor * I 
#     sigIs = numpy.sqrt(I)
    
#     out[:, 3] = I
#     out[:, 4] = sigIs
    
#     numpy.savetxt(f_out, out, fmt='%6d%6d%6d%16.2f%16.2f')